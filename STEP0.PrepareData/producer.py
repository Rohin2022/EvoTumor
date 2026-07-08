import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"

import time, random, uuid, socket, os, time
import torch, hydra, itk, numpy as np, pandas as pd
from omegaconf import DictConfig
from concurrent.futures import ThreadPoolExecutor

import unigradicon
import icon_registration.itk_wrapper as itk_wrapper

from monai.transforms import (
    Compose, CropForegroundd, SpatialPadd, RandCropByLabelClassesd,
    RandScaleIntensityd, RandShiftIntensityd, Lambdad, ToTensord,
)
from monai.transforms.transform import MapTransform
from monai.data import MetaTensor
from monai.transforms import Orientation as MonaiOrientation
from vq_gan_3d.model.vqgan import VQGAN


# =============================================================================
# Constants
# =============================================================================
BACKGROUND, ORGAN, TUMOR = 0, 1, 2
NODE_ID = f"{socket.gethostname()}_{os.getpid()}"


# =============================================================================
# ITK helpers
# =============================================================================
def _read_itk(path, pixel_type=itk.F):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return itk.imread(path, pixel_type=pixel_type)


def _orient_to_ras(img_itk):
    arr = itk.array_from_image(img_itk)
    spacing   = np.array(img_itk.GetSpacing())
    origin    = np.array(img_itk.GetOrigin())
    direction = np.array(img_itk.GetDirection()).reshape(3, 3)

    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = direction * spacing
    affine[:3,  3] = origin

    t = MetaTensor(
        torch.from_numpy(arr.transpose(2, 1, 0)[np.newaxis].copy()).float(),
        affine=torch.from_numpy(affine).float(),
    )
    t_ras = MonaiOrientation(axcodes="RAS")(t)

    new_affine   = t_ras.affine.numpy()
    new_spacing  = np.sqrt((new_affine[:3, :3] ** 2).sum(axis=0))
    new_dir      = new_affine[:3, :3] / new_spacing
    new_origin   = new_affine[:3, 3]

    arr_out = t_ras.numpy()[0].transpose(2, 1, 0)
    out = itk.image_from_array(np.ascontiguousarray(arr_out).astype(np.float32))
    out.SetSpacing(new_spacing.tolist())
    out.SetOrigin(new_origin.tolist())
    out.SetDirection(itk.matrix_from_array(new_dir))
    return out


def _resample_to_spacing(img_itk, target_spacing, is_mask=False):
    orig_spacing = np.array(img_itk.GetSpacing())
    orig_size    = np.array(img_itk.GetLargestPossibleRegion().GetSize())
    target       = np.array(target_spacing)
    new_size     = [int(s) for s in np.round(orig_size * orig_spacing / target)]

    r = itk.ResampleImageFilter.New(img_itk)
    r.SetOutputSpacing(target.tolist())
    r.SetSize(new_size)
    r.SetOutputDirection(img_itk.GetDirection())
    r.SetOutputOrigin(img_itk.GetOrigin())
    r.SetDefaultPixelValue(0)
    r.SetInterpolator(
        itk.NearestNeighborInterpolateImageFunction.New(img_itk) if is_mask
        else itk.LinearInterpolateImageFunction.New(img_itk)
    )
    r.Update()
    return r.GetOutput()


def preprocess_itk(img_itk, target_spacing, is_mask=False):
    return _resample_to_spacing(_orient_to_ras(img_itk), target_spacing, is_mask)


def _warp_image(moving, fixed, transform):
    moving_cast, cast_back = unigradicon.maybe_cast(moving)
    warped = itk.resample_image_filter(
        moving_cast, transform=transform,
        interpolator=itk.LinearInterpolateImageFunction.New(moving_cast),
        use_reference_image=True, reference_image=fixed,
    )
    return cast_back(warped)


def _warp_mask(mask, fixed, transform):
    r = itk.ResampleImageFilter.New(mask)
    r.SetInterpolator(itk.NearestNeighborInterpolateImageFunction.New(mask))
    r.SetTransform(transform)
    r.SetSize(fixed.GetLargestPossibleRegion().GetSize())
    r.SetOutputSpacing(fixed.GetSpacing())
    r.SetOutputOrigin(fixed.GetOrigin())
    r.SetOutputDirection(fixed.GetDirection())
    r.SetDefaultPixelValue(0)
    r.Update()
    warped = r.GetOutput()
    arr = (itk.array_from_image(warped) > 0).astype(np.uint8)
    out = itk.image_from_array(arr)
    out.SetSpacing(fixed.GetSpacing())
    out.SetOrigin(fixed.GetOrigin())
    out.SetDirection(fixed.GetDirection())
    return out


# =============================================================================
# Registration
# =============================================================================
def register_and_warp(net, ct0_img, ct1_img, ct0_organ, ct1_organ, ct1_tumor,
                      finetune_steps=None):
    """Register ct1 (moving) → ct0 (fixed) using register_pair_with_mask so
    the model's similarity loss is masked to the organ region on both sides.
    The organ masks are passed as mask_A / mask_B and resized internally by
    itk_wrapper to the network's identity_map shape.
    """
    ct0_pre = unigradicon.preprocess(ct0_img, modality="ct")
    ct1_pre = unigradicon.preprocess(ct1_img, modality="ct")

    phi, _ = itk_wrapper.register_pair_with_mask(
        net,
        ct1_pre,          # moving (A)
        ct0_pre,          # fixed  (B)
        mask_A=ct1_organ,
        mask_B=ct0_organ,
        finetune_steps=None,
    )
    return (
        _warp_image(ct1_img, ct0_img, phi),
        _warp_mask(ct1_organ, ct0_img, phi),
        _warp_mask(ct1_tumor, ct0_img, phi),
    )


# =============================================================================
# MONAI pipeline
# =============================================================================
class CombineMasksToTernaryd(MapTransform):
    def __init__(self, organ_key="organ_mask_ct0", tumor_key="tumor_mask_ct0",
                 output_key="label", allow_missing_keys=False):
        super().__init__(keys=[organ_key, tumor_key], allow_missing_keys=allow_missing_keys)
        self.organ_key, self.tumor_key, self.output_key = organ_key, tumor_key, output_key

    def __call__(self, data):
        d = dict(data)
        label = (d[self.organ_key] > 0.5).to(torch.int64)
        label[d[self.tumor_key] > 0.5] = 2
        d[self.output_key] = label
        d.pop(self.organ_key, None)
        d.pop(self.tumor_key, None)
        return d


def build_crop_pipeline(cfg):
    roi = (cfg.dataset.roi_x, cfg.dataset.roi_y, cfg.dataset.roi_z)
    crop_keys = ["image_ct0", "image_ct1", "tumor_ct0", "tumor_ct1", "label"]
    return Compose([
        CombineMasksToTernaryd(),
        CropForegroundd(keys=crop_keys, source_key="label",
                        select_fn=lambda x: x > 0, margin=64, allow_smaller=False),
        SpatialPadd(keys=crop_keys, spatial_size=roi, mode="constant"),
        RandCropByLabelClassesd(
            keys=crop_keys, label_key="label", spatial_size=roi,
            ratios=[0, 1, 1], num_classes=3,
            num_samples=cfg.producer.num_samples,
            image_key="image_ct0", image_threshold=-1,
        ),
        RandScaleIntensityd(keys=["image_ct0", "image_ct1"], factors=0.1, prob=0.5),
        RandShiftIntensityd(keys=["image_ct0", "image_ct1"], offsets=0.1, prob=0.5),
        Lambdad(keys=["image_ct0", "image_ct1"],
                func=lambda x: torch.clamp(x, -1.0, 1.0)),
        ToTensord(keys=crop_keys),
    ])


# =============================================================================
# VQGAN
# =============================================================================
def load_vqgan(ckpt):
    m = VQGAN.load_from_checkpoint(ckpt, weights_only=False).cuda().eval()
    with torch.no_grad():
        emb_min = m.codebook.embeddings.min()
        emb_max = m.codebook.embeddings.max()
    return m, emb_min, emb_max


@torch.no_grad()
def encode_batch(vqgan, imgs, emb_min, emb_max):
    """imgs: (B, 1, X, Y, Z) → normalized latent (B, C, ...)"""
    imgs = imgs.permute(0, 1, 4, 2, 3)  # → (B,1,Z,X,Y)
    z = vqgan.encode(imgs, quantize=False, include_embeddings=True)
    return ((z - emb_min) / (emb_max - emb_min)) * 2.0 - 1.0


def encode_crops_batched(vqgan, emb_min, emb_max, crops, vqgan_chunk_size=3):
    """
    Encode all ct0/ct1 crops in chunks of vqgan_chunk_size.
    num_samples (e.g. 8) can exceed vqgan_chunk_size (e.g. 3) — crops are
    split into ceil(num_samples / vqgan_chunk_size) forward passes.
    Returns z0s, z1s: (N, C, ...) bfloat16 CPU tensors.
    """
    n = len(crops)
    img0s = torch.stack([c["image_ct0"].as_tensor() for c in crops]).cuda()  # (N,1,X,Y,Z)
    img1s = torch.stack([c["image_ct1"].as_tensor() for c in crops]).cuda()

    z0s, z1s = [], []
    for i in range(0, n, vqgan_chunk_size):
        z0s.append(encode_batch(vqgan, img0s[i:i+vqgan_chunk_size], emb_min, emb_max))
        z1s.append(encode_batch(vqgan, img1s[i:i+vqgan_chunk_size], emb_min, emb_max))

    z0s = torch.cat(z0s, dim=0).to(torch.bfloat16).cpu()
    z1s = torch.cat(z1s, dim=0).to(torch.bfloat16).cpu()
    return z0s, z1s


# =============================================================================
# Path / manifest helpers
# =============================================================================
def resolve_paths(cfg, bdmap_id, organ):
    organ_dir_name = "gall_bladder" if organ == "gallbladder" else organ
    return (
        os.path.join(cfg.dataset.data_root_path, bdmap_id, "ct.nii.gz"),
        os.path.join(cfg.dataset.organ_segmentations_root_path, bdmap_id,
                     "segmentations", f"{organ_dir_name}.nii.gz"),
        os.path.join(cfg.dataset.segmentations_root_path, bdmap_id,
                     "segmentations", f"{organ}_lesion.nii.gz"),
    )


def load_manifest(cfg):
    df = pd.read_csv(cfg.dataset.datafile)
    required = {"ct0_bdmap","ct1_bdmap","ct0_t","ct1_t",
                "time_delta_readable","unix_delta","normalized_time_delta","organ"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")
    return df.to_dict(orient="records")


# =============================================================================
# Per-pair processing
# =============================================================================
def _itk_to_tensor(img_itk):
    arr = itk.array_from_image(img_itk).astype(np.float32)
    return torch.from_numpy(arr).unsqueeze(0)  # (1, D, H, W)


def _scale(x, a_min, a_max, b_min, b_max):
    return torch.clamp(x, a_min, a_max).sub_(a_min).div_(a_max - a_min).mul_(b_max - b_min).add_(b_min)


def load_and_preprocess_pair(cfg, row):
    """Load + orient + resample one pair (can be run in a thread)."""
    ct0_id, ct1_id, organ = row["ct0_bdmap"], row["ct1_bdmap"], row["organ"]
    spacing = (cfg.dataset.space_x, cfg.dataset.space_y, cfg.dataset.space_z)

    ct0_img_p, ct0_org_p, ct0_tum_p = resolve_paths(cfg, ct0_id, organ)
    ct1_img_p, ct1_org_p, ct1_tum_p = resolve_paths(cfg, ct1_id, organ)

    def load_all(img_p, org_p, tum_p):
        img = preprocess_itk(_read_itk(img_p), spacing, is_mask=False)
        org = preprocess_itk(_read_itk(org_p, itk.UC), spacing, is_mask=True)
        tum = preprocess_itk(_read_itk(tum_p, itk.UC), spacing, is_mask=True)
        return img, org, tum

    return load_all(ct0_img_p, ct0_org_p, ct0_tum_p), load_all(ct1_img_p, ct1_org_p, ct1_tum_p)


def _process_loaded(cfg, net, crop_pipeline, row, loaded):
    """Register + crop a pair whose ITK volumes are already loaded."""
    (ct0_img, ct0_org, ct0_tum), (ct1_img, ct1_org, ct1_tum) = loaded

    t0 = time.time()
    ct1_img_w, _ct1_org_w, ct1_tum_w = register_and_warp(
        net, ct0_img, ct1_img, ct0_org, ct1_org, ct1_tum,
        finetune_steps=cfg.producer.get("registration_finetune_steps", None),
    )
    t1 = time.time()

    a_min, a_max = cfg.dataset.a_min, cfg.dataset.a_max
    b_min, b_max = cfg.dataset.b_min, cfg.dataset.b_max

    data = {
        "image_ct0":      _scale(_itk_to_tensor(ct0_img),   a_min, a_max, b_min, b_max),
        "image_ct1":      _scale(_itk_to_tensor(ct1_img_w), a_min, a_max, b_min, b_max),
        "organ_mask_ct0": _itk_to_tensor(ct0_org),
        "tumor_mask_ct0": _itk_to_tensor(ct0_tum),
        "tumor_ct0":      _itk_to_tensor(ct0_tum),
        "tumor_ct1":      _itk_to_tensor(ct1_tum_w),
    }
    crops = crop_pipeline(data)
    t2 = time.time()

    print(f"[{NODE_ID}] TIMING  registration={t1-t0:.1f}s  crops={t2-t1:.1f}s", flush=True)

    crops = crops if isinstance(crops, list) else [crops]
    return crops, row


# =============================================================================
# Save
# =============================================================================
def save_batch(batch_dict, scratch_dir, idx):
    fname = f"batch_{idx:06d}_{NODE_ID}_{uuid.uuid4().hex[:8]}"
    tmp   = os.path.join(scratch_dir, f"_{fname}.pt.tmp")
    final = os.path.join(scratch_dir, f"{fname}.pt")
    torch.save(batch_dict, tmp)
    os.rename(tmp, final)  # atomic on POSIX filesystems
    return final


# =============================================================================
# Main loop
# =============================================================================
def run_producer_loop(cfg: DictConfig):
    print(f"[{NODE_ID}] Starting producer", flush=True)

    net = unigradicon.get_unigradicon()
    net.cuda().eval()

    vqgan, emb_min, emb_max = load_vqgan(cfg.producer.vqgan_ckpt)
    crop_pipeline = build_crop_pipeline(cfg)
    scratch_dir = cfg.producer.scratch_dir
    os.makedirs(scratch_dir, exist_ok=True)

    existing = len([f for f in os.listdir(scratch_dir)
                    if f.endswith(".pt") and not f.startswith("_")])
    print(f"[{NODE_ID}] Found {existing} existing .pt files", flush=True)

    batches_saved    = existing
    max_batches      = cfg.producer.max_batches
    chunk_size       = cfg.producer.chunk_size
    vqgan_chunk_size = cfg.producer.get("vqgan_chunk_size", 3)

    all_rows = load_manifest(cfg)
    print(f"[{NODE_ID}] Loaded {len(all_rows)} pairs", flush=True)

    buffer = []
    consecutive_failures = 0
    last_save_time = time.time()

    # Prefetch: load+preprocess the next pair on a background thread while
    # the current pair is being registered (CPU/disk vs GPU overlap).
    executor = ThreadPoolExecutor(max_workers=1)

    def submit_load(row):
        return executor.submit(load_and_preprocess_pair, cfg, row)

    while batches_saved < max_batches:
        random.shuffle(all_rows)
        rows_iter = iter(all_rows)

        # Prime the prefetch queue with the first row
        try:
            first_row = next(rows_iter)
        except StopIteration:
            break
        prefetch_future = submit_load(first_row)
        prefetch_row    = first_row

        for next_row in rows_iter:
            if batches_saved >= max_batches:
                break

            row = prefetch_row
            # Start loading the next pair immediately (overlaps with registration below)
            next_future     = submit_load(next_row)

            try:
                loaded = prefetch_future.result()  # blocks only if not ready yet
                crops, row = _process_loaded(cfg, net, crop_pipeline, row, loaded)
                z0s, z1s = encode_crops_batched(vqgan, emb_min, emb_max, crops,
                                               vqgan_chunk_size=vqgan_chunk_size)
                for i, crop in enumerate(crops):
                    buffer.append({
                        "residual":              (z1s[i] - z0s[i]),
                        "z0":                    z0s[i],
                        "normalized_time_delta": torch.tensor(float(row["normalized_time_delta"]),
                                                              dtype=torch.float32),
                        "organ":      row["organ"],
                        "tumor_ct0":  crop["tumor_ct0"].as_tensor().to(torch.uint8),
                        "tumor_ct1_warped": crop["tumor_ct1"].as_tensor().to(torch.uint8),
                        "ct0_bdmap":  row["ct0_bdmap"],
                        "ct1_bdmap":  row["ct1_bdmap"],
                    })
                consecutive_failures = 0

            except Exception as e:
                consecutive_failures += 1
                print(f"[{NODE_ID}] FAILED ct0={row.get('ct0_bdmap')} "
                      f"ct1={row.get('ct1_bdmap')} organ={row.get('organ')}: {e}", flush=True)
                if consecutive_failures >= 10:
                    raise RuntimeError("10 consecutive failures — storage may be offline.")

            prefetch_future = next_future
            prefetch_row    = next_row

        # Process the last row in the epoch (no next_row to prefetch after it)
        if batches_saved < max_batches:
            row = prefetch_row
            try:
                loaded = prefetch_future.result()
                crops, row = _process_loaded(cfg, net, crop_pipeline, row, loaded)
                z0s, z1s = encode_crops_batched(vqgan, emb_min, emb_max, crops,
                                               vqgan_chunk_size=vqgan_chunk_size)
                for i, crop in enumerate(crops):
                    buffer.append({
                        "residual":              (z1s[i] - z0s[i]),
                        "z0":                    z0s[i],
                        "normalized_time_delta": torch.tensor(float(row["normalized_time_delta"]),
                                                              dtype=torch.float32),
                        "organ":      row["organ"],
                        "tumor_ct0":  crop["tumor_ct0"].as_tensor().to(torch.uint8),
                        "tumor_ct1_warped": crop["tumor_ct1"].as_tensor().to(torch.uint8),
                        "ct0_bdmap":  row["ct0_bdmap"],
                        "ct1_bdmap":  row["ct1_bdmap"],
                    })
                consecutive_failures = 0
            except Exception as e:
                consecutive_failures += 1
                print(f"[{NODE_ID}] FAILED ct0={row.get('ct0_bdmap')} "
                      f"ct1={row.get('ct1_bdmap')} organ={row.get('organ')}: {e}", flush=True)
                if consecutive_failures >= 10:
                    raise RuntimeError("10 consecutive failures — storage may be offline.")

            while len(buffer) >= chunk_size:
                chunk, buffer = buffer[:chunk_size], buffer[chunk_size:]
                batch = {k: torch.stack([d[k] for d in chunk])
                         if isinstance(chunk[0][k], torch.Tensor) else [d[k] for d in chunk]
                         for k in chunk[0]}

                batches_saved += 1
                path = save_batch(batch, scratch_dir, batches_saved)
                now  = time.time()
                dt   = now - last_save_time
                last_save_time = now
                print(f"[{NODE_ID}] Saved {os.path.basename(path)} "
                      f"| total={batches_saved}/{max_batches} "
                      f"| Δt={dt:.1f}s | {chunk_size/dt:.0f} samples/s", flush=True)

    print(f"[{NODE_ID}] Done. Total: {batches_saved}", flush=True)


@hydra.main(config_path='config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
    run_producer_loop(cfg)


if __name__ == "__main__":
    run()