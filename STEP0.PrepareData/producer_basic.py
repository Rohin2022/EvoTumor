import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"

import random, socket, time, uuid
import itk, numpy as np, pandas as pd, torch, hydra
from omegaconf import DictConfig
from concurrent.futures import ThreadPoolExecutor

import unigradicon
import icon_registration.itk_wrapper as itk_wrapper

from monai.data import MetaTensor
from monai.transforms import Orientation as MonaiOrientation


NODE_ID = f"{socket.gethostname()}_{os.getpid()}"
CROP_SIZE = (175, 175, 175)


# =============================================================================
# ITK helpers
# =============================================================================
def _read_itk(path, pixel_type=itk.F):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return itk.imread(path, pixel_type=pixel_type)


def _orient_to_ras(img_itk, is_mask=False):
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

    new_affine  = t_ras.affine.numpy()
    new_spacing = np.sqrt((new_affine[:3, :3] ** 2).sum(axis=0))
    new_dir     = new_affine[:3, :3] / new_spacing
    new_origin  = new_affine[:3, 3]

    arr_out = t_ras.numpy()[0].transpose(2, 1, 0)
    arr_out = np.ascontiguousarray(arr_out)
    if is_mask:
        out = itk.image_from_array(np.round(arr_out).astype(np.uint8))
    else:
        out = itk.image_from_array(arr_out.astype(np.float32))
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
    return _resample_to_spacing(_orient_to_ras(img_itk, is_mask=is_mask), target_spacing, is_mask)


# =============================================================================
# Cropping: deterministic CROP_SIZE crop centered on mask extent
# =============================================================================
def _get_center_crop_index(img, mask, crop_size):
    size = np.array(img.GetLargestPossibleRegion().GetSize())
    mask_arr = itk.array_from_image(mask)

    nz = np.argwhere(mask_arr > 0)
    if nz.size == 0:
        center_zyx = np.array(mask_arr.shape) / 2.0
    else:
        bbox_min = nz.min(axis=0)
        bbox_max = nz.max(axis=0)
        center_zyx = (bbox_min + bbox_max) / 2.0

    center_xyz = center_zyx[::-1]

    crop = np.array(crop_size)
    start = np.round(center_xyz - crop / 2.0).astype(int)
    start = np.clip(start, 0, np.maximum(size - crop, 0))
    return start


def _pad_to_at_least(img, min_size):
    size = np.array(img.GetLargestPossibleRegion().GetSize())
    min_size = np.array(min_size)
    deficit = np.maximum(min_size - size, 0)
    if not deficit.any():
        return img
    lower = (deficit // 2).tolist()
    upper = (deficit - deficit // 2).tolist()
    pad = itk.ConstantPadImageFilter.New(img)
    pad.SetPadLowerBound([int(x) for x in lower])
    pad.SetPadUpperBound([int(x) for x in upper])
    pad.SetConstant(0)
    pad.Update()
    return pad.GetOutput()


def _crop_itk(img, start, crop_size):
    start_idx = [int(s) for s in start]

    region = itk.ImageRegion[3]()
    region.SetIndex(start_idx)
    region.SetSize([int(c) for c in crop_size])

    extractor = itk.ExtractImageFilter.New(img)
    extractor.SetExtractionRegion(region)
    extractor.Update()
    out = extractor.GetOutput()
    out.DisconnectPipeline()

    new_origin = img.TransformIndexToPhysicalPoint(start_idx)

    new_region = out.GetLargestPossibleRegion()
    new_region.SetIndex([0, 0, 0])
    out.SetRegions(new_region)
    out.SetOrigin(new_origin)

    return out


def crop_image_and_organ_mask(img, org, crop_size=CROP_SIZE):
    """Crop (img, organ_mask) to crop_size, centered on the organ mask's
    bounding-box center. Falls back to centering on the image if the organ
    mask is empty."""
    img = _pad_to_at_least(img, crop_size)
    org = _pad_to_at_least(org, crop_size)
    start = _get_center_crop_index(img, org, crop_size)
    return _crop_itk(img, start, crop_size), _crop_itk(org, start, crop_size)


# =============================================================================
# Atomic writes
# =============================================================================
def _atomic_transform_write(transform, path):
    ext = os.path.splitext(path)[1]
    tmp = os.path.join(os.path.dirname(path), f"_tmp_{uuid.uuid4().hex[:8]}{ext}")
    itk.transformwrite([transform], tmp)
    os.replace(tmp, path)


# =============================================================================
# Path helpers
# =============================================================================
def resolve_source_paths(cfg, bdmap_id, organ):
    organ_dir_name = "gall_bladder" if organ == "gallbladder" else organ
    return (
        os.path.join(cfg.dataset.data_root_path, bdmap_id, "ct.nii.gz"),
        os.path.join(cfg.dataset.organ_segmentations_root_path, bdmap_id,
                     "segmentations", f"{organ_dir_name}.nii.gz"),
        os.path.join(cfg.dataset.segmentations_root_path, bdmap_id,
                     "segmentations", f"{organ}_lesion.nii.gz"),
    )


def transform_path(transforms_dir, ct0_id, ct1_id, organ):
    # transform warps ct1 (moving) -> ct0 (fixed)
    return os.path.join(transforms_dir, f"{ct1_id}_to_{ct0_id}_{organ}.h5")


def load_manifest(cfg):
    df = pd.read_csv(cfg.dataset.datafile)
    required = {"ct0_bdmap", "ct1_bdmap", "ct0_t", "ct1_t",
                "time_delta_readable", "unix_delta", "normalized_time_delta", "organ"}

    organs_list = tumor_types = ['bladder', 'colon', 'duodenum', 'esophagus',
       'gallbladder', 
       'prostate',
       'spleen', 'stomach',
       'uterus']
    df = df[df["organ"].isin(organs_list)]
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")
    return df.to_dict(orient="records")


# =============================================================================
# In-memory volume loading (no disk caching — only transforms get written)
# =============================================================================
def build_image(cfg, bdmap_id, spacing):
    src_img_p = os.path.join(cfg.dataset.data_root_path, bdmap_id, "ct.nii.gz")
    return preprocess_itk(_read_itk(src_img_p), spacing, is_mask=False)


def build_masks(cfg, bdmap_id, organ, spacing):
    _, src_org_p, src_tum_p = resolve_source_paths(cfg, bdmap_id, organ)
    org = preprocess_itk(_read_itk(src_org_p, itk.UC), spacing, is_mask=True)
    tum = preprocess_itk(_read_itk(src_tum_p, itk.UC), spacing, is_mask=True)
    return org, tum


def build_volumes(cfg, bdmap_id, organ):
    spacing = (cfg.dataset.space_x, cfg.dataset.space_y, cfg.dataset.space_z)
    img = build_image(cfg, bdmap_id, spacing)
    org, tum = build_masks(cfg, bdmap_id, organ, spacing)
    return img, org, tum


def prefetch_pair_volumes(cfg, row):
    """Build ct0/ct1 volumes fully in memory (nothing written to disk)."""
    ct0_id, ct1_id, organ = row["ct0_bdmap"], row["ct1_bdmap"], row["organ"]
    ct0 = build_volumes(cfg, ct0_id, organ)
    ct1 = build_volumes(cfg, ct1_id, organ)
    return ct0, ct1


# =============================================================================
# Registration (transform only — no warping)
# =============================================================================
def register_pair(net, ct0_img, ct1_img, ct0_organ, ct1_organ, finetune_steps=None,
                  crop_size=CROP_SIZE):
    """Register ct1 (moving) -> ct0 (fixed), cropping each independently
    (centered on its own organ-mask bbox) to crop_size before
    preprocessing/registration. Returns the ITK transform only."""
    ct0_crop, ct0_organ_crop = crop_image_and_organ_mask(ct0_img, ct0_organ, crop_size)
    ct1_crop, ct1_organ_crop = crop_image_and_organ_mask(ct1_img, ct1_organ, crop_size)

    ct0_pre = unigradicon.preprocess(ct0_crop, modality="ct")
    ct1_pre = unigradicon.preprocess(ct1_crop, modality="ct")
    phi, _ = itk_wrapper.register_pair_with_mask(
        net, ct1_pre, ct0_pre,
        finetune_steps=finetune_steps,
    )
    return phi


# =============================================================================
# Main loop
# =============================================================================
def run_producer_loop(cfg: DictConfig):
    print(f"[{NODE_ID}] Starting producer (transforms only, no volume caching)", flush=True)

    weights_location = cfg.producer.get("unigradicon_weights", None)
    if weights_location is None:
        raise ValueError(
            "cfg.producer.unigradicon_weights must be set to the pretrained "
            "checkpoint path (e.g. 'network_weights/network_weights_9500.trch'). "
            "Loading without this arg can silently use different/uninitialized "
            "weights, which produces degenerate, flat similarity scores."
        )
    net = unigradicon.get_unigradicon(weights_location=weights_location)
    net.cuda().eval()

    scratch_dir    = cfg.producer.scratch_dir
    transforms_dir = os.path.join(scratch_dir, "transforms")
    os.makedirs(transforms_dir, exist_ok=True)

    all_rows = load_manifest(cfg)
    print(f"[{NODE_ID}] Loaded {len(all_rows)} pairs", flush=True)
    random.shuffle(all_rows)

    max_pairs = cfg.producer.get("max_pairs", cfg.producer.get("max_batches", len(all_rows)))
    finetune_steps = cfg.producer.get("registration_finetune_steps", None)

    processed = 0
    consecutive_failures = 0

    executor = ThreadPoolExecutor(max_workers=1)

    def submit_prefetch(row):
        return executor.submit(prefetch_pair_volumes, cfg, row)

    def already_done(row):
        out_path = transform_path(transforms_dir, row["ct0_bdmap"], row["ct1_bdmap"], row["organ"])
        return os.path.exists(out_path)

    rows_to_process = [r for r in all_rows if not already_done(r)]
    skipped_existing = len(all_rows) - len(rows_to_process)
    print(f"[{NODE_ID}] {skipped_existing} pairs already have transforms, "
          f"{len(rows_to_process)} remaining", flush=True)

    rows_to_process = rows_to_process[:max_pairs]
    if not rows_to_process:
        print(f"[{NODE_ID}] Nothing to do. Done.", flush=True)
        return

    rows_iter = iter(rows_to_process)
    prefetch_row = next(rows_iter)
    prefetch_future = submit_prefetch(prefetch_row)

    def handle_row(row, volumes):
        nonlocal processed, consecutive_failures
        out_path = transform_path(transforms_dir, row["ct0_bdmap"], row["ct1_bdmap"], row["organ"])
        try:
            t0 = time.time()
            (ct0_img, ct0_org, _ct0_tum), (ct1_img, ct1_org, _ct1_tum) = volumes
            phi = register_pair(net, ct0_img, ct1_img, ct0_org, ct1_org,
                                finetune_steps=finetune_steps)
            t1 = time.time()
            _atomic_transform_write(phi, out_path)
            processed += 1
            print(f"[{NODE_ID}] Saved transform {os.path.basename(out_path)} "
                  f"| registration={t1-t0:.1f}s | total={processed}/{len(rows_to_process)}",
                  flush=True)
            consecutive_failures = 0
        except Exception as e:
            consecutive_failures += 1
            print(f"[{NODE_ID}] FAILED ct0={row.get('ct0_bdmap')} "
                  f"ct1={row.get('ct1_bdmap')} organ={row.get('organ')}: {e}", flush=True)
            if consecutive_failures >= 10:
                raise RuntimeError("10 consecutive failures — storage may be offline.")

    for next_row in rows_iter:
        next_future = submit_prefetch(next_row)
        volumes = prefetch_future.result()
        handle_row(prefetch_row, volumes)
        prefetch_future, prefetch_row = next_future, next_row

    # last row has no successor to prefetch
    volumes = prefetch_future.result()
    handle_row(prefetch_row, volumes)

    print(f"[{NODE_ID}] Done. Processed {processed} pairs "
          f"({skipped_existing} already existed).", flush=True)


@hydra.main(config_path='config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
    run_producer_loop(cfg)


if __name__ == "__main__":
    run()