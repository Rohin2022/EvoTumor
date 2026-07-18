import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"

import json, random, socket, time, uuid
from datetime import datetime
from pathlib import Path

import itk, numpy as np, pandas as pd, torch, hydra
from omegaconf import DictConfig

import unigradicon
import icon_registration.itk_wrapper as itk_wrapper

from monai.data import MetaTensor
from monai.transforms import Orientation as MonaiOrientation


NODE_ID = f"{socket.gethostname()}_{os.getpid()}"
CROP_SIZE = (175, 175, 175)


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def existing_file(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


# =============================================================================
# ITK helpers (preprocessing only — unrelated to registration)
# =============================================================================
def _read_itk(path, pixel_type=itk.F):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return itk.imread(path, pixel_type=pixel_type)


def _read_itk_safe(path, pixel_type=itk.F):
    """Like _read_itk but returns None instead of raising when the path is
    missing, empty, or fails to load (corrupt/invalid file)."""
    if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    try:
        return itk.imread(path, pixel_type=pixel_type)
    except Exception:  # noqa: BLE001 - any read failure counts as "unavailable"
        return None


def _zero_mask_like(reference_img):
    """Build an all-zero uint8 mask sharing reference_img's grid (size,
    spacing, origin, direction). Used as a stand-in organ/lesion mask when
    no mask file is available or the file is invalid — downstream cropping
    already falls back to centering on the image when a mask is empty."""
    size = itk.size(reference_img)
    arr = np.zeros((size[2], size[1], size[0]), dtype=np.uint8)
    mask = itk.image_from_array(arr)
    mask.SetSpacing(reference_img.GetSpacing())
    mask.SetOrigin(reference_img.GetOrigin())
    mask.SetDirection(reference_img.GetDirection())
    return mask


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


def _crop_like(img, reference_img, reference_org, crop_size=CROP_SIZE):
    """Crop img to the same window used for (reference_img, reference_org),
    e.g. to crop a lesion mask to the same window as its own scan's
    image/organ-mask crop."""
    img = _pad_to_at_least(img, crop_size)
    ref_img_padded = _pad_to_at_least(reference_img, crop_size)
    ref_org_padded = _pad_to_at_least(reference_org, crop_size)
    start = _get_center_crop_index(ref_img_padded, ref_org_padded, crop_size)
    return _crop_itk(img, start, crop_size)


# =============================================================================
# Atomic writes
# =============================================================================
def _atomic_transform_write(transform, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f"_tmp_{uuid.uuid4().hex[:8]}{path.suffix}"
    itk.transformwrite([transform], str(tmp))
    tmp.replace(path)


def _atomic_image_write(img, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f"_tmp_{uuid.uuid4().hex[:8]}{''.join(path.suffixes)}"
    itk.imwrite(img, str(tmp))
    tmp.replace(path)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


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


def pair_id_str(ct0_id, ct1_id, organ):
    # transform warps ct1 (moving) -> ct0 (fixed)
    return f"{ct1_id}_to_{ct0_id}_{organ}"


def load_manifest(cfg):
    df = pd.read_csv(cfg.dataset.datafile)
    required = {"ct0_bdmap", "ct1_bdmap", "ct0_t", "ct1_t",
                "time_delta_readable", "unix_delta", "normalized_time_delta", "organ"}

    organs_list = ['bladder', 'colon', 'duodenum', 'esophagus',
       'gallbladder',
       'prostate',
       'spleen', 'stomach',
       'uterus']
    #df = df[df["organ"].isin(organs_list)]
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")
    return df.to_dict(orient="records")


# =============================================================================
# In-memory volume loading (no disk caching of source volumes)
# =============================================================================
def build_image(cfg, bdmap_id, spacing):
    src_img_p = os.path.join(cfg.dataset.data_root_path, bdmap_id, "ct.nii.gz")
    return preprocess_itk(_read_itk(src_img_p), spacing, is_mask=False)


def build_masks(cfg, bdmap_id, organ, spacing, image_ref):
    """Load + preprocess the organ and lesion masks for bdmap_id. If either
    file is missing or fails to load, substitute an all-zero mask on
    image_ref's grid instead of raising — downstream cropping already
    treats an empty mask as "center on the image", and warping steps that
    need a real lesion mask are skipped by the caller when unavailable."""
    _, src_org_p, src_tum_p = resolve_source_paths(cfg, bdmap_id, organ)

    org_raw = _read_itk_safe(src_org_p, itk.UC)
    org_available = org_raw is not None
    org = preprocess_itk(org_raw, spacing, is_mask=True) if org_available else _zero_mask_like(image_ref)

    tum_raw = _read_itk_safe(src_tum_p, itk.UC)
    lesion_available = tum_raw is not None
    tum = preprocess_itk(tum_raw, spacing, is_mask=True) if lesion_available else _zero_mask_like(image_ref)

    return org, tum, org_available, lesion_available


def build_volumes(cfg, bdmap_id, organ):
    spacing = (cfg.dataset.space_x, cfg.dataset.space_y, cfg.dataset.space_z)
    img = build_image(cfg, bdmap_id, spacing)
    org, tum, org_available, lesion_available = build_masks(cfg, bdmap_id, organ, spacing, img)
    return img, org, tum, org_available, lesion_available


# =============================================================================
# Registration + warp — same structure/status-tracking as the CLI producer,
# but calling the unigradicon Python package directly so a custom local
# checkpoint (weights_location) can be used instead of a named built-in model.
# =============================================================================
def register_pair(net, ct0_crop, ct1_crop, finetune_steps=None):
    """Register ct1_crop (moving) -> ct0_crop (fixed). Returns the ITK
    transform (phi) mapping moving -> fixed."""
    ct0_pre = unigradicon.preprocess(ct0_crop, modality="ct")
    ct1_pre = unigradicon.preprocess(ct1_crop, modality="ct")
    phi, _ = itk_wrapper.register_pair_with_mask(
        net, ct1_pre, ct0_pre,
        finetune_steps=finetune_steps,
    )
    return phi


def warp_mask(fixed_reference_img, moving_mask, phi):
    """Warp moving_mask into fixed_reference_img's grid using transform phi,
    with nearest-neighbor interpolation (mask-appropriate)."""
    resampler = itk.ResampleImageFilter.New(moving_mask)
    resampler.SetReferenceImage(fixed_reference_img)
    resampler.UseReferenceImageOn()
    resampler.SetTransform(phi)
    resampler.SetInterpolator(
        itk.NearestNeighborInterpolateImageFunction.New(moving_mask)
    )
    resampler.SetDefaultPixelValue(0)
    resampler.Update()
    return resampler.GetOutput()


def warp_image(fixed_reference_img, moving_img, phi):
    """Warp moving_img (intensity CT) into fixed_reference_img's grid using
    transform phi, with linear interpolation (image-appropriate) — mirrors
    unigradicon-register's --warped_moving_out in the CLI version."""
    resampler = itk.ResampleImageFilter.New(moving_img)
    resampler.SetReferenceImage(fixed_reference_img)
    resampler.UseReferenceImageOn()
    resampler.SetTransform(phi)
    resampler.SetInterpolator(
        itk.LinearInterpolateImageFunction.New(moving_img)
    )
    resampler.SetDefaultPixelValue(0)
    resampler.Update()
    return resampler.GetOutput()


def process_pair(row, cfg, net, transforms_dir: Path, crops_root: Path, statuses_dir: Path,
                  finetune_steps, overwrite: bool, dry_run: bool) -> dict:
    ct0_id, ct1_id, organ = row["ct0_bdmap"], row["ct1_bdmap"], row["organ"]
    pid = pair_id_str(ct0_id, ct1_id, organ)

    crop_dir = crops_root / pid
    transform_path = transforms_dir / f"{pid}.h5"
    warped_ct_path = transforms_dir / f"{pid}_ct1_warped_to_ct0.nii.gz"
    warped_lesion_path = transforms_dir / f"{pid}_ct1_lesion_warped_to_ct0.nii.gz"
    status_path = statuses_dir / f"{pid}.json"

    crop_paths = {
        "ct0_ct_crop": crop_dir / f"{ct0_id}_ct_crop.nii.gz",
        "ct0_organ_crop": crop_dir / f"{ct0_id}_{organ}_crop.nii.gz",
        "ct1_ct_crop": crop_dir / f"{ct1_id}_ct_crop.nii.gz",
        "ct1_organ_crop": crop_dir / f"{ct1_id}_{organ}_crop.nii.gz",
        "ct1_lesion_crop": crop_dir / f"{ct1_id}_{organ}_lesion_crop.nii.gz",
    }

    payload = {
        "pair_id": pid,
        "ct0_bdmap": ct0_id,
        "ct1_bdmap": ct1_id,
        "organ": organ,
        "transform": str(transform_path),
        "warped_ct": str(warped_ct_path),
        "warped_lesion": str(warped_lesion_path),
        **{k: str(v) for k, v in crop_paths.items()},
        "started_at": now(),
        "status": "started",
    }

    if existing_file(transform_path) and existing_file(warped_ct_path) and not overwrite:
        payload["status"] = "skipped_existing"
        payload["finished_at"] = now()
        write_json(status_path, payload)
        return payload

    try:
        ct0_img, ct0_org, _ct0_tum, ct0_org_available, _ct0_lesion_available = build_volumes(cfg, ct0_id, organ)
        ct1_img, ct1_org, ct1_tum, ct1_org_available, ct1_lesion_available = build_volumes(cfg, ct1_id, organ)

        payload["ct0_organ_mask_available"] = ct0_org_available
        payload["ct1_organ_mask_available"] = ct1_org_available
        payload["ct1_lesion_mask_available"] = ct1_lesion_available
        if not ct0_org_available:
            payload.setdefault("warnings", []).append(
                f"ct0 organ mask missing/invalid for {ct0_id}/{organ}; cropped on image center."
            )
        if not ct1_org_available:
            payload.setdefault("warnings", []).append(
                f"ct1 organ mask missing/invalid for {ct1_id}/{organ}; cropped on image center."
            )
        if not ct1_lesion_available:
            payload.setdefault("warnings", []).append(
                f"ct1 lesion mask missing/invalid for {ct1_id}/{organ}; lesion crop/warp skipped."
            )

        ct0_img_crop, ct0_org_crop = crop_image_and_organ_mask(ct0_img, ct0_org)
        ct1_img_crop, ct1_org_crop = crop_image_and_organ_mask(ct1_img, ct1_org)

        # write the crop volumes to disk, same file set the CLI producer's
        # "prepare" step writes, for parity/debugging.
        _atomic_image_write(ct0_img_crop, crop_paths["ct0_ct_crop"])
        _atomic_image_write(ct0_org_crop, crop_paths["ct0_organ_crop"])
        _atomic_image_write(ct1_img_crop, crop_paths["ct1_ct_crop"])
        _atomic_image_write(ct1_org_crop, crop_paths["ct1_organ_crop"])

        ct1_tum_crop = None
        if ct1_lesion_available:
            ct1_tum_crop = _crop_like(ct1_tum, ct1_img, ct1_org)
            _atomic_image_write(ct1_tum_crop, crop_paths["ct1_lesion_crop"])
    except Exception as exc:  # noqa: BLE001 - keep the loop going, record failure
        payload["status"] = "prepare_failed"
        payload["error"] = str(exc)
        payload["finished_at"] = now()
        write_json(status_path, payload)
        return payload

    if dry_run:
        payload["status"] = "dry_run"
        payload["finished_at"] = now()
        write_json(status_path, payload)
        return payload

    try:
        phi = register_pair(net, ct0_img_crop, ct1_img_crop, finetune_steps=finetune_steps)
    except Exception as exc:  # noqa: BLE001
        payload["status"] = "register_failed"
        payload["error"] = str(exc)
        payload["finished_at"] = now()
        write_json(status_path, payload)
        return payload

    try:
        _atomic_transform_write(phi, transform_path)
    except Exception as exc:  # noqa: BLE001
        payload["status"] = "transform_write_failed"
        payload["error"] = str(exc)
        payload["finished_at"] = now()
        write_json(status_path, payload)
        return payload

    try:
        warped_ct = warp_image(ct0_img_crop, ct1_img_crop, phi)
        _atomic_image_write(warped_ct, warped_ct_path)
    except Exception as exc:  # noqa: BLE001
        payload["status"] = "warp_ct_failed"
        payload["error"] = str(exc)
        payload["finished_at"] = now()
        write_json(status_path, payload)
        return payload

    if ct1_tum_crop is not None:
        try:
            warped_lesion = warp_mask(ct0_img_crop, ct1_tum_crop, phi)
            _atomic_image_write(warped_lesion, warped_lesion_path)
        except Exception as exc:  # noqa: BLE001
            payload["status"] = "warp_lesion_failed"
            payload["error"] = str(exc)
            payload["finished_at"] = now()
            write_json(status_path, payload)
            return payload
    else:
        payload["warped_lesion"] = None

    payload["status"] = "ok"
    payload["finished_at"] = now()
    write_json(status_path, payload)
    return payload


# =============================================================================
# Main loop
# =============================================================================
def run_producer_loop(cfg: DictConfig):
    print(f"[{NODE_ID}] Starting producer (unigradicon package, register+warp)", flush=True)

    weights_location = cfg.producer.get("unigradicon_weights", None)
    if weights_location is None:
        raise ValueError(
            "cfg.producer.unigradicon_weights must be set to the pretrained "
            "checkpoint path (e.g. 'network_weights/network_weights_9500.trch'). "
            "Loading without this arg can silently use different/uninitialized "
            "weights, which produces degenerate, flat similarity scores."
        )
    net = unigradicon.get_unigradicon()
    net.cuda().eval()

    scratch_dir    = Path(cfg.producer.scratch_dir)
    transforms_dir = scratch_dir / "transforms"
    crops_root     = scratch_dir / "_crops"
    statuses_dir   = scratch_dir / "status"
    transforms_dir.mkdir(parents=True, exist_ok=True)
    crops_root.mkdir(parents=True, exist_ok=True)
    statuses_dir.mkdir(parents=True, exist_ok=True)

    finetune_steps = cfg.producer.get("registration_finetune_steps", None)
    overwrite      = cfg.producer.get("overwrite", False)
    dry_run        = cfg.producer.get("dry_run", False)

    all_rows = load_manifest(cfg)
    print(f"[{NODE_ID}] Loaded {len(all_rows)} pairs", flush=True)
    random.shuffle(all_rows)

    max_pairs = cfg.producer.get("max_pairs", cfg.producer.get("max_batches", len(all_rows)))

    def already_done(row):
        pid = pair_id_str(row["ct0_bdmap"], row["ct1_bdmap"], row["organ"])
        # warped lesion mask is optional (only produced when a valid lesion
        # mask was available), so don't require it for "already done".
        return existing_file(transforms_dir / f"{pid}.h5") and \
            existing_file(transforms_dir / f"{pid}_ct1_warped_to_ct0.nii.gz")

    rows_to_process = all_rows if overwrite else [r for r in all_rows if not already_done(r)]
    skipped_existing = len(all_rows) - len(rows_to_process)
    print(f"[{NODE_ID}] {skipped_existing} pairs already have transforms, "
          f"{len(rows_to_process)} remaining", flush=True)

    rows_to_process = rows_to_process[:max_pairs]
    if not rows_to_process:
        print(f"[{NODE_ID}] Nothing to do. Done.", flush=True)
        return

    processed = 0
    consecutive_failures = 0

    for index, row in enumerate(rows_to_process, start=1):
        pid = pair_id_str(row["ct0_bdmap"], row["ct1_bdmap"], row["organ"])
        print(f"[{NODE_ID}] [{index}/{len(rows_to_process)}] {pid}", flush=True)
        t0 = time.time()
        result = process_pair(
            row, cfg, net, transforms_dir, crops_root, statuses_dir,
            finetune_steps=finetune_steps, overwrite=overwrite, dry_run=dry_run,
        )
        t1 = time.time()
        print(f"[{NODE_ID}]   status={result['status']} ({t1 - t0:.1f}s)", flush=True)

        if result["status"] in ("ok", "dry_run"):
            processed += 1
            consecutive_failures = 0
        elif result["status"] != "skipped_existing":
            consecutive_failures += 1
            if consecutive_failures >= 10:
                raise RuntimeError("10 consecutive failures — storage may be offline.")

    print(f"[{NODE_ID}] Done. Processed {processed} pairs "
          f"({skipped_existing} already existed).", flush=True)


@hydra.main(config_path='config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
    run_producer_loop(cfg)


if __name__ == "__main__":
    run()