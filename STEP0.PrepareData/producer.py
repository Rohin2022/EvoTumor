import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"

import json, random, socket, subprocess, time, uuid
from datetime import datetime
from pathlib import Path

import itk, numpy as np, pandas as pd, torch, hydra
from omegaconf import DictConfig

from monai.data import MetaTensor
from monai.transforms import Orientation as MonaiOrientation


NODE_ID = f"{socket.gethostname()}_{os.getpid()}"
CROP_SIZE = (175, 175, 175)


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def existing_file(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


# =============================================================================
# ITK helpers (preprocessing only — same as before, unrelated to registration)
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
    df = df[df["organ"].isin(organs_list)]
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")
    return df.to_dict(orient="records")


# =============================================================================
# Building the (cropped, preprocessed) crop volumes on disk for a pair
# =============================================================================
def prepare_pair_crops(cfg, row, crop_dir: Path):
    """Load ct0/ct1 + organ/lesion masks, preprocess (RAS + resample), crop
    each scan independently to CROP_SIZE centered on its own organ mask, and
    write the crops to crop_dir. Returns a dict of paths, mirroring the
    prepare-step outputs the other scripts consume."""
    spacing = (cfg.dataset.space_x, cfg.dataset.space_y, cfg.dataset.space_z)
    ct0_id, ct1_id, organ = row["ct0_bdmap"], row["ct1_bdmap"], row["organ"]

    ct0_img_p, ct0_org_p, ct0_tum_p = resolve_source_paths(cfg, ct0_id, organ)
    ct1_img_p, ct1_org_p, ct1_tum_p = resolve_source_paths(cfg, ct1_id, organ)

    ct0_img = preprocess_itk(_read_itk(ct0_img_p), spacing, is_mask=False)
    ct0_org = preprocess_itk(_read_itk(ct0_org_p, itk.UC), spacing, is_mask=True)
    ct0_tum = preprocess_itk(_read_itk(ct0_tum_p, itk.UC), spacing, is_mask=True)

    ct1_img = preprocess_itk(_read_itk(ct1_img_p), spacing, is_mask=False)
    ct1_org = preprocess_itk(_read_itk(ct1_org_p, itk.UC), spacing, is_mask=True)
    ct1_tum = preprocess_itk(_read_itk(ct1_tum_p, itk.UC), spacing, is_mask=True)

    ct0_img_crop, ct0_org_crop = crop_image_and_organ_mask(ct0_img, ct0_org)
    ct1_img_crop, ct1_org_crop = crop_image_and_organ_mask(ct1_img, ct1_org)
    # crop the lesion masks to the same windows as their own scan
    ct0_tum_crop = _crop_itk(_pad_to_at_least(ct0_tum, CROP_SIZE),
                              _get_center_crop_index(_pad_to_at_least(ct0_img, CROP_SIZE),
                                                      _pad_to_at_least(ct0_org, CROP_SIZE), CROP_SIZE),
                              CROP_SIZE)
    ct1_tum_crop = _crop_itk(_pad_to_at_least(ct1_tum, CROP_SIZE),
                              _get_center_crop_index(_pad_to_at_least(ct1_img, CROP_SIZE),
                                                      _pad_to_at_least(ct1_org, CROP_SIZE), CROP_SIZE),
                              CROP_SIZE)

    crop_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "ct0_ct_crop": crop_dir / f"{ct0_id}_ct_crop.nii.gz",
        "ct0_organ_crop": crop_dir / f"{ct0_id}_{organ}_crop.nii.gz",
        "ct0_lesion_crop": crop_dir / f"{ct0_id}_{organ}_lesion_crop.nii.gz",
        "ct1_ct_crop": crop_dir / f"{ct1_id}_ct_crop.nii.gz",
        "ct1_organ_crop": crop_dir / f"{ct1_id}_{organ}_crop.nii.gz",
        "ct1_lesion_crop": crop_dir / f"{ct1_id}_{organ}_lesion_crop.nii.gz",
    }
    itk.imwrite(ct0_img_crop, str(paths["ct0_ct_crop"]))
    itk.imwrite(ct0_org_crop, str(paths["ct0_organ_crop"]))
    itk.imwrite(ct0_tum_crop, str(paths["ct0_lesion_crop"]))
    itk.imwrite(ct1_img_crop, str(paths["ct1_ct_crop"]))
    itk.imwrite(ct1_org_crop, str(paths["ct1_organ_crop"]))
    itk.imwrite(ct1_tum_crop, str(paths["ct1_lesion_crop"]))
    return paths


# =============================================================================
# Registration + warp — identical CLI pattern to the other scripts
# =============================================================================
def append_log(path: Path, title: str, command: list, dry_run: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{now()}] {title}\n")
        handle.write("dry_run: " + str(dry_run) + "\n")
        handle.write("command: " + " ".join(command) + "\n")


def run_command(command: list, log_path: Path, dry_run: bool) -> int:
    append_log(log_path, "run", command, dry_run)
    if dry_run:
        return 0
    with log_path.open("a", encoding="utf-8") as log_handle:
        process = subprocess.run(command, stdout=log_handle, stderr=subprocess.STDOUT, text=True)
    return process.returncode


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def process_pair(row, cfg, transforms_dir: Path, crops_root: Path, statuses_dir: Path,
                  model: str, io_iterations: int, overwrite: bool, dry_run: bool) -> dict:
    ct0_id, ct1_id, organ = row["ct0_bdmap"], row["ct1_bdmap"], row["organ"]
    pid = pair_id_str(ct0_id, ct1_id, organ)

    crop_dir = crops_root / pid
    transform_path = transforms_dir / f"{pid}.h5"
    warped_ct_path = transforms_dir / f"{pid}_ct1_warped_to_ct0.nii.gz"
    warped_lesion_path = transforms_dir / f"{pid}_ct1_lesion_warped_to_ct0.nii.gz"
    status_path = statuses_dir / f"{pid}.json"
    log_path = transforms_dir / "logs" / f"{pid}.log"

    payload = {
        "pair_id": pid,
        "ct0_bdmap": ct0_id,
        "ct1_bdmap": ct1_id,
        "organ": organ,
        "transform": str(transform_path),
        "warped_ct": str(warped_ct_path),
        "warped_lesion": str(warped_lesion_path),
        "started_at": now(),
        "status": "started",
    }

    if existing_file(transform_path) and existing_file(warped_lesion_path) and not overwrite:
        payload["status"] = "skipped_existing"
        payload["finished_at"] = now()
        write_json(status_path, payload)
        return payload

    try:
        crop_paths = prepare_pair_crops(cfg, row, crop_dir)
    except Exception as exc:  # noqa: BLE001 - keep the loop going, record failure
        payload["status"] = "prepare_failed"
        payload["error"] = str(exc)
        payload["finished_at"] = now()
        write_json(status_path, payload)
        return payload

    register_command = [
        "unigradicon-register",
        "--fixed", str(crop_paths["ct0_ct_crop"]),
        "--fixed_modality", "ct",
        "--moving", str(crop_paths["ct1_ct_crop"]),
        "--moving_modality", "ct",
        "--transform_out", str(transform_path),
        "--warped_moving_out", str(warped_ct_path),
        "--model", model,
        "--io_iterations", str(io_iterations),
    ]
    return_code = run_command(register_command, log_path, dry_run)
    if return_code != 0:
        payload["status"] = "register_failed"
        payload["returncode"] = return_code
        payload["finished_at"] = now()
        write_json(status_path, payload)
        return payload

    warp_command = [
        "unigradicon-warp",
        "--fixed", str(crop_paths["ct0_ct_crop"]),
        "--moving", str(crop_paths["ct1_lesion_crop"]),
        "--transform", str(transform_path),
        "--warped_moving_out", str(warped_lesion_path),
        "--nearest_neighbor",
    ]
    return_code = run_command(warp_command, log_path, dry_run)
    if return_code != 0:
        payload["status"] = "warp_failed"
        payload["returncode"] = return_code
        payload["finished_at"] = now()
        write_json(status_path, payload)
        return payload

    payload["status"] = "ok" if not dry_run else "dry_run"
    payload["finished_at"] = now()
    write_json(status_path, payload)
    return payload


# =============================================================================
# Main loop
# =============================================================================
def run_producer_loop(cfg: DictConfig):
    print(f"[{NODE_ID}] Starting producer (unigradicon CLI, register+warp)", flush=True)

    scratch_dir    = Path(cfg.producer.scratch_dir)
    transforms_dir = scratch_dir / "transforms"
    crops_root     = scratch_dir / "_crops"
    statuses_dir   = scratch_dir / "status"
    transforms_dir.mkdir(parents=True, exist_ok=True)
    crops_root.mkdir(parents=True, exist_ok=True)
    statuses_dir.mkdir(parents=True, exist_ok=True)

    model         = cfg.producer.get("model", "unigradicon")
    io_iterations = cfg.producer.get("io_iterations", 50)
    overwrite     = cfg.producer.get("overwrite", False)
    dry_run       = cfg.producer.get("dry_run", False)

    all_rows = load_manifest(cfg)
    print(f"[{NODE_ID}] Loaded {len(all_rows)} pairs", flush=True)
    random.shuffle(all_rows)

    max_pairs = cfg.producer.get("max_pairs", cfg.producer.get("max_batches", len(all_rows)))

    def already_done(row):
        pid = pair_id_str(row["ct0_bdmap"], row["ct1_bdmap"], row["organ"])
        return existing_file(transforms_dir / f"{pid}.h5") and \
            existing_file(transforms_dir / f"{pid}_ct1_lesion_warped_to_ct0.nii.gz")

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
            row, cfg, transforms_dir, crops_root, statuses_dir,
            model=model, io_iterations=io_iterations, overwrite=overwrite, dry_run=dry_run,
        )
        t1 = time.time()
        print(f"[{NODE_ID}]   status={result['status']} ({t1 - t0:.1f}s)", flush=True)

        if result["status"].startswith("ok") or result["status"] == "dry_run":
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