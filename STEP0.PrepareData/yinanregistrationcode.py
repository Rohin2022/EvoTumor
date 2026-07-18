#!/usr/bin/env python3
"""Run resumable uniGradICON diagnostic-to-earlier registration jobs."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path


DEFAULT_PAIRS_CSV = Path("data/longitudinal/registration_pairs_diagnostic_to_earlier.csv")
DEFAULT_CT_ROOT = Path("/projects/bodymaps/Data/image_only/AbdomenAtlasPro/AbdomenAtlasPro")
DEFAULT_REFINED_ROOT = Path("experiments/r-super_pancreatic_tumor_masks/refined")
DEFAULT_OUTPUT_ROOT = Path("experiments/registration/diagnostic_to_earlier")
DEFAULT_MASK_ONLY_ROOT = Path("/projects/bodymaps/Data/mask_only/AbdomenAtlasPro/AbdomenAtlasPro")
DEFAULT_ORIGINAL_MASK_ROOT = Path("experiments/r-super_pancreatic_tumor_masks")


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_pairs(path: Path, diagnostic_id: str | None, earlier_id: str | None, max_cases: int | None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if diagnostic_id and row.get("diagnostic_bdmap_id") != diagnostic_id:
                continue
            if earlier_id and row.get("earlier_bdmap_id") != earlier_id:
                continue
            rows.append(row)
            if max_cases is not None and len(rows) >= max_cases:
                break
    return rows


def existing_file(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def resolved_refined_roots(args: argparse.Namespace) -> list[Path]:
    return args.refined_root or [DEFAULT_REFINED_ROOT]


def ct_path(ct_root: Path, bdmap_id: str) -> Path:
    return ct_root / bdmap_id / "ct.nii.gz"


def segmentation_path(mask_only_root: Path, bdmap_id: str, segmentation_name: str) -> Path:
    return mask_only_root / bdmap_id / "segmentations" / segmentation_name


def segmentation_cache_path(cache_root: Path, bdmap_id: str, segmentation_name: str) -> Path:
    return cache_root / bdmap_id / "segmentations" / segmentation_name


def convert_segmentation_to_uint8(source: Path, destination: Path, overwrite: bool) -> Path:
    if existing_file(destination) and not overwrite:
        return destination
    try:
        import nibabel as nib
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("nibabel/numpy are required to prepare uint8 registration segmentations.") from exc

    image = nib.load(str(source))
    array = (np.asanyarray(image.dataobj) > 0).astype(np.uint8)
    destination.parent.mkdir(parents=True, exist_ok=True)
    converted = nib.Nifti1Image(array, image.affine, image.header)
    converted.set_data_dtype(np.uint8)
    nib.save(converted, str(destination))
    return destination


def save_binary_mask(source: Path, destination: Path, label: int, overwrite: bool) -> Path:
    if existing_file(destination) and not overwrite:
        return destination
    try:
        import nibabel as nib
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("nibabel/numpy are required to prepare original tumor masks.") from exc

    image = nib.load(str(source))
    array = (np.asanyarray(image.dataobj) == label).astype(np.uint8)
    destination.parent.mkdir(parents=True, exist_ok=True)
    binary = nib.Nifti1Image(array, image.affine, image.header)
    binary.set_data_dtype("uint8")
    nib.save(binary, str(destination))
    return destination


def pancreas_overlap_fraction(tumor_path: Path, pancreas_path: Path) -> float | None:
    if not existing_file(tumor_path) or not existing_file(pancreas_path):
        return None
    try:
        import nibabel as nib
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("nibabel/numpy are required to calculate tumor-pancreas overlap.") from exc

    tumor_img = nib.load(str(tumor_path))
    pancreas_img = nib.load(str(pancreas_path))
    tumor = np.asanyarray(tumor_img.dataobj) > 0
    pancreas = np.asanyarray(pancreas_img.dataobj) > 0
    tumor_voxels = int(tumor.sum())
    if tumor_voxels <= 0:
        return 0.0
    if tumor.shape != pancreas.shape:
        return None
    return int((tumor & pancreas).sum()) / tumor_voxels


def find_refined_mask(bdmap_id: str, roots: list[Path]) -> Path | None:
    relative_candidates = [
        Path(bdmap_id) / "refined_pancreatic_lesion_mask.nii.gz",
        Path(f"{bdmap_id}.nii.gz"),
    ]
    glob_patterns = [
        f"*/{bdmap_id}/refined_pancreatic_lesion_mask.nii.gz",
        f"*/*/{bdmap_id}/refined_pancreatic_lesion_mask.nii.gz",
    ]
    for root in roots:
        for relative in relative_candidates:
            path = root / relative
            if existing_file(path):
                return path
        for pattern in glob_patterns:
            matches = sorted(root.glob(pattern))
            for path in matches:
                if existing_file(path):
                    return path
    return None


def find_original_multilabel_mask(bdmap_id: str, roots: list[Path]) -> Path | None:
    candidates = [
        Path(f"{bdmap_id}.nii.gz"),
        Path(f"{bdmap_id}.nii"),
        Path("pdac_lessthan2cm") / f"{bdmap_id}.nii.gz",
        Path("pdac_greaterthan2cm") / f"{bdmap_id}.nii.gz",
    ]
    for root in roots:
        for relative in candidates:
            path = root / relative
            if existing_file(path):
                return path
        for pattern in [f"*/{bdmap_id}.nii.gz", f"*/{bdmap_id}.nii"]:
            for path in sorted(root.glob(pattern)):
                if existing_file(path):
                    return path
    return None


def collect_refined_mask_ids(roots: list[Path]) -> set[str]:
    ids: set[str] = set()
    for root in roots:
        for path in root.glob("*/refined_pancreatic_lesion_mask.nii.gz"):
            ids.add(path.parent.name)
        for path in root.glob("*/*/refined_pancreatic_lesion_mask.nii.gz"):
            ids.add(path.parent.name)
        for path in root.glob("*/*/*/refined_pancreatic_lesion_mask.nii.gz"):
            ids.add(path.parent.name)
        for path in root.glob("BDMAP_*.nii.gz"):
            ids.add(path.name.removesuffix(".nii.gz"))
        for path in root.glob("BDMAP_*.nii"):
            ids.add(path.name.removesuffix(".nii"))
    return ids


def pair_id(row: dict[str, str]) -> str:
    return f'{row["diagnostic_bdmap_id"]}_to_{row["earlier_bdmap_id"]}'


def build_pairs_from_metadata(args: argparse.Namespace) -> Path:
    from build_longitudinal_scan_manifest import (
        dedupe_scans,
        read_ids,
        stream_scans,
        write_patient_summary,
        write_registration_pairs,
        write_scan_manifest,
    )

    if args.pairs_output_dir is None:
        args.pairs_output_dir = args.output_root / "_pair_inputs"

    target_ids: set[str] = set()
    if args.target_ids:
        target_ids.update(read_ids(args.target_ids))
    if args.target_ids_from_refined_root:
        target_ids.update(collect_refined_mask_ids(resolved_refined_roots(args)))
    if not target_ids:
        raise SystemExit("Use --target-ids or --target-ids-from-refined-root with --build-pairs-from-metadata.")

    records, total_rows, rows_with_patient, rows_with_date = stream_scans(
        args.metadata,
        target_ids,
        args.ct_root,
        only_target_patients=not args.all_scans,
    )
    records = dedupe_scans(records)
    grouped: dict[str, list] = defaultdict(list)
    for record in records:
        grouped[record.patient_key].append(record)

    scan_manifest = args.pairs_output_dir / "scan_manifest.csv"
    patient_summary = args.pairs_output_dir / "patients_multi_timepoints.csv"
    registration_pairs = args.pairs_output_dir / "registration_pairs_diagnostic_to_earlier.csv"

    write_scan_manifest(scan_manifest, records)
    multi_time_count = write_patient_summary(patient_summary, grouped)
    pair_count = write_registration_pairs(
        registration_pairs,
        grouped,
        resolved_refined_roots(args),
        args.min_days_before,
        args.max_days_before,
    )

    print(f"Metadata rows scanned: {total_rows}")
    print(f"Rows with patient key: {rows_with_patient}")
    print(f"Rows with parseable scan date: {rows_with_date}")
    print(f"Target diagnostic IDs loaded: {len(target_ids)}")
    print(f"Manifest scans written: {len(records)}")
    print(f"Patients in manifest: {len(grouped)}")
    print(f"Patients with multiple scan dates: {multi_time_count}")
    print(f"Registration candidate pairs written: {pair_count}")
    print(f"Wrote: {scan_manifest}")
    print(f"Wrote: {patient_summary}")
    print(f"Wrote: {registration_pairs}")
    return registration_pairs


def warped_mask_output_path(out_dir: Path, pid: str, tumor_mask_source: str) -> Path:
    if tumor_mask_source == "refined":
        return out_dir / "warped_masks" / f"{pid}_refined_tumor_mask_warped_to_earlier.nii.gz"
    return out_dir / f"warped_masks_{tumor_mask_source}" / f"{pid}_{tumor_mask_source}_tumor_mask_warped_to_earlier.nii.gz"


def registration_status_path(out_dir: Path, tumor_mask_source: str) -> Path:
    if tumor_mask_source == "refined":
        return out_dir / "registration_status.json"
    return out_dir / f"registration_status_{tumor_mask_source}.json"


def selected_tumor_mask(
    diagnostic_id: str,
    diagnostic_pancreas: Path,
    args: argparse.Namespace,
) -> dict:
    refined_mask = find_refined_mask(diagnostic_id, resolved_refined_roots(args))
    original_multilabel = find_original_multilabel_mask(diagnostic_id, args.original_mask_root or [DEFAULT_ORIGINAL_MASK_ROOT])
    original_binary = None
    if original_multilabel is not None:
        original_binary = args.output_root / "_tumor_masks" / "original_label23" / f"{diagnostic_id}_original_label{args.original_tumor_label}.nii.gz"
        original_binary = save_binary_mask(original_multilabel, original_binary, args.original_tumor_label, args.overwrite)

    refined_fraction = None
    original_fraction = None
    if refined_mask is not None:
        refined_fraction = pancreas_overlap_fraction(refined_mask, diagnostic_pancreas)
    if original_binary is not None:
        original_fraction = pancreas_overlap_fraction(original_binary, diagnostic_pancreas)

    requested = args.tumor_mask_source
    selected_source = requested
    selection_reason = f"requested_{requested}"
    selected_path = refined_mask

    if requested == "original":
        selected_path = original_binary
    elif requested == "hybrid":
        if refined_mask is None and original_binary is not None:
            selected_source = "original"
            selected_path = original_binary
            selection_reason = "refined_missing"
        elif refined_fraction is None and original_binary is not None:
            selected_source = "original"
            selected_path = original_binary
            selection_reason = "refined_overlap_unavailable"
        elif refined_fraction is not None and refined_fraction < args.hybrid_refined_overlap_threshold and original_binary is not None:
            selected_source = "original"
            selected_path = original_binary
            selection_reason = f"refined_overlap_below_{args.hybrid_refined_overlap_threshold:g}"
        else:
            selected_source = "refined"
            selected_path = refined_mask
            selection_reason = "refined_overlap_accepted"

    return {
        "requested_tumor_mask_source": requested,
        "selected_tumor_mask_source": selected_source,
        "tumor_mask_selection_reason": selection_reason,
        "diagnostic_mask": selected_path,
        "refined_tumor_mask": refined_mask,
        "original_multilabel_mask": original_multilabel,
        "original_tumor_mask": original_binary,
        "refined_original_pancreas_fraction": refined_fraction,
        "original_original_pancreas_fraction": original_fraction,
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_log(path: Path, title: str, command: list[str], dry_run: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{now()}] {title}\n")
        handle.write("dry_run: " + str(dry_run) + "\n")
        handle.write("command: " + " ".join(command) + "\n")


def run_command(command: list[str], log_path: Path, dry_run: bool) -> int:
    append_log(log_path, "run", command, dry_run)
    if dry_run:
        return 0
    with log_path.open("a", encoding="utf-8") as log_handle:
        process = subprocess.run(command, stdout=log_handle, stderr=subprocess.STDOUT, text=True)
    return process.returncode


def process_pair(row: dict[str, str], args: argparse.Namespace) -> dict:
    pid = pair_id(row)
    out_dir = args.output_root / pid
    transform_path = out_dir / "transforms" / f"{pid}.hdf5"
    warped_image_path = out_dir / "warped_images" / f"{pid}_diagnostic_ct_warped_to_earlier.nii.gz"
    warped_mask_path = warped_mask_output_path(out_dir, pid, args.tumor_mask_source)
    status_path = registration_status_path(out_dir, args.tumor_mask_source)
    log_path = out_dir / "logs" / "registration.log"

    diagnostic_id = row["diagnostic_bdmap_id"]
    earlier_id = row["earlier_bdmap_id"]
    diagnostic_ct = ct_path(args.ct_root, diagnostic_id)
    earlier_ct = ct_path(args.ct_root, earlier_id)
    use_registration_segmentations = args.registration_segmentation_mode != "none"
    moving_segmentation_source = (
        segmentation_path(args.mask_only_root, diagnostic_id, args.registration_segmentation_name)
        if use_registration_segmentations
        else None
    )
    fixed_segmentation_source = (
        segmentation_path(args.mask_only_root, earlier_id, args.registration_segmentation_name)
        if use_registration_segmentations
        else None
    )
    moving_segmentation = moving_segmentation_source
    fixed_segmentation = fixed_segmentation_source
    diagnostic_pancreas = segmentation_path(args.mask_only_root, diagnostic_id, args.registration_segmentation_name)

    try:
        tumor_selection = selected_tumor_mask(diagnostic_id, diagnostic_pancreas, args)
    except RuntimeError as exc:
        payload = {
            "pair_id": pid,
            "diagnostic_bdmap_id": diagnostic_id,
            "earlier_bdmap_id": earlier_id,
            "days_before_diagnosis": row.get("days_before_diagnosis", ""),
            "status": "tumor_mask_selection_failed",
            "error": str(exc),
            "finished_at": now(),
        }
        write_json(status_path, payload)
        return payload
    diagnostic_mask = tumor_selection["diagnostic_mask"]

    payload = {
        "pair_id": pid,
        "diagnostic_bdmap_id": diagnostic_id,
        "earlier_bdmap_id": earlier_id,
        "days_before_diagnosis": row.get("days_before_diagnosis", ""),
        "diagnostic_ct": str(diagnostic_ct),
        "earlier_ct": str(earlier_ct),
        "diagnostic_mask": str(diagnostic_mask or ""),
        "requested_tumor_mask_source": tumor_selection["requested_tumor_mask_source"],
        "selected_tumor_mask_source": tumor_selection["selected_tumor_mask_source"],
        "tumor_mask_selection_reason": tumor_selection["tumor_mask_selection_reason"],
        "refined_tumor_mask": str(tumor_selection["refined_tumor_mask"] or ""),
        "original_multilabel_mask": str(tumor_selection["original_multilabel_mask"] or ""),
        "original_tumor_mask": str(tumor_selection["original_tumor_mask"] or ""),
        "refined_original_pancreas_fraction": tumor_selection["refined_original_pancreas_fraction"],
        "original_original_pancreas_fraction": tumor_selection["original_original_pancreas_fraction"],
        "registration_segmentation_mode": args.registration_segmentation_mode,
        "moving_segmentation_source": str(moving_segmentation_source or ""),
        "fixed_segmentation_source": str(fixed_segmentation_source or ""),
        "moving_segmentation": str(moving_segmentation or ""),
        "fixed_segmentation": str(fixed_segmentation or ""),
        "transform": str(transform_path),
        "warped_image": str(warped_image_path),
        "warped_mask": str(warped_mask_path),
        "started_at": now(),
        "status": "started",
    }

    missing = []
    for label, path in [("diagnostic_ct", diagnostic_ct), ("earlier_ct", earlier_ct)]:
        if not existing_file(path):
            missing.append(label)
    if diagnostic_mask is None:
        missing.append("diagnostic_mask")
    if use_registration_segmentations:
        for label, path in [("moving_segmentation", moving_segmentation_source), ("fixed_segmentation", fixed_segmentation_source)]:
            if path is None or not existing_file(path):
                missing.append(label)
    if missing:
        payload["status"] = "missing_inputs"
        payload["missing_inputs"] = missing
        payload["finished_at"] = now()
        write_json(status_path, payload)
        return payload

    if use_registration_segmentations and args.prepare_registration_segmentations:
        cache_root = args.registration_segmentation_cache_root or (args.output_root / "_registration_segmentations_uint8")
        try:
            moving_segmentation = convert_segmentation_to_uint8(
                moving_segmentation_source,
                segmentation_cache_path(cache_root, diagnostic_id, args.registration_segmentation_name),
                args.overwrite,
            )
            fixed_segmentation = convert_segmentation_to_uint8(
                fixed_segmentation_source,
                segmentation_cache_path(cache_root, earlier_id, args.registration_segmentation_name),
                args.overwrite,
            )
        except RuntimeError as exc:
            payload["status"] = "segmentation_conversion_failed"
            payload["error"] = str(exc)
            payload["finished_at"] = now()
            write_json(status_path, payload)
            return payload
        payload["moving_segmentation"] = str(moving_segmentation)
        payload["fixed_segmentation"] = str(fixed_segmentation)

    if existing_file(warped_mask_path) and existing_file(transform_path) and not args.overwrite:
        payload["status"] = "skipped_existing"
        payload["finished_at"] = now()
        write_json(status_path, payload)
        return payload

    for directory in [transform_path.parent, warped_image_path.parent, warped_mask_path.parent, log_path.parent]:
        directory.mkdir(parents=True, exist_ok=True)

    if not existing_file(transform_path) or args.overwrite:
        register_command = [
            "unigradicon-register",
            "--fixed",
            str(earlier_ct),
            "--fixed_modality",
            "ct",
            "--moving",
            str(diagnostic_ct),
            "--moving_modality",
            "ct",
            "--transform_out",
            str(transform_path),
            "--warped_moving_out",
            str(warped_image_path),
            "--model",
            args.model,
            "--io_iterations",
            str(args.io_iterations),
        ]
        if use_registration_segmentations:
            register_command.extend(
                [
                    "--fixed_segmentation",
                    str(fixed_segmentation),
                    "--moving_segmentation",
                    str(moving_segmentation),
                ]
            )
        if args.registration_segmentation_mode == "loss_mask":
            register_command.append("--loss_function_masking")
        code = run_command(register_command, log_path, args.dry_run)
        if code != 0:
            payload["status"] = "register_failed"
            payload["returncode"] = code
            payload["finished_at"] = now()
            write_json(status_path, payload)
            return payload
    else:
        append_log(log_path, "skip_register_existing_transform", ["existing", str(transform_path)], args.dry_run)

    if not existing_file(warped_mask_path) or args.overwrite:
        warp_command = [
            "unigradicon-warp",
            "--fixed",
            str(earlier_ct),
            "--moving",
            str(diagnostic_mask),
            "--transform",
            str(transform_path),
            "--warped_moving_out",
            str(warped_mask_path),
            "--nearest_neighbor",
        ]
        code = run_command(warp_command, log_path, args.dry_run)
        if code != 0:
            payload["status"] = "warp_failed"
            payload["returncode"] = code
            payload["finished_at"] = now()
            write_json(status_path, payload)
            return payload
    else:
        append_log(log_path, "skip_warp_existing_mask", ["existing", str(warped_mask_path)], args.dry_run)

    payload["status"] = "ok" if not args.dry_run else "dry_run"
    payload["finished_at"] = now()
    write_json(status_path, payload)
    return payload


def write_summary(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "pair_id",
        "diagnostic_bdmap_id",
        "earlier_bdmap_id",
        "days_before_diagnosis",
        "status",
        "diagnostic_ct",
        "earlier_ct",
        "diagnostic_mask",
        "requested_tumor_mask_source",
        "selected_tumor_mask_source",
        "tumor_mask_selection_reason",
        "refined_tumor_mask",
        "original_multilabel_mask",
        "original_tumor_mask",
        "refined_original_pancreas_fraction",
        "original_original_pancreas_fraction",
        "registration_segmentation_mode",
        "moving_segmentation_source",
        "fixed_segmentation_source",
        "moving_segmentation",
        "fixed_segmentation",
        "transform",
        "warped_image",
        "warped_mask",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run resumable uniGradICON registration and tumor-mask warp-back.")
    parser.add_argument("--pairs-csv", type=Path, default=DEFAULT_PAIRS_CSV)
    parser.add_argument("--build-pairs-from-metadata", action="store_true")
    parser.add_argument("--metadata", type=Path, default=Path("/projects/bodymaps/YinanShi/metadata/metadata_ucsf_batch_1_to_6_and_merlin.csv"))
    parser.add_argument("--target-ids", type=Path)
    parser.add_argument("--target-ids-from-refined-root", action="store_true")
    parser.add_argument("--pairs-output-dir", type=Path)
    parser.add_argument("--all-scans", action="store_true")
    parser.add_argument("--min-days-before", type=int, default=1)
    parser.add_argument("--max-days-before", type=int)
    parser.add_argument("--diagnostic-id")
    parser.add_argument("--earlier-id")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--ct-root", type=Path, default=DEFAULT_CT_ROOT)
    parser.add_argument("--refined-root", type=Path, action="append")
    parser.add_argument("--original-mask-root", type=Path, action="append")
    parser.add_argument("--original-tumor-label", type=int, default=23)
    parser.add_argument(
        "--tumor-mask-source",
        choices=["refined", "original", "hybrid"],
        default="refined",
        help=(
            "Which diagnostic tumor mask to warp. 'refined' keeps the original behavior. "
            "'original' extracts label 23 from the original R-Super multi-label mask. "
            "'hybrid' uses refined unless refined tumor-pancreas overlap is below threshold."
        ),
    )
    parser.add_argument("--hybrid-refined-overlap-threshold", type=float, default=0.05)
    parser.add_argument("--mask-only-root", type=Path, default=DEFAULT_MASK_ONLY_ROOT)
    parser.add_argument("--registration-segmentation-name", default="pancreas.nii.gz")
    parser.add_argument("--registration-segmentation-cache-root", type=Path)
    parser.add_argument(
        "--prepare-registration-segmentations",
        dest="prepare_registration_segmentations",
        action="store_true",
        help="Convert registration segmentations to uint8 cache files before passing them to uniGradICON. Default.",
    )
    parser.add_argument(
        "--no-prepare-registration-segmentations",
        dest="prepare_registration_segmentations",
        action="store_false",
        help="Pass registration segmentations directly to uniGradICON.",
    )
    parser.add_argument(
        "--registration-segmentation-mode",
        choices=["none", "image_mask", "loss_mask"],
        default="none",
        help=(
            "How to use fixed/moving segmentations during registration. "
            "'none' is image-only CT-to-CT. 'image_mask' passes segmentations so uniGradICON masks images before "
            "registration. 'loss_mask' passes segmentations with --loss_function_masking during IO."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_OUTPUT_ROOT / "registration_status_summary.csv")
    parser.add_argument("--model", default="unigradicon")
    parser.add_argument("--io-iterations", default="50")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.set_defaults(prepare_registration_segmentations=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.build_pairs_from_metadata:
        args.pairs_csv = build_pairs_from_metadata(args)
    rows = read_pairs(args.pairs_csv, args.diagnostic_id, args.earlier_id, args.max_cases)
    if not rows:
        raise SystemExit("No registration pairs selected.")

    results = []
    for idx, row in enumerate(rows, start=1):
        print(f"[{idx}/{len(rows)}] {pair_id(row)}")
        result = process_pair(row, args)
        results.append(result)
        print(f"  status: {result['status']}")

    write_summary(args.summary_csv, results)
    print(f"Wrote summary: {args.summary_csv}")


if __name__ == "__main__":
    main()
