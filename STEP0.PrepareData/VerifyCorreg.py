"""
Standalone script: pick N random pairs from the manifest, independently crop to 192^3 
based on mask centers, co-register ct1 -> ct0 with uniGradICON using organ masks, 
compute Dice scores for multiple organs, and dump the specified warped/unwarped images and ternary masks.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"

import random
import itk
import numpy as np
import pandas as pd

import unigradicon
import icon_registration.itk_wrapper as itk_wrapper

# =============================================================================
# Config — edit these
# =============================================================================
MANIFEST_CSV   = "Datapairs_with_masks.csv"
DATA_ROOT      = "/projects/bodymaps/Data/image_only/AbdomenAtlasPro/AbdomenAtlasPro/"
ORGAN_SEG_ROOT = "/projects/bodymaps/Data/mask_only/AbdomenAtlasPro/AbdomenAtlasPro/"
TUMOR_SEG_ROOT = "/projects/bodymaps/Data/radiologist_annotations_merlin_ucsf_atlas_multi_cancer/"
OUT_DIR        = "./coreg_output"
UNIGRAD_PATH   = "/projects/bodymaps/Rohin/EvoTumor/STEP0.PrepareData/network_weights/network_weights_26000.trch"
N_PAIRS        = 20
SPACING        = (1, 1, 1)
CROP_SIZE      = (175, 175, 175)
SEED           = 0

# Additional organs to evaluate Dice on (beyond the primary `organ` column in the manifest).
# These masks may not exist for every case/organ combo, so missing files are skipped gracefully.
EXTRA_ORGANS = [
    "spleen", "bladder", "gallbladder", "esophagus", "stomach",
    "duodenum", "colon", "prostate", "uterus",
]


# =============================================================================
# ITK & Math Helpers
# =============================================================================
def read_itk(path, pixel_type=itk.F):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return itk.imread(path, pixel_type=pixel_type)


def resample_to_spacing(img, spacing, is_mask=False):
    orig_spacing = np.array(img.GetSpacing())
    orig_size    = np.array(img.GetLargestPossibleRegion().GetSize())
    new_size     = [int(s) for s in np.round(orig_size * orig_spacing / np.array(spacing))]

    r = itk.ResampleImageFilter.New(img)
    r.SetOutputSpacing(list(spacing))
    r.SetSize(new_size)
    r.SetOutputDirection(img.GetDirection())
    r.SetOutputOrigin(img.GetOrigin())
    r.SetDefaultPixelValue(0)
    r.SetInterpolator(
        itk.NearestNeighborInterpolateImageFunction.New(img) if is_mask
        else itk.LinearInterpolateImageFunction.New(img)
    )
    r.Update()
    return r.GetOutput()


def warp_image(moving, fixed, transform):
    moving_cast, cast_back = unigradicon.maybe_cast(moving)
    warped = itk.resample_image_filter(
        moving_cast, transform=transform,
        interpolator=itk.LinearInterpolateImageFunction.New(moving_cast),
        use_reference_image=True, reference_image=fixed,
    )
    return cast_back(warped)


def warp_mask(mask, fixed, transform):
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


def combine_to_ternary(org_img, tum_img):
    org_arr = itk.array_from_image(org_img)
    tum_arr = itk.array_from_image(tum_img)

    ternary_arr = np.zeros_like(org_arr, dtype=np.uint8)
    ternary_arr[org_arr > 0] = 1
    ternary_arr[tum_arr > 0] = 2

    ternary_img = itk.image_from_array(ternary_arr)
    ternary_img.SetOrigin(org_img.GetOrigin())
    ternary_img.SetSpacing(org_img.GetSpacing())
    ternary_img.SetDirection(org_img.GetDirection())

    return ternary_img


def compute_dice(mask_a, mask_b):
    """Computes the Dice coefficient between two ITK mask images."""
    arr_a = itk.array_from_image(mask_a) > 0
    arr_b = itk.array_from_image(mask_b) > 0

    intersection = np.logical_and(arr_a, arr_b).sum()
    volume = arr_a.sum() + arr_b.sum()

    if volume == 0:
        return 1.0  # Both masks are empty

    return 2.0 * intersection / volume


def resolve_paths(bdmap_id, organ):
    organ_dir = "gall_bladder" if organ == "gallbladder" else organ
    return (
        os.path.join(DATA_ROOT, bdmap_id, "ct.nii.gz"),
        os.path.join(ORGAN_SEG_ROOT, bdmap_id, "segmentations", f"{organ_dir}.nii.gz"),
        os.path.join(TUMOR_SEG_ROOT, bdmap_id, "segmentations", f"{organ}_lesion.nii.gz"),
    )


def resolve_organ_mask_path(bdmap_id, organ):
    """Path to just the organ segmentation mask (no tumor), for extra Dice-only organs."""
    organ_dir = "gall_bladder" if organ == "gallbladder" else organ
    return os.path.join(ORGAN_SEG_ROOT, bdmap_id, "segmentations", f"{organ_dir}.nii.gz")


def load_triplet(bdmap_id, organ):
    img_p, org_p, tum_p = resolve_paths(bdmap_id, organ)
    img = resample_to_spacing(read_itk(img_p), SPACING, is_mask=False)
    org = resample_to_spacing(read_itk(org_p, itk.UC), SPACING, is_mask=True)
    tum = resample_to_spacing(read_itk(tum_p, itk.UC), SPACING, is_mask=True)
    return img, org, tum


def load_organ_mask(bdmap_id, organ):
    """Loads + resamples a single organ mask if it exists on disk. Returns None if missing."""
    path = resolve_organ_mask_path(bdmap_id, organ)
    if not os.path.exists(path):
        return None
    try:
        mask = read_itk(path, itk.UC)
        return resample_to_spacing(mask, SPACING, is_mask=True)
    except Exception:
        return None


# =============================================================================
# Cropping: 192^3 crop deterministically centered on mask extent
# =============================================================================
def get_center_crop_index(img, mask, crop_size):
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


def pad_to_at_least(img, min_size, is_mask=False):
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


def crop_itk(img, start, crop_size):
    start_idx = [int(s) for s in start]

    region = itk.ImageRegion[3]()
    region.SetIndex(start_idx)
    region.SetSize([int(c) for c in crop_size])

    extractor = itk.ExtractImageFilter.New(img)
    extractor.SetExtractionRegion(region)
    extractor.Update()
    out = extractor.GetOutput()
    out.DisconnectPipeline()

    # Calculate the true physical coordinate of the new crop's origin
    new_origin = img.TransformIndexToPhysicalPoint(start_idx)

    # Reset the voxel index to [0,0,0] so NIfTI writers behave predictably
    new_region = out.GetLargestPossibleRegion()
    new_region.SetIndex([0, 0, 0])
    out.SetRegions(new_region)

    # Apply the correctly calculated physical origin
    out.SetOrigin(new_origin)

    return out


def crop_triplet(img, org, tum, crop_size):
    img = pad_to_at_least(img, crop_size)
    org = pad_to_at_least(org, crop_size, is_mask=True)
    tum = pad_to_at_least(tum, crop_size, is_mask=True)

    start = get_center_crop_index(img, org if itk.array_from_image(org).any() else tum, crop_size)

    return (
        crop_itk(img, start, crop_size),
        crop_itk(org, start, crop_size),
        crop_itk(tum, start, crop_size),
        start,
    )


def pad_and_crop_extra_mask(mask, start, crop_size):
    """Pads (if needed) and crops an extra organ's mask using an already-determined start index.

    Since padding is symmetric around the original volume, and `start` was computed in the
    padded-image index space of the primary organ/tumor volumes (which share the same original
    image geometry), we replicate that padding here before applying the same crop window.
    """
    mask = pad_to_at_least(mask, crop_size, is_mask=True)
    size = np.array(mask.GetLargestPossibleRegion().GetSize())
    crop = np.array(crop_size)
    clamped_start = np.clip(start, 0, np.maximum(size - crop, 0))
    return crop_itk(mask, clamped_start, crop_size)


# =============================================================================
# Main
# =============================================================================
def main():
    random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(MANIFEST_CSV)
    rows = df.to_dict(orient="records")
    pairs = rows[:N_PAIRS]  # random.sample(rows, min(N_PAIRS, len(rows)))

    print(f"Loading uniGradICON...")
    net = unigradicon.get_unigradicon(weights_location=UNIGRAD_PATH)
    net.cuda().eval()

    all_results = []

    for i, row in enumerate(pairs):
        ct0_id, ct1_id, organ = row["ct0_bdmap"], row["ct1_bdmap"], row["organ"]
        pair_dir = os.path.join(OUT_DIR, f"{i:03d}_{ct0_id}_{ct1_id}")
        os.makedirs(pair_dir, exist_ok=True)

        dice_scores = {}

        try:
            ct0_img, ct0_org, ct0_tum = load_triplet(ct0_id, organ)
            ct1_img, ct1_org, ct1_tum = load_triplet(ct1_id, organ)

            ct0_img, ct0_org, ct0_tum, ct0_start = crop_triplet(ct0_img, ct0_org, ct0_tum, CROP_SIZE)
            ct1_img, ct1_org, ct1_tum, ct1_start = crop_triplet(ct1_img, ct1_org, ct1_tum, CROP_SIZE)

            ct0_pre = unigradicon.preprocess(ct0_img, modality="ct")
            ct1_pre = unigradicon.preprocess(ct1_img, modality="ct")

            phi, _ = itk_wrapper.register_pair_with_mask(
                net, ct1_pre, ct0_pre,
                # mask_A=ct1_org, mask_B=ct0_org,
            )

            ct1_img_w = warp_image(ct1_img, ct0_img, phi)
            ct1_org_w = warp_mask(ct1_org, ct0_img, phi)
            ct1_tum_w = warp_mask(ct1_tum, ct0_img, phi)

            # Primary organ Dice (organ masks, Fixed vs Warped Moving)
            dice_scores[organ] = compute_dice(ct0_org, ct1_org_w)

            # --- Extra organs: load, crop (using each scan's own start), warp, Dice ---
            for extra_organ in EXTRA_ORGANS:
                if extra_organ == organ:
                    continue  # already computed above

                ct0_extra = load_organ_mask(ct0_id, extra_organ)
                ct1_extra = load_organ_mask(ct1_id, extra_organ)

                if ct0_extra is None or ct1_extra is None:
                    dice_scores[extra_organ] = None  # mask unavailable for this case
                    continue

                try:
                    ct0_extra_c = pad_and_crop_extra_mask(ct0_extra, ct0_start, CROP_SIZE)
                    ct1_extra_c = pad_and_crop_extra_mask(ct1_extra, ct1_start, CROP_SIZE)

                    # Skip if the mask is empty after cropping (organ not in FOV of the crop)
                    if not itk.array_from_image(ct0_extra_c).any() or not itk.array_from_image(ct1_extra_c).any():
                        dice_scores[extra_organ] = None
                        continue

                    ct1_extra_w = warp_mask(ct1_extra_c, ct0_img, phi)
                    dice_scores[extra_organ] = compute_dice(ct0_extra_c, ct1_extra_w)
                except Exception as e:
                    print(f"    [extra organ {extra_organ} failed]: {e}")
                    dice_scores[extra_organ] = None

            ct0_ternary = combine_to_ternary(ct0_org, ct0_tum)
            ct1_ternary_w = combine_to_ternary(ct1_org_w, ct1_tum_w)

            # itk.imwrite(ct0_img, os.path.join(pair_dir, "ct0.nii.gz"))
            # itk.imwrite(ct1_img, os.path.join(pair_dir, "ct1_gt.nii.gz"))
            # itk.imwrite(ct1_img_w, os.path.join(pair_dir, "ct1_warped.nii.gz"))
            # itk.imwrite(ct0_ternary, os.path.join(pair_dir, "ct0_ternary_mask.nii.gz"))
            # itk.imwrite(ct1_ternary_w, os.path.join(pair_dir, "ct1_warped_ternary_mask.nii.gz"))

            dice_str = " | ".join(
                f"{k}: {v:.4f}" if v is not None else f"{k}: n/a"
                for k, v in dice_scores.items()
            )
            print(f"[{i+1}/{len(pairs)}] OK  {ct0_id} <- {ct1_id} ({organ}) | {dice_str}")

            result_row = {"idx": i, "ct0": ct0_id, "ct1": ct1_id, "primary_organ": organ}
            result_row.update({f"dice_{k}": v for k, v in dice_scores.items()})
            all_results.append(result_row)

        except Exception as e:
            print(f"[{i+1}/{len(pairs)}] FAILED  {ct0_id} <- {ct1_id} ({organ}): {e}")

    # Dump a summary CSV of all Dice scores across organs/pairs
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_csv = os.path.join(OUT_DIR, "dice_scores_summary.csv")
        results_df.to_csv(results_csv, index=False)
        print(f"\nSaved Dice summary to {results_csv}")

    print("Done.")


if __name__ == "__main__":
    main()