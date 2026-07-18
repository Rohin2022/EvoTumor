import SimpleITK as sitk
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
import glob
import os
from scipy.ndimage import label
from radiomics import featureextractor
import logging

logging.getLogger("radiomics").setLevel(logging.ERROR)

BBOX_LIMIT_MM = 128.0

COLUMNS = [
    "bdmap_id", "organ",
    # attenuation_delta is (mean_tumor - mean_organ) / std_organ
    "attenuation_mean", "attenuation_stdev", "attenuation_delta",
    "attenuation_skew", "attenuation_10th", "attenuation_uniformity",
    "glcm_contrast", "glcm_autocorrelation", "glcm_idm"
]

_extractor = None


def worker_init():
    global _extractor
    # CRITICAL: binWidth ensures consistent GLCM matrix generation across different tumors
    settings = {'geometryTolerance': 1e-4, 'label': 1, 'binWidth': 25}
    _extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    _extractor.disableAllFeatures()
    
    _extractor.enableFeaturesByName(firstorder=[
        'Mean',
        'StandardDeviation',
        'Skewness',
        '10Percentile',  # Maps to "attenuation_10th"
        'Uniformity'
    ])

    _extractor.enableFeaturesByName(glcm=[
        'Contrast',
        'Autocorrelation',
        'Idm'  # Maps to Inverse Difference Moment (Homogeneity)
    ])


def compute_metrics(ct_sitk, organ_mask_sitk, tumor_mask_sitk, bin_tumor_mask):
    zeros = {col: 0.0 for col in COLUMNS if col not in ["bdmap_id", "organ"]}

    if not bin_tumor_mask.any():
        return zeros

    try:
        # 1. Execute PyRadiomics (Requires both CT image and Tumor Mask)
        features = _extractor.execute(ct_sitk, tumor_mask_sitk)

        # 2. Extract arrays for manual delta calculation
        ct_img = sitk.GetArrayFromImage(ct_sitk)
        bin_organ_mask = sitk.GetArrayFromImage(organ_mask_sitk) > 0
        
        # 3. Isolate healthy organ tissue (Organ is True, Tumor is False)
        healthy_organ_voxels = ct_img[(bin_organ_mask) & (~bin_tumor_mask)]
        
        if len(healthy_organ_voxels) > 0:
            mean_organ = np.mean(healthy_organ_voxels)
            std_organ = np.std(healthy_organ_voxels)
            std_organ = std_organ if std_organ != 0 else 1e-5
        else:
            # Fallback if mask is missing or tumor covers entire organ
            mean_organ = 0.0
            std_organ = 1e-5

        mean_tumor = float(features.get('original_firstorder_Mean', 0.0))
        attenuation_delta = (mean_tumor - mean_organ) / std_organ

        # 4. Map to exact columns
        return {
            "attenuation_mean":       mean_tumor,
            "attenuation_stdev":      float(features.get('original_firstorder_StandardDeviation', 0.0)),
            "attenuation_delta":      float(attenuation_delta),
            "attenuation_skew":       float(features.get('original_firstorder_Skewness', 0.0)),
            "attenuation_10th":       float(features.get('original_firstorder_10Percentile', 0.0)),
            "attenuation_uniformity": float(features.get('original_firstorder_Uniformity', 0.0)),
            "glcm_contrast":          float(features.get('original_glcm_Contrast', 0.0)),
            "glcm_autocorrelation":   float(features.get('original_glcm_Autocorrelation', 0.0)),
            "glcm_idm":               float(features.get('original_glcm_Idm', 0.0))
        }

    except Exception as e:
        print(f"Extraction failed: {e}")
        return zeros


def process_one(segmentation_path):
    parts = segmentation_path.split("/")
    bdmap = parts[-3]  # This acts as your IMAGE_ID
    
    # Assuming file format like "liver_tumor.nii.gz", this gets "liver"
    organ = "_".join(parts[-1].split("_")[:-1]) 

    # --- UPDATED PATH LOGIC ---
    # 1. Fetch the CT from the image_only/AbdomenAtlasPro directory
    ct_path = f"/projects/bodymaps/Data/image_only/AbdomenAtlasPro/AbdomenAtlasPro/{bdmap}/ct.nii.gz"
    

    def parseOrganName(organName):
            if (organName == "gallbladder"):
                return 'gall_bladder'
            return organName

    # 2. Fetch the healthy organ mask from the annotations directory
    base_dir = f"/projects/bodymaps/Data/mask_only/AbdomenAtlasPro/AbdomenAtlasPro/{bdmap}"  # goes up to the bdmap_id folder in the annotations tree
    organ_mask_path = os.path.join(base_dir, "segmentations", f"{parseOrganName(organ)}.nii.gz")
    # --------------------------

    try:
        tumor_mask_sitk = sitk.ReadImage(segmentation_path)
        bin_tumor_mask = sitk.GetArrayFromImage(tumor_mask_sitk) > 0

        # Load CT and Organ Mask
        ct_sitk = sitk.ReadImage(ct_path)
        organ_mask_sitk = sitk.ReadImage(organ_mask_path)

        metrics = compute_metrics(ct_sitk, organ_mask_sitk, tumor_mask_sitk, bin_tumor_mask)
        
        return [bdmap, organ] + [metrics[col] for col in COLUMNS[2:]]

    except Exception as e:
        print(f"ERROR processing {segmentation_path}: {bdmap}")
        return None


def generate_tumor_diffusion_data(segmentation_paths, num_workers=16, output_csv="mask_metrics.csv"):
    # ── Resume: load existing CSV and skip already-processed bdmap_ids ────────
    existing_rows = []
    already_done = set()
    if os.path.exists(output_csv):
        df_existing = pd.read_csv(output_csv)
        existing_rows = df_existing.values.tolist()
        already_done = set(df_existing["bdmap_id"].astype(str).unique())
        print(f"Resuming — found {len(df_existing)} rows across "
              f"{len(already_done)} bdmap_ids already in {output_csv}")

    def bdmap_from_path(p):
        return p.split("/")[-3]

    segmentation_paths = [
        p for p in segmentation_paths
        if bdmap_from_path(p) not in already_done
    ]
    print(f"{len(segmentation_paths)} paths remaining to process")
    # ─────────────────────────────────────────────────────────────────────────

    # Sort largest files first so heavy work runs early, not at the tail
    segmentation_paths = sorted(
        segmentation_paths, key=os.path.getsize, reverse=True)

    rows = list(existing_rows)  # seed with already-completed results

    with Pool(
        processes=num_workers,
        initializer=worker_init,
        maxtasksperchild=200,
    ) as pool:
        for result in tqdm(
            pool.imap_unordered(process_one, segmentation_paths, chunksize=1),
            total=len(segmentation_paths),
        ):
            if result is not None:
                rows.append(result)

            if len(rows) % 50 == 0 and len(rows) > 0:
                pd.DataFrame(rows, columns=COLUMNS).to_csv(
                    output_csv, index=False)

    df = pd.DataFrame(rows, columns=COLUMNS)
    df = df.sort_values(["bdmap_id", "organ"]).reset_index(drop=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} rows to {output_csv}")
    return df


if __name__ == '__main__':
    search_path = (
        "/projects/bodymaps/Data/"
        "radiologist_annotations_merlin_ucsf_atlas_multi_cancer/"
        "*/segmentations/*"
    )
    bdmaps_with_tumor_mask = glob.glob(search_path)

    df = generate_tumor_diffusion_data(
        bdmaps_with_tumor_mask, num_workers=90, output_csv="./cross_eval/abdomen_atlas_pro/tumor_metrics_v1.csv"
    )