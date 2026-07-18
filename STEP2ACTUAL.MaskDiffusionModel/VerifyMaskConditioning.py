import nibabel as nib
import numpy as np
import torch
import pprint
import torch.nn.functional as F
from pathlib import Path
from monai.transforms import FillHoles, KeepLargestConnectedComponent, Compose


def load_nifti_and_compute_metrics(cleaned_nifti_path):
    """Wrapper function to handle NIfTI file I/O safely."""
    img = nib.load(cleaned_nifti_path)
    raw_data = img.get_fdata() # Typically (X, Y, Z)
    
    # Extract spacing (zooms) and the affine matrix directly from nibabel
    spacing = img.header.get_zooms()[:3] 
    affine = img.affine
    
    return compute_mask_metrics(raw_data, spacing, affine)


COLUMNS = [
    "bdmap_id", "organ",
    "diameter_x_mm", "diameter_y_mm", "diameter_z_mm",
    "mean_x_mm", "mean_y_mm", "mean_z_mm",
    "std_x_mm", "std_y_mm", "std_z_mm",
    "volume_ml",
]


def compute_diameters_and_coords(mask, spacing, affine):
    # Handle torch tensors if they are passed instead of numpy arrays
    if hasattr(mask, "numpy"):
        mask = mask.cpu().numpy()
        
    if mask.ndim == 4:
        mask = mask.squeeze(0)
    
    zeros = {col: 0.0 for col in COLUMNS if col not in ["bdmap_id", "organ"]}
    
    bin_mask = mask > 0
    if not bin_mask.any():
        return zeros

    def span(axis):
        projected = np.any(bin_mask, axis=axis)
        nonzero_idx = np.where(projected)[0]
        return nonzero_idx[-1] - nonzero_idx[0] + 1

    dx = span(axis=(1, 2))
    dy = span(axis=(0, 2))
    dz = span(axis=(0, 1))
    
    # Convert spacing to a numpy array, handling potential tensor inputs
    if hasattr(spacing, "numpy"):
        spacing = spacing.cpu().numpy()
    spacing = np.abs(spacing)
    
    diameters_mm = np.array([dx, dy, dz]) * spacing

    coords = np.argwhere(bin_mask)
    
    # Ensure affine is a numpy array
    if hasattr(affine, "numpy"):
        affine = affine.cpu().numpy()
    else:
        affine = np.asarray(affine)
        
    coords_homo = np.pad(coords, ((0, 0), (0, 1)), constant_values=1)
    
    coords_world = coords_homo @ affine.T
    coords_world = coords_world[:, :3] 

    means_mm = coords_world.mean(axis=0)
    stds_mm  = coords_world.std(axis=0)

    # 4. Volume (1000 mm^3 = 1 ml)
    voxel_vol_ml = (spacing[0] * spacing[1] * spacing[2]) / 1000.0
    volume_ml = float(bin_mask.sum() * voxel_vol_ml)

    return {
        "diameter_x_mm": float(diameters_mm[0]),
        "diameter_y_mm": float(diameters_mm[1]),
        "diameter_z_mm": float(diameters_mm[2]),
        "mean_x_mm":     float(means_mm[0]),
        "mean_y_mm":     float(means_mm[1]),
        "mean_z_mm":     float(means_mm[2]),
        "std_x_mm":      float(stds_mm[0]),
        "std_y_mm":      float(stds_mm[1]),
        "std_z_mm":      float(stds_mm[2]),
        "volume_ml":     volume_ml
    }


def compute_mask_metrics(mask, spacing, affine):
    return compute_diameters_and_coords(mask, spacing, affine)


if __name__ == "__main__":
    cleaned_nifti_path = "checkpoints/liver_tumor_train/mask_diffusion_train_pro/debug_masks/step_0_sample_0_RECON.nii.gz"
    
    if Path(cleaned_nifti_path).exists():
        metrics = load_nifti_and_compute_metrics(
            cleaned_nifti_path=cleaned_nifti_path,
        )
        pprint.pprint(metrics)
    else:
        print(f"File not found: {cleaned_nifti_path}")