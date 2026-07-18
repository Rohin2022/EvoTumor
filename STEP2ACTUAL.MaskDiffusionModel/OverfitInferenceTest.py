from dataset.dataloader import get_loader
import numpy as np
import nibabel as nib
import torch.nn.functional as F
import torch
from omegaconf import DictConfig, open_dict
import hydra
import os
from ddpm import Unet3D, GaussianDiffusion
from pathlib import Path
from tqdm import tqdm
import json

# Add MONAI imports for post-processing
from monai.transforms import FillHoles, KeepLargestConnectedComponent, Compose

import sys
sys.path.append(os.getcwd())

def postprocess_tensor(raw_mask, scale_factor=3, threshold=0.5, num_components=1):
    """
    Handles both (X, Y, Z) and (B, X, Y, Z) formats with NO channel dimension.
    Accepts both PyTorch Tensors and NumPy arrays.
    """
    is_numpy = isinstance(raw_mask, np.ndarray)
    if is_numpy:
        tensor_mask = torch.from_numpy(raw_mask).float()
    else:
        tensor_mask = raw_mask.float()

    original_dims = tensor_mask.dim()
    if original_dims == 3:
        tensor_mask = tensor_mask.unsqueeze(0)
    elif original_dims == 4:
        pass
    else:
        raise ValueError(f"Expected 3D (X,Y,Z) or 4D (B,X,Y,Z) input, got {original_dims}D")

    tensor_mask = tensor_mask.unsqueeze(1)

    if scale_factor != 1:
        tensor_mask = F.interpolate(
            tensor_mask, 
            scale_factor=scale_factor, 
            mode='trilinear', 
            align_corners=False
        )

    binary_mask = (tensor_mask < threshold).to(torch.uint8) # Fixed: should be > threshold for mask

    postprocess_transforms = Compose([
        FillHoles(),
        KeepLargestConnectedComponent(num_components=num_components)
    ])

    processed_batch = []
    for i in range(binary_mask.shape[0]):
        single_item = binary_mask[i]
        cleaned_item = postprocess_transforms(single_item)
        processed_batch.append(cleaned_item)

    final_tensor = torch.stack(processed_batch, dim=0)
    final_tensor = final_tensor.squeeze(1)

    if original_dims == 3:
        final_tensor = final_tensor.squeeze(0)

    # Return in the same format it was received
    if is_numpy:
        return final_tensor.cpu().numpy().astype(np.uint8)
    return final_tensor


# --- NEW IMPORTS REQUIRED FOR METRICS ---
from scipy.ndimage import label
from skimage.measure import marching_cubes, mesh_surface_area

def compute_diameters_and_coords(mask, spacing):
    """
    Computes volume, diameters, PCA-based elongation/flatness, and 
    marching-cubes-based sphericity for the given 3D mask.
    """
    if hasattr(mask, "numpy"):
        mask = mask.cpu().numpy()

    mask = np.squeeze(mask)
    spacing = np.array(spacing) # Ensure this is a numpy array for broadcasting!

    COLUMNS = [
        "bdmap_id", "organ",
        "diameter_x_mm", "diameter_y_mm", "diameter_z_mm",
        "volume_ml",
        "sphericity", "surface_volume_ratio",
        "elongation", "flatness", "max_3d_diameter_mm",
        "num_components"
    ]

    zeros = {col: 0.0 for col in COLUMNS if col not in ["bdmap_id", "organ"]}
    zeros["num_components"] = 0

    bin_mask = mask > 0
    if not bin_mask.any():
        return zeros

    # 1. Connected Components Tracking
    structure = np.ones((3, 3, 3), dtype=bool)
    _, num_components = label(bin_mask, structure=structure)

    # 2. Extract Physical Coordinates for All Voxels
    coords = np.argwhere(bin_mask)
    coords_mm = coords * spacing  # Vectorized conversion to physical space

    # 3. Axis-Aligned Box Diameters
    min_coords = coords_mm.min(axis=0)
    max_coords = coords_mm.max(axis=0)
    # Adding 1 single voxel width to accurately reflect physical boundary span
    diameters = max_coords - min_coords + spacing
    max_x, max_y, max_z = diameters[0], diameters[1], diameters[2]

    # 4. Volume
    voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
    volume_mm3 = len(coords_mm) * voxel_volume_mm3
    volume_ml = volume_mm3 / 1000.0

    # 5. Fast Principle Component Analysis (PCA)
    try:
        centered_coords = coords_mm - coords_mm.mean(axis=0)
        cov = np.cov(centered_coords.T)

        eigvals = np.linalg.eigvals(cov)
        eigvals = np.sort(eigvals)[::-1]  
        eigvals = np.maximum(eigvals, 1e-8)  

        elongation = float(np.sqrt(eigvals[1] / eigvals[0]))
        flatness = float(np.sqrt(eigvals[2] / eigvals[0]))
        max_3d_diameter_mm = float(4.0 * np.sqrt(eigvals[0]))
    except Exception:
        elongation, flatness, max_3d_diameter_mm = 0.0, 0.0, 0.0

    # 6. Standard Surface Area via Marching Cubes
    try:
        padded = np.pad(bin_mask, 1, mode='constant', constant_values=False)
        verts, faces, normals, values = marching_cubes(padded, level=0.5, spacing=spacing)
        surface_area_mm2 = mesh_surface_area(verts, faces)

        surface_volume_ratio = float(surface_area_mm2 / volume_mm3)
        sphericity = float((np.pi ** (1 / 3) * (6 * volume_mm3) ** (2 / 3)) / surface_area_mm2)
        sphericity = min(sphericity, 1.0)

    except Exception:
        # Failsafe for degenerate shapes (e.g., flat 2D slices that cannot be meshed)
        surface_volume_ratio, sphericity = 0.0, 0.0

    return {
        "diameter_x_mm": max_x,
        "diameter_y_mm": max_y,
        "diameter_z_mm": max_z,
        "volume_ml": volume_ml,
        "sphericity": sphericity,
        "surface_volume_ratio": surface_volume_ratio,
        "elongation": elongation,
        "flatness": flatness,
        "max_3d_diameter_mm": max_3d_diameter_mm,
        "num_components": int(num_components)
    }

def prepare_conditional_vector(data, device):
    """
    Extracts tabular features into a single tensor, one-hot encoding the organ.
    Output shape: (Batch, 19) -> 9 organ classes + 10 numerical features
    """
    numerical_features = [
        "diameter_x_mm", "diameter_y_mm", "diameter_z_mm",
        "volume_ml",
        "sphericity", "surface_volume_ratio",
        "elongation", "flatness", "max_3d_diameter_mm",
        "num_components"
    ]

    # 1. Handle the categorical "organ" feature
    organ_idx = torch.as_tensor(
        data["organ"], dtype=torch.long, device=device).view(-1)

    # One-hot encode to shape (Batch, 9) and cast back to float32
    organ_one_hot = F.one_hot(organ_idx, num_classes=9).float()

    # 2. Handle the remaining continuous numerical features
    num_tensors = []
    for key in numerical_features:
        val = torch.as_tensor(
            data[key], dtype=torch.float32, device=device).view(-1)
        num_tensors.append(val)

    # Stack continuous features to shape (Batch, 10)
    continuous_vector = torch.stack(num_tensors, dim=1)

    # 3. Concatenate the one-hot organ with the continuous features
    # Resulting shape: (Batch, 19)
    cond_vector = torch.cat([organ_one_hot, continuous_vector], dim=1)

    return cond_vector

@hydra.main(config_path='config', config_name='base_cfg', version_base=None)
def reconstruct(cfg: DictConfig):
    torch.cuda.set_device(cfg.model.gpus)
    device = torch.device(f"cuda:{cfg.model.gpus}")

    with open_dict(cfg):
        cfg.model.results_folder = os.path.join(
            cfg.model.results_folder, cfg.dataset.name, cfg.model.results_folder_postfix)

    print("1. Initializing Model...")
    model = Unet3D(
        dim=cfg.model.diffusion_img_size,
        dim_mults=cfg.model.dim_mults,
        # target (1) + img_cond (VQ_dim) + organ (1) + feat (1)
        channels=cfg.model.diffusion_num_channels,
        out_dim=1,
        num_continuous_conditioners=10,
        num_organs=9
    ).cuda()

    diffusion = GaussianDiffusion(
        model,
        image_size=cfg.model.diffusion_img_size,
        num_frames=cfg.model.diffusion_depth_size,
        channels=cfg.model.diffusion_num_channels,
        timesteps=cfg.model.timesteps,
        loss_type=cfg.model.loss_type
    ).to(device)

    print("2. Loading Checkpoint...")
    ckpt_path = os.path.join(cfg.model.results_folder, '75.pt')
    data = torch.load(ckpt_path, map_location=device)
    diffusion.load_state_dict(data['ema'])
    diffusion.eval()

    print("3. Loading Data & Diagnosing Labels...")
    train_dataloader, _, _ = get_loader(cfg.dataset)
    loader_iter = iter(train_dataloader)
    train_data = None
    step = 0

    while train_data is None or (train_data["volume_ml"] < 1.045).all():

        try:
            train_data = next(loader_iter)
            print(train_data["volume_ml"])
            step += 1
        except StopIteration:
            print("Dataloader exhausted without finding a valid sample.")
            train_data = None
            break
    print(f"FOUND DATA {step}")

    batch_size = train_data["heatmap"].shape[0]
    tumor_mask_dims = (batch_size, 1, 32, 32, 32)

    heatmap = train_data["heatmap"].permute(0, 1, -1, -3, -2).cuda()
    organ_mask_p = train_data["organ_mask"].permute(0, 1, -1, -3, -2).cuda()
    
    tabular_cond = prepare_conditional_vector(train_data, heatmap.device)
    cond = torch.cat([organ_mask_p, heatmap], dim=1)

    T_START = 1000
    noisy_latent = torch.randn(tumor_mask_dims).cuda()

    # 1. Add no_grad() to prevent memory leaks during inference
    with torch.no_grad():
        for i in tqdm(reversed(range(T_START))):
            t_i = torch.full((batch_size,), i, device=device, dtype=torch.long)
            noisy_latent = diffusion.p_sample(
                noisy_latent, t_i, cond=cond, tabular_cond=tabular_cond, cond_scale=2.0)

        # 2. Reverse the permutation to restore original spatial dimensions!
        recon = noisy_latent.permute(0, 1, -2, -1, -3)

        # 3. Normalize using the correctly oriented tensor
        recon_normalized = (recon + 1.0) / 2.0
        
    raw_np = recon_normalized.cpu().numpy()
    
    tumor_mask = train_data.get("tumor_mask", torch.zeros_like(recon_normalized))
    targets_np = tumor_mask.cpu().numpy().astype(np.uint8)

    debug_folder = Path("inference_masks")
    debug_folder.mkdir(exist_ok=True)

    # --- PHYSICAL SPACING SETUP ---
    # Assuming original model outputs 3mm spacing
    base_spacing = (3.0, 3.0, 3.0) 
    scale_factor = 3
    
    # Calculate the new voxel spacing after upsampling
    post_spacing = (
        base_spacing[0] / scale_factor, 
        base_spacing[1] / scale_factor, 
        base_spacing[2] / scale_factor
    )

    base_affine = np.array([
        [base_spacing[0], 0, 0, 0],
        [0, base_spacing[1], 0, 0],
        [0, 0, base_spacing[2], 0],
        [0, 0, 0, 1]
    ])

    new_affine = base_affine.copy()
    new_affine[:3, :3] /= scale_factor

    with open("dataset_norm_stats.json", "r") as f:
        normalized_stats = json.load(f)
        
        for b_idx in range(min(3, raw_np.shape[0])):
            raw_3d = raw_np[b_idx, 0, :, :, :]
            targ_3d = targets_np[b_idx, 0, :, :, :]

            # --- POST-PROCESSING ---
            # Pass the raw un-thresholded probabilities into the post-processor
            cleaned_pred_3d = postprocess_tensor(
                raw_3d, 
                scale_factor=scale_factor, 
                threshold=0.5, 
                num_components=1
            )

            # Pass the POST_SPACING, not the spatial_shape grid dims
            metrics = compute_diameters_and_coords(cleaned_pred_3d, post_spacing)
            
            print(f"\n===== SAMPLE {b_idx+1} =====")
            for key in metrics.keys():
                # Denormalize the requested target condition
                normalized_conditioner = train_data[key][b_idx].item()
                target_real_val = (normalized_conditioner * normalized_stats[key]["std"]) + normalized_stats[key]["mean"]
                
                print(f"  {key}:")
                print(f"    Requested: {target_real_val:.2f}")
                print(f"    Generated: {metrics[key]:.2f}")
                print(f"    Delta:     {abs(target_real_val - metrics[key]):.2f}")
            print("======================\n")

            # --- SAVE NIFTIS WITH CORRECT AFFINES ---
            # Save the raw 32x32x32 output with base affine
            nib.save(
                nib.Nifti1Image(raw_3d, affine=base_affine),
                str(debug_folder / f"step_inference_sample_{b_idx}_RAW.nii.gz")
            )
            
            # Save the upscaled and cleaned output with new affine
            nib.save(
                nib.Nifti1Image(cleaned_pred_3d, affine=new_affine),
                str(debug_folder / f"step_inference_sample_{b_idx}_CLEANED.nii.gz")
            )

            # Save the ground truth (assumes ground truth is at base resolution)
            nib.save(
                nib.Nifti1Image(targ_3d, affine=base_affine),
                str(debug_folder / f"step_inference_sample_{b_idx}_GT.nii.gz")
            )


if __name__ == '__main__':
    reconstruct()