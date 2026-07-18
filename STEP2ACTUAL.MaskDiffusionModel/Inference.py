from scipy.ndimage import label
from dataset.dataloader import get_loader
import numpy as np
import nibabel as nib
import torch.nn.functional as F
import pandas as pd
import torch
from omegaconf import DictConfig, open_dict
import hydra
from ddpm.ddim import DDIMSampler
import os
from ddpm import Unet3D, GaussianDiffusion, MASK_COLUMNS
from pathlib import Path
from tqdm import tqdm
import json

from metrics import RadiomicsMetricsEvaluator

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
        raise ValueError(
            f"Expected 3D (X,Y,Z) or 4D (B,X,Y,Z) input, got {original_dims}D")

    tensor_mask = tensor_mask.unsqueeze(1)

    if scale_factor != 1:
        tensor_mask = F.interpolate(
            tensor_mask,
            scale_factor=scale_factor,
            mode='nearest'
        )

    binary_mask = (tensor_mask < threshold).to(torch.uint8)

    postprocess_transforms = Compose([
        FillHoles(),
        # KeepLargestConnectedComponent(num_components=num_components)
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


def prepare_conditional_vector(data, device):
    """
    Extracts tabular features into a single tensor, one-hot encoding the organ.
    Output shape: (Batch, 19) -> 9 organ classes + 10 numerical features
    """
    numerical_features = MASK_COLUMNS

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


def generate_samples(train_data, step, diffusion, radiomics_evaluator, cond_scale=2.0, spacing=(4, 4, 4), dim_size=32):

    ddim_sampler = DDIMSampler(diffusion)

    batch_size = train_data["heatmap"].shape[0]
    tumor_mask_dims = (batch_size, 1, dim_size, dim_size, dim_size)

    heatmap = train_data["heatmap"].permute(0, 1, -1, -3, -2).cuda()
    organ_mask_p = train_data["organ_mask"].permute(0, 1, -1, -3, -2).cuda()

    tabular_cond = prepare_conditional_vector(train_data, heatmap.device)
    cond = torch.cat([organ_mask_p, heatmap], dim=1)

    # Use the correct cond_scale parameter in your DDIM sampler!
    img_out, _ = ddim_sampler.sample(
        50, batch_size, (1, dim_size, dim_size, dim_size), conditioning=cond, tabular_cond=tabular_cond, cond_scale=cond_scale
    )

    recon = img_out.permute(0, 1, -2, -1, -3)

    # Normalize from [-1, 1] back to [0, 1]
    recon_normalized = (recon + 1.0) / 2.0

    raw_np = recon_normalized.cpu().numpy()

    tumor_mask = train_data.get(
        "tumor_mask", torch.zeros_like(recon_normalized))
    targets_np = tumor_mask.cpu().numpy().astype(np.uint8)

    debug_folder = Path("inference_masks")
    debug_folder.mkdir(exist_ok=True)

    # --- PHYSICAL SPACING SETUP ---
    base_spacing = spacing
    scale_factor = spacing[0]

    base_affine = np.array([
        [base_spacing[0], 0, 0, 0],
        [0, base_spacing[1], 0, 0],
        [0, 0, base_spacing[2], 0],
        [0, 0, 0, 1]
    ])

    new_affine = base_affine.copy()
    new_affine[:3, :3] /= scale_factor

    output_metrics = []

    with open("dataset_norm_stats.json", "r") as f:
        normalized_stats = json.load(f)

        for b_idx in range(raw_np.shape[0]):
            raw_3d = raw_np[b_idx, 0, :, :, :]

            # --- 1. COMPUTE METRICS AT NATIVE RESOLUTION (Matches Debug) ---
            # Threshold first at 32x32x32 to get the precise mask volume the model intended
            native_binary_mask = (raw_3d < 0.5).astype(np.uint8)

            # Calculate metrics using the BASE spacing (4.0, 4.0, 4.0)
            metrics = radiomics_evaluator.compute_mask(native_binary_mask)

            print(f"\n===== SAMPLE {b_idx+1} (CFG: {cond_scale}) =====")

            for key in metrics.keys():
                # Denormalize the requested target condition
                normalized_conditioner = train_data[key][b_idx].item()
                target_real_val = (
                    normalized_conditioner * normalized_stats[key]["std"]) + normalized_stats[key]["mean"]
                print(f"  {key}:")
                print(f"    Requested: {target_real_val:.2f}")
                print(f"    Generated: {metrics[key]:.2f}")
                print(
                    f"    Delta:     {abs(target_real_val - metrics[key]):.2f}")

                output_metrics.append({
                    "cond_scale": cond_scale,
                    "column": key,
                    "desired_val": target_real_val,
                    "actual_val": metrics[key]
                })

            print("======================\n")

            # --- 2. POST-PROCESSING & UPSAMPLING FOR SAVING ONLY ---
            # Pass to post-processor *only* if you intend to save smooth, high-res NIfTI files
            cleaned_pred_3d = postprocess_tensor(
                raw_3d,
                scale_factor=scale_factor,
                threshold=0.5,
                num_components=1
            )

            # Optional: Uncomment if you want to save out the corrected affines to disk
            # nib.save(
            #     nib.Nifti1Image(raw_3d, affine=base_affine),
            #     str(debug_folder / f"step_stomach_inference_{step}_sample_{b_idx}_cfg_{cond_scale}_RAW.nii.gz")
            # )
            nib.save(
                nib.Nifti1Image(cleaned_pred_3d, affine=new_affine),
                str(debug_folder /
                    f"step_stomach_inference_{step}_sample_{b_idx}_cfg_{cond_scale}_CLEANED.nii.gz")
            )

    return output_metrics


@hydra.main(config_path='config', config_name='base_cfg', version_base=None)
def reconstruct(cfg: DictConfig):
    torch.cuda.set_device(cfg.model.gpus)
    device = torch.device(f"cuda:{cfg.model.gpus}")

    spacing = (cfg.dataset.space_x, cfg.dataset.space_y, cfg.dataset.space_z)

    radiomics_extractor = RadiomicsMetricsEvaluator(MASK_COLUMNS, [], spacing)

    with open_dict(cfg):
        cfg.model.results_folder = os.path.join(
            cfg.model.results_folder, cfg.dataset.name, cfg.model.results_folder_postfix)

    print("1. Initializing Model...")
    model = Unet3D(
        dim=64,
        dim_mults=cfg.model.dim_mults,
        # target (1) + img_cond (VQ_dim) + organ (1) + feat (1)
        channels=cfg.model.diffusion_num_channels,
        out_dim=1,
        num_continuous_conditioners=len(MASK_COLUMNS),
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
    ckpt_path = os.path.join(cfg.model.results_folder, 'model_best.pt')
    data = torch.load(ckpt_path, map_location=device)
    diffusion.load_state_dict(data['ema'])
    diffusion.eval()

    print("3. Loading Data & Diagnosing Labels...")
    val_loader, _, _ = get_loader(cfg.dataset)
    loader_iter = iter(val_loader)
    step = 0
    all_metrics = []

    cond_scales = [1.0, 2.0, 4.0, 6.0]

    for train_data in tqdm(loader_iter):
        print(f"STEP: {step+1}")
        for scale in cond_scales:
            output_metrics = generate_samples(
                train_data, step+1, diffusion, radiomics_extractor, cond_scale=scale, spacing=spacing)
            all_metrics.extend(output_metrics)
        print(all_metrics)
        print("================")

        step += 1
        if (step == 2):
            break

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv("inference_metrics.csv", index=False)


if __name__ == '__main__':
    reconstruct()
