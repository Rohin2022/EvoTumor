"""
eval_negative_delta_t.py

Loads a trained EvoTumor STEP3 diffusion checkpoint and, for each sample
pulled from the dataloader (batch_size=1), duplicates that sample across a
sweep of delta_t values (including negative/backward-in-time deltas) into
a single batch, then runs one shared reverse-diffusion pass. All duplicates
within a batch start from the SAME initial noise, so any difference in
output across the delta_t sweep is attributable to conditioning alone, not
sampling variance. Iterates over the full dataloader and saves every
generated volume.
"""

import os
import sys
sys.path.append(os.getcwd())

import json
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from omegaconf import DictConfig, open_dict
import hydra

from dataset.dataloader import get_loader
from ddpm import Unet3D, GaussianDiffusion


# Normalized delta_t values to sweep. Since normalized_time_delta is
# presumably z-scored (mean 0, std 1) rather than min-max scaled, small
# integer-ish values going negative correspond to "further into the past
# than the training distribution typically saw." Adjust range once you've
# checked dataset_norm_stats.json (if you kept one) or the raw column's
# mean/std to know what a "1 year back" equivalent actually looks like in
# normalized units.
DELTA_T_SWEEP = [3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0]

# Fixed seed so the initial noise is reproducible across separate script runs.
NOISE_SEED = 42

# Cap on how many dataset items to run the sweep over. None = whole dataset.
MAX_ITEMS = None


@hydra.main(config_path='config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
    torch.cuda.set_device(cfg.model.gpus)

    with open_dict(cfg):
        cfg.model.results_folder = os.path.join(
            cfg.model.results_folder, cfg.dataset.name, cfg.model.results_folder_postfix)

    device = torch.device(f"cuda:{cfg.model.gpus}")

    # ------------------------------------------------------------------
    # 1. Build model + diffusion wrapper (must match training config)
    # ------------------------------------------------------------------
    if cfg.model.denoising_fn == 'Unet3D':
        model = Unet3D(
            dim=64,
            dim_mults=cfg.model.dim_mults,
            channels=cfg.model.diffusion_num_channels,
            out_dim=1,
            num_organs=9,
        ).cuda()
    else:
        raise ValueError(f"Model {cfg.model.denoising_fn} doesn't exist")

    diffusion = GaussianDiffusion(
        model,
        image_size=cfg.model.diffusion_img_size,
        num_frames=cfg.model.diffusion_depth_size,
        channels=cfg.model.diffusion_num_channels,
        timesteps=cfg.model.timesteps,
        loss_type=cfg.model.loss_type,
    ).cuda()

    # ------------------------------------------------------------------
    # 2. Load checkpoint (prefer EMA weights for eval)
    # ------------------------------------------------------------------
    ckpt_dir = Path(cfg.model.results_folder)
    ckpt_path = ckpt_dir / f"model_best.pt"
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    diffusion.load_state_dict(ckpt["ema"])
    diffusion.eval()

    # cfg.dataset.batch_size should be 1 for this script — assert to be safe
    assert cfg.dataset.batch_size == 1, (
        "This script expects batch_size=1 in cfg.dataset — each dataloader "
        "item is duplicated internally across the delta_t sweep instead."
    )

    dataloader, _, dataset_size = get_loader(cfg.dataset)

    space_x, space_y, space_z = cfg.dataset.space_x, cfg.dataset.space_y, cfg.dataset.space_z
    affine = np.array([
        [space_x, 0, 0, 0],
        [0, space_y, 0, 0],
        [0, 0, space_z, 0],
        [0, 0, 0, 1],
    ])

    out_folder = ckpt_dir / 'eval_negative_delta_t'
    out_folder.mkdir(exist_ok=True, parents=True)

    num_deltas = len(DELTA_T_SWEEP)
    T_START = diffusion.num_timesteps
    cond_scale = 6.0

    n_items = len(dataloader) if MAX_ITEMS is None else min(MAX_ITEMS, len(dataloader))
    print(f"Running delta_t sweep over {n_items} dataset items, "
          f"{num_deltas} delta_t values each.")

    with torch.no_grad():
        for item_idx, data in enumerate(tqdm(dataloader, total=n_items, desc="dataset items")):
            if MAX_ITEMS is not None and item_idx >= MAX_ITEMS:
                break

            item_folder = out_folder / f"item_{item_idx}"
            item_folder.mkdir(exist_ok=True, parents=True)

            # --- Pull the single sample and duplicate it num_deltas times ---
            tumor_mask_0 = data['tumor_mask_0'].to(device)   # (1, C, D, H, W)
            organ_mask_0 = data['organ_mask_0'].to(device)
            heatmap = data['heatmap'].to(device)
            organ_id = torch.as_tensor(
                data['organ_id'], dtype=torch.long, device=device).view(-1)
            organ_one_hot = F.one_hot(organ_id, num_classes=9).float()  # (1, 9)

            tumor_mask_0_rep = tumor_mask_0.repeat(num_deltas, 1, 1, 1, 1)
            organ_mask_0_rep = organ_mask_0.repeat(num_deltas, 1, 1, 1, 1)
            heatmap_rep = heatmap.repeat(num_deltas, 1, 1, 1, 1)
            organ_one_hot_rep = organ_one_hot.repeat(num_deltas, 1)

            delta_t_batch = torch.tensor(
                DELTA_T_SWEEP, dtype=torch.float32, device=device)  # (num_deltas,)

            cond = torch.cat([
                tumor_mask_0_rep.permute(0, 1, -1, -3, -2),
                organ_mask_0_rep.permute(0, 1, -1, -3, -2),
                heatmap_rep.permute(0, 1, -1, -3, -2),
            ], dim=1)  # (num_deltas, C, D, H, W)

            # --- Save the real ct0 tumor mask once per item (not per delta) ---
            real_t0 = tumor_mask_0[0, 0].cpu().numpy()
            nib.save(
                nib.Nifti1Image(real_t0, affine=affine),
                str(item_folder / "REAL_tumor_mask_0.nii.gz"),
            )

            # --- Shared initial noise across all delta_t duplicates in this batch ---
            noise_generator = torch.Generator(device=device).manual_seed(NOISE_SEED + item_idx)
            single_shape = cond[:1, :1].shape  # single-channel target, one sample
            base_noise = torch.randn(
                single_shape, generator=noise_generator, device=device)
            recon_latent = base_noise.repeat(num_deltas, 1, 1, 1, 1)

            # --- Single reverse-diffusion pass covering the whole delta_t sweep ---
            for i in tqdm(reversed(range(T_START)), total=T_START,
                          desc=f"item {item_idx} sampling", leave=False):
                t_i = torch.full(
                    (num_deltas,), i, device=device, dtype=torch.long)
                recon_latent = diffusion.p_sample(
                    recon_latent, t_i,
                    cond=cond, delta_t=delta_t_batch, organ=organ_one_hot_rep,
                    cond_scale=cond_scale,
                )

            recon = recon_latent.permute(0, 1, -2, -1, -3)
            recon_normalized = (recon + 1.0) / 2.0
            generated_masks = (recon_normalized < 0.5).float()

            masks_np = generated_masks.cpu().numpy().astype(np.uint8)
            raw_np = recon_normalized.cpu().numpy()

            # --- Save one output pair per delta_t value ---
            for d_idx, delta_t_val in enumerate(DELTA_T_SWEEP):
                delta_tag = f"{delta_t_val:+.2f}".replace(".", "p")

                pred_3d = masks_np[d_idx, 0]
                raw_3d = raw_np[d_idx, 0]

                nib.save(
                    nib.Nifti1Image(pred_3d, affine=affine),
                    str(item_folder / f"delta_t_{delta_tag}_RECON.nii.gz"),
                )
                nib.save(
                    nib.Nifti1Image(raw_3d, affine=affine),
                    str(item_folder / f"delta_t_{delta_tag}_RAW_RECON.nii.gz"),
                )

    print(f"Done. Outputs written to {out_folder}")


if __name__ == '__main__':
    run()