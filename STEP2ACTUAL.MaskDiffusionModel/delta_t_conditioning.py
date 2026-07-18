"""
diagnose_delta_t_embedding.py

Loads model_best.pt via the same hydra config as training, then probes the
Unet3D's delta_t embedding pathway in isolation:
  1. delta_t_mlp(delta_t) — does the sinusoidal + linear embedding actually
     vary across different delta_t values, or is it ~constant?
  2. cond_mlp([delta_t_emb, organ_one_hot]) — does the combined embedding
     fed into the timestep embedding (t = t + tab_emb) carry any signal,
     or has it stayed near its zero-init?

This isolates whether delta_t conditioning is dead at the embedding stage
(sinusoidal frequency scale mismatch) vs. the fusion/training stage
(cond_mlp never learned to move away from zero-init).
"""

import os
import sys
sys.path.append(os.getcwd())

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, open_dict
import hydra

from ddpm import Unet3D, GaussianDiffusion


# Sweep of normalized delta_t values to probe. Widen/narrow this once you
# know the actual range of normalized_time_delta in your training data.
DELTA_T_PROBE = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]

# Organ id to probe with (index into ORGAN_MAPPING). Doesn't matter much
# for this diagnostic, just needs to be a valid one-hot.
PROBE_ORGAN_ID = 0


@hydra.main(config_path='config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
    torch.cuda.set_device(cfg.model.gpus)

    with open_dict(cfg):
        cfg.model.results_folder = os.path.join(
            cfg.model.results_folder, cfg.dataset.name, cfg.model.results_folder_postfix)

    device = torch.device(f"cuda:{cfg.model.gpus}")

    # ------------------------------------------------------------------
    # 1. Build model (must match training config)
    # ------------------------------------------------------------------
    model = Unet3D(
        dim=64,
        dim_mults=cfg.model.dim_mults,
        channels=cfg.model.diffusion_num_channels,
        out_dim=1,
        num_organs=9,
    ).cuda()

    diffusion = GaussianDiffusion(
        model,
        image_size=cfg.model.diffusion_img_size,
        num_frames=cfg.model.diffusion_depth_size,
        channels=cfg.model.diffusion_num_channels,
        timesteps=cfg.model.timesteps,
        loss_type=cfg.model.loss_type,
    ).cuda()

    # ------------------------------------------------------------------
    # 2. Load checkpoint (EMA weights)
    # ------------------------------------------------------------------
    ckpt_path = os.path.join(cfg.model.results_folder, "model_best.pt")
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    diffusion.load_state_dict(ckpt["ema"])
    diffusion.eval()

    unet = diffusion.denoise_fn  # the Unet3D instance

    # ------------------------------------------------------------------
    # 3. Build the delta_t probe batch
    # ------------------------------------------------------------------
    delta_t = torch.tensor(DELTA_T_PROBE, dtype=torch.float32, device=device)
    batch = delta_t.shape[0]

    organ_id = torch.full((batch,), PROBE_ORGAN_ID, dtype=torch.long, device=device)
    organ_one_hot = F.one_hot(organ_id, num_classes=9).float()

    with torch.no_grad():
        # --- Stage 1: raw delta_t_mlp output (sinusoidal emb -> linear proj) ---
        delta_t_emb = unet.delta_t_mlp(delta_t)  # (batch, time_dim)

        # --- Stage 2: combined cond_mlp output (delta_t_emb + organ one-hot) ---
        cond_input = torch.cat([delta_t_emb, organ_one_hot], dim=1)
        tab_emb = unet.cond_mlp(cond_input)  # (batch, time_dim)

    print("\n=== delta_t_mlp output (Stage 1: sinusoidal + linear) ===")
    print(f"{'delta_t':>10} | {'emb norm':>10} | {'L2 dist from delta_t=0.0':>26}")
    ref_idx = DELTA_T_PROBE.index(0.0) if 0.0 in DELTA_T_PROBE else 0
    ref_emb = delta_t_emb[ref_idx]
    for i, dt in enumerate(DELTA_T_PROBE):
        norm = delta_t_emb[i].norm().item()
        dist = (delta_t_emb[i] - ref_emb).norm().item()
        print(f"{dt:>10.2f} | {norm:>10.4f} | {dist:>26.4f}")

    print(f"\nAcross-batch std of delta_t_emb (mean over dims): "
          f"{delta_t_emb.std(dim=0).mean().item():.6f}")
    print("  (near-zero => embedding barely varies with delta_t: "
          "likely a sinusoidal frequency-scale mismatch given your "
          "normalized_time_delta range)")

    print("\n=== cond_mlp output (Stage 2: delta_t_emb + organ -> tab_emb) ===")
    print(f"{'delta_t':>10} | {'tab_emb norm':>12} | {'L2 dist from delta_t=0.0':>26}")
    ref_tab = tab_emb[ref_idx]
    for i, dt in enumerate(DELTA_T_PROBE):
        norm = tab_emb[i].norm().item()
        dist = (tab_emb[i] - ref_tab).norm().item()
        print(f"{dt:>10.2f} | {norm:>12.4f} | {dist:>26.4f}")

    print(f"\nAcross-batch std of tab_emb (mean over dims): "
          f"{tab_emb.std(dim=0).mean().item():.6f}")
    print(f"Overall tab_emb norm (mean over batch): "
          f"{tab_emb.norm(dim=1).mean().item():.6f}")
    print("  (norm near 0 => cond_mlp hasn't moved much from its zero-init "
          "after training: fusion/gradient-flow issue)")
    print("  (norm sizable but std-across-delta_t near 0 => cond_mlp learned "
          "something, but it's dominated by the organ one-hot, not delta_t)")

    # ------------------------------------------------------------------
    # 4. Isolate delta_t's contribution vs organ's contribution to tab_emb
    # ------------------------------------------------------------------
    with torch.no_grad():
        # Fix organ, vary delta_t (already have this: tab_emb)
        # Now: fix delta_t at 0, vary organ, to compare magnitude of each signal
        fixed_delta_t = torch.zeros(9, device=device)
        all_organs = torch.arange(9, device=device)
        organ_one_hot_all = F.one_hot(all_organs, num_classes=9).float()

        delta_t_emb_fixed = unet.delta_t_mlp(fixed_delta_t)
        cond_input_organ_sweep = torch.cat([delta_t_emb_fixed, organ_one_hot_all], dim=1)
        tab_emb_organ_sweep = unet.cond_mlp(cond_input_organ_sweep)

    print("\n=== Sanity check: organ-only sweep (delta_t fixed at 0.0) ===")
    print(f"Across-organ std of tab_emb (mean over dims): "
          f"{tab_emb_organ_sweep.std(dim=0).mean().item():.6f}")
    print("Compare this to the delta_t-sweep std above — if organ's std is "
          "much larger than delta_t's, cond_mlp is mostly organ-driven and "
          "delta_t's contribution is comparatively negligible.")


if __name__ == '__main__':
    run()