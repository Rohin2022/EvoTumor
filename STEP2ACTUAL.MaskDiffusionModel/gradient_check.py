"""
diagnose_cond_mlp_gradients.py

Runs a handful of real training steps (same data pipeline as training) on
the LONGITUDINAL model and logs gradient norms flowing into cond_mlp,
delta_t_mlp, and a couple of comparison modules (time_mlp, init_conv).
This tells you whether cond_mlp is getting a meaningfully-sized gradient
signal (just needs more steps) or is essentially starved (a real bug /
architectural issue, not a training-duration issue).

Does NOT save any checkpoint changes — this is read-only diagnostics, run
with the optimizer stepping normally but nothing persisted afterward.
"""

import os
import sys
sys.path.append(os.getcwd())

import torch
import torch.nn.functional as F
from torch.optim import Adam
from omegaconf import DictConfig, open_dict
import hydra

from dataset.dataloader import get_loader
from ddpm import Unet3D, GaussianDiffusion


NUM_DIAGNOSTIC_STEPS = 50
MODULES_TO_TRACK = ["cond_mlp", "delta_t_mlp", "time_mlp", "init_conv", "final_conv"]


def get_module_grad_norm(model, module_name):
    module = getattr(model, module_name, None)
    if module is None:
        return None
    total_norm_sq = 0.0
    any_grad = False
    for p in module.parameters():
        if p.grad is not None:
            any_grad = True
            total_norm_sq += p.grad.detach().norm().item() ** 2
    if not any_grad:
        return 0.0
    return total_norm_sq ** 0.5


@hydra.main(config_path='config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
    torch.cuda.set_device(cfg.model.gpus)
    device = torch.device(f"cuda:{cfg.model.gpus}")

    with open_dict(cfg):
        cfg.model.results_folder = os.path.join(
            cfg.model.results_folder, cfg.dataset.name, cfg.model.results_folder_postfix)

    # ------------------------------------------------------------------
    # 1. Build model + diffusion wrapper
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
    # 2. Load checkpoint — continue from current trained state so the
    #    gradient behavior reflects where training actually is, not a
    #    fresh random init
    # ------------------------------------------------------------------
    ckpt_path = os.path.join(cfg.model.results_folder, "model_best.pt")
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    diffusion.load_state_dict(ckpt["model"])  # use raw (non-EMA) weights for live training
    diffusion.train()

    opt = Adam(diffusion.parameters(), lr=cfg.model.train_lr)

    # ------------------------------------------------------------------
    # 3. Data
    # ------------------------------------------------------------------
    with open_dict(cfg):
        cfg.dataset.phase = "train"
    dataloader, _, _ = get_loader(cfg.dataset)
    data_iter = iter(dataloader)

    # ------------------------------------------------------------------
    # 4. Run diagnostic steps, log grad norms per tracked module
    # ------------------------------------------------------------------
    history = {name: [] for name in MODULES_TO_TRACK}
    loss_history = []

    print(f"\nRunning {NUM_DIAGNOSTIC_STEPS} diagnostic training steps...\n")
    print(f"{'step':>5} | {'loss':>10} | " + " | ".join(f"{m:>14}" for m in MODULES_TO_TRACK))

    for step in range(NUM_DIAGNOSTIC_STEPS):
        try:
            data = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            data = next(data_iter)

        tumor_mask_0 = data['tumor_mask_0'].to(device)
        tumor_mask_1 = data['tumor_mask_1'].to(device)
        organ_mask_0 = data['organ_mask_0'].to(device)
        heatmap = data['heatmap'].to(device)
        delta_t = data['delta_t'].to(device).float()
        organ_id = torch.as_tensor(data['organ_id'], dtype=torch.long, device=device).view(-1)
        organ = F.one_hot(organ_id, num_classes=9).float()

        opt.zero_grad()

        loss = diffusion(
            heatmap=heatmap,
            tumor_mask_0=tumor_mask_0,
            tumor_mask_1=tumor_mask_1,
            organ_mask_0=organ_mask_0,
            delta_t=delta_t,
            organ=organ,
            null_cond_prob=0.1,
        )

        loss.backward()

        # Log grad norms BEFORE opt.step() clears/updates anything
        row = []
        for module_name in MODULES_TO_TRACK:
            grad_norm = get_module_grad_norm(model, module_name)
            history[module_name].append(grad_norm)
            row.append(f"{grad_norm:>14.6f}" if grad_norm is not None else f"{'N/A':>14}")

        loss_history.append(loss.item())
        print(f"{step:>5} | {loss.item():>10.6f} | " + " | ".join(row))

        opt.step()

    # ------------------------------------------------------------------
    # 5. Summary statistics
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY: mean / std / min / max gradient norm per module")
    print("=" * 70)
    print(f"{'module':>14} | {'mean':>10} | {'std':>10} | {'min':>10} | {'max':>10}")

    for module_name in MODULES_TO_TRACK:
        vals = torch.tensor([v for v in history[module_name] if v is not None])
        if len(vals) == 0:
            print(f"{module_name:>14} | module not found")
            continue
        print(f"{module_name:>14} | {vals.mean().item():>10.6f} | {vals.std().item():>10.6f} | "
              f"{vals.min().item():>10.6f} | {vals.max().item():>10.6f}")

    print("\n--- Ratio of cond_mlp grad norm to time_mlp / init_conv grad norm ---")
    cond_vals = torch.tensor([v for v in history["cond_mlp"] if v is not None])
    for compare_to in ["time_mlp", "init_conv", "final_conv"]:
        compare_vals = torch.tensor([v for v in history[compare_to] if v is not None])
        if len(compare_vals) == 0 or compare_vals.mean().item() == 0:
            continue
        ratio = cond_vals.mean().item() / compare_vals.mean().item()
        print(f"cond_mlp / {compare_to}: {ratio:.6f}")

    print("\n(ratio << 1 => cond_mlp is getting a much weaker gradient signal "
          "than the rest of the network, consistent with conditioning starvation. "
          "If the ratio is closer to comparable modules' typical range, cond_mlp "
          "is training, just slowly — more steps may resolve it on their own.)")


if __name__ == '__main__':
    run()