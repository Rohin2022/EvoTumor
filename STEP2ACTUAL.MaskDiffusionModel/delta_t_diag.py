"""
diagnose_delta_t_emb.py

Follow-up diagnostic after fixing the cond_mlp zero-init gate. cond_mlp is
now training at parity with time_mlp, but delta_t_mlp is still ~196x
weaker than time_mlp -- i.e. gradient is reaching cond_mlp fine, but not
propagating one hop further back to delta_t_mlp. This script distinguishes
between two possible causes:

  (A) delta_t_emb itself carries little information (low variance across
      the batch after the *300 scaling + SinusoidalPosEmb), so cond_mlp
      has nothing useful to grab onto even if it wanted to.

  (B) delta_t_emb is fine, but cond_mlp has learned to weight the organ
      one-hot columns much more heavily than the delta_t_emb columns in
      its first linear layer -- i.e. it's ignoring delta_t_emb by choice,
      not because it can't see it.

Read-only: loads the checkpoint, runs a few real batches through
delta_t_mlp / cond_mlp only (no backward, no optimizer step), and reports:

  1. Per-dimension variance (and overall mean variance) of delta_t_emb
     across the batch, for a handful of batches.
  2. A same-computation comparison against time_mlp's output variance
     (on the diffusion timestep), as a reference for "healthy" variance.
  3. The weight-norm split of cond_mlp[0].weight between the columns that
     read from delta_t_emb vs. the columns that read from organ.
  4. The gradient-norm split of the same, accumulated over a few real
     backward passes (so it reflects live signal, not just weight magnitude).
"""

import os
import sys
sys.path.append(os.getcwd())

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, open_dict
import hydra

from dataset.dataloader import get_loader
from ddpm import Unet3D, GaussianDiffusion


NUM_BATCHES = 10


@hydra.main(config_path='config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
    torch.cuda.set_device(cfg.model.gpus)
    device = torch.device(f"cuda:{cfg.model.gpus}")

    with open_dict(cfg):
        cfg.model.results_folder = os.path.join(
            cfg.model.results_folder, cfg.dataset.name, cfg.model.results_folder_postfix)

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

    ckpt_path = os.path.join(cfg.model.results_folder, "model_best.pt")
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    diffusion.load_state_dict(ckpt["model"])
    diffusion.train()

    with open_dict(cfg):
        cfg.dataset.phase = "train"
    dataloader, _, _ = get_loader(cfg.dataset)
    data_iter = iter(dataloader)

    time_dim = model.time_mlp[1].out_features  # dim * 4
    num_organs = model.num_organs

    # ------------------------------------------------------------------
    # Part 1: forward-only variance check of delta_t_emb vs time_mlp output
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PART 1: delta_t_emb variance across batch (forward-only, no grad)")
    print("=" * 70)

    all_delta_t_emb = []
    all_time_emb = []
    all_delta_t_raw = []

    with torch.no_grad():
        for step in range(NUM_BATCHES):
            try:
                data = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                data = next(data_iter)

            delta_t = data['delta_t'].to(device).float().view(-1)
            batch = delta_t.shape[0]

            # match training-time scaling
            delta_t_emb = model.delta_t_mlp(delta_t * 300)
            all_delta_t_emb.append(delta_t_emb.cpu())
            all_delta_t_raw.append(delta_t.cpu())

            # reference: a random plausible diffusion timestep batch
            rand_t = torch.randint(0, cfg.model.timesteps, (batch,), device=device).float()
            time_emb = model.time_mlp(rand_t)
            all_time_emb.append(time_emb.cpu())

    delta_t_emb_cat = torch.cat(all_delta_t_emb, dim=0)   # (N, time_dim)
    time_emb_cat = torch.cat(all_time_emb, dim=0)         # (N, time_dim)
    delta_t_raw_cat = torch.cat(all_delta_t_raw, dim=0)   # (N,)

    print(f"\nRaw delta_t stats (before *300 scaling):")
    print(f"  min={delta_t_raw_cat.min().item():.6f}  max={delta_t_raw_cat.max().item():.6f}  "
          f"mean={delta_t_raw_cat.mean().item():.6f}  std={delta_t_raw_cat.std().item():.6f}")
    print(f"Scaled (delta_t * 300) stats:")
    scaled = delta_t_raw_cat * 300
    print(f"  min={scaled.min().item():.6f}  max={scaled.max().item():.6f}  "
          f"mean={scaled.mean().item():.6f}  std={scaled.std().item():.6f}")
    print(f"  (compare to diffusion timesteps range: 0-{cfg.model.timesteps})")

    delta_t_emb_var_per_dim = delta_t_emb_cat.var(dim=0)  # (time_dim,)
    time_emb_var_per_dim = time_emb_cat.var(dim=0)

    print(f"\ndelta_t_emb per-dim variance:  mean={delta_t_emb_var_per_dim.mean().item():.8f}  "
          f"std={delta_t_emb_var_per_dim.std().item():.8f}  "
          f"min={delta_t_emb_var_per_dim.min().item():.8f}  max={delta_t_emb_var_per_dim.max().item():.8f}")
    print(f"time_mlp(rand_t) per-dim variance:  mean={time_emb_var_per_dim.mean().item():.8f}  "
          f"std={time_emb_var_per_dim.std().item():.8f}  "
          f"min={time_emb_var_per_dim.min().item():.8f}  max={time_emb_var_per_dim.max().item():.8f}")

    var_ratio = delta_t_emb_var_per_dim.mean().item() / max(time_emb_var_per_dim.mean().item(), 1e-12)
    print(f"\nvariance ratio (delta_t_emb / time_emb): {var_ratio:.6f}")
    print("(if this ratio is << 1, delta_t_emb carries much less information "
          "per-dimension than time_mlp's output -- consistent with cause (A): "
          "delta_t_emb itself is close to a constant vector regardless of the "
          "actual delta_t value, so cond_mlp has little to learn from it.)")

    # fraction of dead/near-constant dims
    dead_dims = (delta_t_emb_var_per_dim < 1e-6).sum().item()
    print(f"\ndelta_t_emb dims with variance < 1e-6 across batch: {dead_dims} / {time_dim}")

    # ------------------------------------------------------------------
    # Part 2: cond_mlp[0].weight column split -- delta_t_emb vs organ
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PART 2: cond_mlp[0].weight magnitude split (delta_t_emb cols vs organ cols)")
    print("=" * 70)

    W = model.cond_mlp[0].weight.detach()  # shape (time_dim_out, time_dim + num_organs)
    delta_t_cols = W[:, :time_dim]
    organ_cols = W[:, time_dim:time_dim + num_organs]

    print(f"\ncond_mlp[0].weight shape: {tuple(W.shape)}")
    print(f"  delta_t_emb-connected columns ({time_dim} cols): "
          f"weight norm = {delta_t_cols.norm().item():.6f}, "
          f"mean|w| = {delta_t_cols.abs().mean().item():.6f}")
    print(f"  organ-connected columns ({num_organs} cols): "
          f"weight norm = {organ_cols.norm().item():.6f}, "
          f"mean|w| = {organ_cols.abs().mean().item():.6f}")

    # normalize by number of columns to compare "per-column" magnitude fairly
    per_col_delta_t = delta_t_cols.abs().mean().item()
    per_col_organ = organ_cols.abs().mean().item()
    print(f"\nper-column mean|w| ratio (delta_t / organ): {per_col_delta_t / max(per_col_organ, 1e-12):.6f}")
    print("(if << 1, cond_mlp has learned much smaller weights on delta_t_emb "
          "columns than organ columns -- consistent with cause (B): cond_mlp "
          "sees delta_t_emb fine, but has learned to mostly ignore it.)")

    # ------------------------------------------------------------------
    # Part 3: live gradient split over a few real backward passes
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PART 3: live gradient-norm split (delta_t_emb cols vs organ cols), "
          f"{NUM_BATCHES} real backward passes")
    print("=" * 70)

    delta_t_grad_norms = []
    organ_grad_norms = []

    for step in range(NUM_BATCHES):
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

        model.zero_grad(set_to_none=True)

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

        if model.cond_mlp[0].weight.grad is not None:
            g = model.cond_mlp[0].weight.grad.detach()
            delta_t_grad_norms.append(g[:, :time_dim].norm().item())
            organ_grad_norms.append(g[:, time_dim:time_dim + num_organs].norm().item())

    if delta_t_grad_norms:
        dt_g = torch.tensor(delta_t_grad_norms)
        og_g = torch.tensor(organ_grad_norms)
        print(f"\ndelta_t_emb-connected cols grad norm: mean={dt_g.mean().item():.6f}  std={dt_g.std().item():.6f}")
        print(f"organ-connected cols grad norm:       mean={og_g.mean().item():.6f}  std={og_g.std().item():.6f}")
        print(f"\nratio (delta_t / organ): {dt_g.mean().item() / max(og_g.mean().item(), 1e-12):.6f}")
        print("(if << 1, this confirms live gradient into cond_mlp is dominated "
              "by organ, not delta_t_emb -- cause (B).)")
    else:
        print("No gradient found on cond_mlp[0].weight -- check requires_grad / autograd graph.")

    print("\n" + "=" * 70)
    print("HOW TO READ THIS TOGETHER")
    print("=" * 70)
    print("""
  - Low delta_t_emb variance (Part 1) + comparable weight/grad split (Part 2/3)
    => cause (A): re-check the *300 scaling / SinusoidalPosEmb frequency range,
       delta_t_emb has little to say regardless of what cond_mlp does with it.

  - Healthy delta_t_emb variance (Part 1, ratio near 1) + skewed weight/grad
    split toward organ (Part 2/3, ratio << 1)
    => cause (B): cond_mlp can see delta_t_emb fine but has learned to lean on
       organ instead. Consider separate projection heads for delta_t_emb and
       organ (summed rather than concatenated+shared-first-layer), or an
       auxiliary loss that forces delta_t_emb to be predictive on its own.

  - Both healthy => the bottleneck is likely downstream of cond_mlp, e.g. the
    ResnetBlock's time_emb_dim projection washing out the delta_t component
    relative to the diffusion timestep component within `t`.
""")


if __name__ == '__main__':
    run()