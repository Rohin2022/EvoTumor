import sys
import os

sys.path.append(os.getcwd())

import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf
import torch
import json
import numpy as np

from dataset.dataloader import get_longitudinal_loader
from vq_gan_3d.model.vqgan import VQGAN


# -----------------------------------------------------------------------------
# Compose the same Hydra config used by run.py
# -----------------------------------------------------------------------------

with initialize(version_base=None, config_path="config"):
    cfg = compose(
        config_name="base_cfg",
    )

print(OmegaConf.to_yaml(cfg))


device = torch.device(f"cuda:{cfg.model.gpus}")
torch.cuda.set_device(device)

num_batches = (
    cfg.diagnostic.num_batches
    if "diagnostic" in cfg and "num_batches" in cfg.diagnostic
    else 465
)

with torch.no_grad():
    print("Building longitudinal dataloader...")
    train_loader, _, dataset_size = get_longitudinal_loader(cfg.dataset)
    print(f"Dataset has {dataset_size} pairs.")

    print(f"Loading VQGAN from {cfg.model.vqgan_ckpt} ...")
    vqgan = VQGAN.load_from_checkpoint(
        cfg.model.vqgan_ckpt,
        weights_only=False,
    ).to(device)
    vqgan.eval()

    emb_min = vqgan.codebook.embeddings.min().item()
    emb_max = vqgan.codebook.embeddings.max().item()
    emb_denom = emb_max - emb_min

    print(
        f"Codebook embedding range: "
        f"min={emb_min:.4f}, "
        f"max={emb_max:.4f}, "
        f"denom={emb_denom:.4f}"
    )

    z0_vals = []
    z1_vals = []
    residual_vals = []

    # Per-pair summary stats only (NOT full flattened tensors — those would
    # blow up FileStats.json for 450+ pairs at latent resolution). If you
    # need the raw per-voxel values for something downstream, say so and
    # we can switch this to e.g. per-pair .pt files instead of one big json.
    per_pair_stats = []

    residual_abs_max = 0.0
    per_batch_residual_std = []

    n_batches_seen = 0

    for batch in train_loader:
        print("1 iteration")
        ct0 = batch["ct0"].to(device)
        ct1 = batch["ct1"].to(device)

        # Match your VQGAN input ordering
        ct0 = ct0.permute(0, 1, -1, -3, -2)
        ct1 = ct1.permute(0, 1, -1, -3, -2)

        z0 = vqgan.encode(ct0, quantize=False, include_embeddings=True)
        z1 = vqgan.encode(ct1, quantize=False, include_embeddings=True)

        residual = z1 - z0

        z0_flat = z0.flatten().float().cpu()
        z1_flat = z1.flatten().float().cpu()
        residual_flat = residual.flatten().float().cpu()

        z0_vals.append(z0_flat)
        z1_vals.append(z1_flat)
        residual_vals.append(residual_flat)

        # batch may contain multiple pairs if batch_size > 1 — handle both
        # list-like (post list_data_collate) and scalar cases per-sample.
        ct0_ids = batch["ct0_bdmap"]
        ct1_ids = batch["ct1_bdmap"]
        if not isinstance(ct0_ids, (list, tuple)):
            ct0_ids = [ct0_ids]
            ct1_ids = [ct1_ids]

        b = residual.shape[0]
        for i in range(b):
            res_i = residual[i].flatten().float().cpu()
            z0_i = z0[i].flatten().float().cpu()
            z1_i = z1[i].flatten().float().cpu()
            pair_id = f"{ct0_ids[i]}_to_{ct1_ids[i]}"
            per_pair_stats.append({
                "pair_id": pair_id,
                "ct0_bdmap": str(ct0_ids[i]),
                "ct1_bdmap": str(ct1_ids[i]),
                "residual_mean": res_i.mean().item(),
                "residual_std": res_i.std().item(),
                "residual_abs_max": res_i.abs().max().item(),
                "z0_std": z0_i.std().item(),
                "z1_std": z1_i.std().item(),
            })

        residual_abs_max = max(residual_abs_max, residual.abs().max().item())
        per_batch_residual_std.append(residual_flat.std().item())

        n_batches_seen += 1
        if n_batches_seen % 10 == 0:
            print(f"...processed {n_batches_seen} batches")

z0_all = torch.cat(z0_vals)
z1_all = torch.cat(z1_vals)
residual_all = torch.cat(residual_vals)

residual_mean = residual_all.mean().item()
residual_std = residual_all.std().item()

print(f"\n=== Latent statistics ({n_batches_seen} batches) ===")
print(
    f"z0: mean={z0_all.mean():.4f} std={z0_all.std():.4f} "
    f"min={z0_all.min():.4f} max={z0_all.max():.4f}"
)
print(
    f"z1: mean={z1_all.mean():.4f} std={z1_all.std():.4f} "
    f"min={z1_all.min():.4f} max={z1_all.max():.4f}"
)
print(
    f"Residual: mean={residual_mean:.4f} std={residual_std:.4f} "
    f"abs_max={residual_abs_max:.4f}"
)
print(
    f"Per-batch residual std range: "
    f"{min(per_batch_residual_std):.4f} - {max(per_batch_residual_std):.4f}"
)

print("\n=== Scale comparison ===")
print(f"Codebook range: {emb_denom:.4f}")
print(f"Residual std : {residual_std:.4f}")
ratio = residual_std / emb_denom if emb_denom != 0 else float("nan")
print(f"Residual std / codebook range = {ratio:.6f}")

scale_factor = 1.0 / residual_std if residual_std != 0 else float("nan")
print("\n=== Multiplicative scale factor (Stable-Diffusion style) ===")
print(f"scale_factor = 1 / residual_std = {scale_factor:.6f}")

print("\nSuggested normalization constants (z-score, if you want them too):")
print(f"residual_mean = {residual_mean:.6f}")
print(f"residual_std  = {residual_std:.6f}")

# -----------------------------------------------------------------------------
# Sort pairs by residual std — cheapest way to spot likely registration
# failures (either extreme: near-zero std can mean a collapsed/degenerate
# warp aligning flat background; very high std can mean gross misalignment
# rather than true anatomical change).
# -----------------------------------------------------------------------------

sorted_by_std = sorted(per_pair_stats, key=lambda p: p["residual_std"])

n_show = min(10, len(sorted_by_std))
print(f"\n=== {n_show} LOWEST residual-std pairs (check for degenerate/collapsed registration) ===")
for p in sorted_by_std[:n_show]:
    print(
        f"  {p['pair_id']:60s} residual_std={p['residual_std']:.4f} "
        f"abs_max={p['residual_abs_max']:.4f}"
    )

print(f"\n=== {n_show} HIGHEST residual-std pairs (check for gross misalignment) ===")
for p in sorted_by_std[-n_show:][::-1]:
    print(
        f"  {p['pair_id']:60s} residual_std={p['residual_std']:.4f} "
        f"abs_max={p['residual_abs_max']:.4f}"
    )

# Trimmed stats (drop extreme 1% of pairs by residual_std) for comparison
# against the raw global stats above — if these are close, raw is fine;
# if they diverge a lot, a meaningful chunk of your data is likely noisy
# registrations skewing the global scale factor.
stds_sorted = sorted(p["residual_std"] for p in per_pair_stats)
n_trim = max(1, int(0.01 * len(stds_sorted)))
trimmed = stds_sorted[n_trim:-n_trim] if len(stds_sorted) > 2 * n_trim else stds_sorted
trimmed_mean_std = float(np.mean(trimmed))
print(f"\n=== Trimmed comparison (dropped top/bottom {n_trim} pairs by residual_std) ===")
print(f"Mean per-pair residual_std, trimmed: {trimmed_mean_std:.6f}")
print(f"Mean per-pair residual_std, raw:     {float(np.mean(stds_sorted)):.6f}")
print(f"Global residual_std (all voxels pooled): {residual_std:.6f}")

# -----------------------------------------------------------------------------
# Save per-pair stats (lightweight) + summary to disk
# -----------------------------------------------------------------------------


def convert(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


output = {
    "summary": {
        "n_pairs": n_batches_seen,
        "z0_mean": z0_all.mean().item(),
        "z0_std": z0_all.std().item(),
        "z1_mean": z1_all.mean().item(),
        "z1_std": z1_all.std().item(),
        "residual_mean": residual_mean,
        "residual_std": residual_std,
        "residual_abs_max": residual_abs_max,
        "codebook_emb_min": emb_min,
        "codebook_emb_max": emb_max,
        "codebook_denom": emb_denom,
        "scale_factor": scale_factor,
        "trimmed_mean_per_pair_std": trimmed_mean_std,
    },
    "per_pair_stats_sorted_by_residual_std": sorted_by_std,
}

with open("FileStats.json", "w") as f:
    json.dump(output, f, indent=4, default=convert)

print("\nSaved per-pair stats + summary to FileStats.json")