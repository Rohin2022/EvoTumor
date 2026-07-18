"""
scale_sweep_delta_t_embedding.py

No training involved. Directly probes SinusoidalPosEmb(dim) at several
scale factors applied to your ACTUAL delta_t range (-4.68 to 4.68) to
check whether the embedding varies smoothly (adjacent delta_t -> adjacent
embedding) or aliases (adjacent delta_t -> uncorrelated embedding, i.e.
noise), AND whether it sufficiently discriminates between distinct delta_t
values (too coarse a scale is smooth but nearly constant end-to-end).

Two checks:
  1. Smoothness on REAL data spacing (450 actual training samples, sorted)
  2. Smoothness + discrimination on a DENSE synthetic grid spanning your
     true range, for the visual sweep plot
"""

import torch
import matplotlib.pyplot as plt
from ddpm import SinusoidalPosEmb  # adjust import path as needed
import pandas as pd

DIM = 64
SCALE_FACTORS = [1, 5, 10, 15, 20, 30, 50, 100, 300]

train_input = pd.read_csv("cross_eval/abdomen_atlas_pro/Datapairs_with_masks.csv")

# Real training delta_t values, sorted -- checks smoothness at the ACTUAL
# spacing the network will see during training, not an idealized grid.
real_delta_t_sorted, _ = torch.sort(
    torch.tensor(train_input['normalized_time_delta'].values, dtype=torch.float32))

real_std = real_delta_t_sorted.std().item()
real_min = real_delta_t_sorted.min().item()
real_max = real_delta_t_sorted.max().item()
print(f"Real delta_t stats: min={real_min:.4f}  max={real_max:.4f}  std={real_std:.4f}  n={len(real_delta_t_sorted)}")

# Dense synthetic grid spanning the true data range (-4.68 to 4.68), used
# for the visual sweep plot and the discrimination check.
delta_t_dense = torch.linspace(real_min, real_max, 1000)

pos_emb = SinusoidalPosEmb(DIM)

# ----------------------------------------------------------------------
# PART 1: smoothness on REAL data spacing
# ----------------------------------------------------------------------
print("\n=== PART 1: smoothness on REAL data spacing ===")
for scale in SCALE_FACTORS:
    with torch.no_grad():
        emb = pos_emb(real_delta_t_sorted * scale)
    cos_sim_adjacent = torch.nn.functional.cosine_similarity(emb[:-1], emb[1:], dim=1)
    print(f"scale={scale:>4}: mean adjacent cos-sim on REAL data spacing = "
          f"{cos_sim_adjacent.mean().item():.4f}")

# ----------------------------------------------------------------------
# PART 2: smoothness + discrimination on DENSE synthetic grid
# ----------------------------------------------------------------------
print("\n=== PART 2: smoothness + discrimination on dense grid ===")
mid_idx = len(delta_t_dense) // 2
one_std_idx_offset = max(1, int(len(delta_t_dense) * (real_std / (real_max - real_min))))

for scale in SCALE_FACTORS:
    with torch.no_grad():
        emb = pos_emb(delta_t_dense * scale)

    cos_sim_adjacent = torch.nn.functional.cosine_similarity(emb[:-1], emb[1:], dim=1)
    mean_smoothness = cos_sim_adjacent.mean().item()

    extreme_dist = (emb[0] - emb[-1]).norm().item()
    one_std_dist = (emb[mid_idx] - emb[mid_idx + one_std_idx_offset]).norm().item()

    print(f"scale={scale:>4}: smoothness={mean_smoothness:.4f}  "
          f"extreme_dist={extreme_dist:.4f}  one_std_dist={one_std_dist:.4f}  "
          f"one_std/extreme ratio={one_std_dist / max(extreme_dist, 1e-8):.4f}")

# ----------------------------------------------------------------------
# PART 3: visual sweep plot
# ----------------------------------------------------------------------
fig, axes = plt.subplots(len(SCALE_FACTORS), 1, figsize=(10, 3 * len(SCALE_FACTORS)))

for ax, scale in zip(axes, SCALE_FACTORS):
    with torch.no_grad():
        emb = pos_emb(delta_t_dense * scale)  # (1000, dim)

    cos_sim_adjacent = torch.nn.functional.cosine_similarity(emb[:-1], emb[1:], dim=1)
    mean_smoothness = cos_sim_adjacent.mean().item()

    ax.plot(delta_t_dense.numpy(), emb[:, 0].numpy(), label='dim 0')
    ax.plot(delta_t_dense.numpy(), emb[:, DIM // 2].numpy(), label=f'dim {DIM//2}')
    ax.set_title(f"scale={scale}  |  mean adjacent cos-sim={mean_smoothness:.4f} "
                 f"(near 1 = smooth, near 0/neg = aliased)")
    ax.legend()

plt.tight_layout()
plt.savefig("delta_t_scale_sweep.png")
print("\nSaved delta_t_scale_sweep.png")
print("Pick the scale that keeps smoothness high (>0.99) on the REAL data")
print("spacing (Part 1) while maximizing discrimination -- the one_std/extreme")
print("ratio in Part 2 (higher = more resolution per unit of delta_t range).")