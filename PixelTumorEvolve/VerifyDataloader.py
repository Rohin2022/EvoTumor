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
        print(batch.keys())