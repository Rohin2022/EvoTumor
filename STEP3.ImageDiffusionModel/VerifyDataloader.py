"""
Loads one (or a few) batches from the longitudinal dataloader and saves every
image/mask tensor to disk as .nii.gz for visual inspection (e.g. in ITK-SNAP
or 3D Slicer).

Usage:
    python dump_dataloader_batch.py

Adjust `OUTPUT_DIR` and `NUM_BATCHES` below as needed. Uses the same Hydra
config composition as your VerifyDataloader.py script.
"""

import sys
import os

sys.path.append(os.getcwd())

import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf
import torch
import numpy as np
import nibabel as nib

from dataset.dataloader import get_longitudinal_loader, ALL_SPATIAL_KEYS

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
OUTPUT_DIR = "debug_nifti_dump"
NUM_BATCHES = 1  # how many batches to pull from the loader and dump

# Keys to dump. Defaults to every spatial key in the pipeline
# (ct0, ct1, organ_mask_fixed, tumor_mask_fixed, organ_mask_moving,
#  tumor_mask_moving, binary_tumor_union_mask, tumor_mask_union).
KEYS_TO_DUMP = ALL_SPATIAL_KEYS


def to_numpy_3d(tensor):
    """
    Squeezes a (C, H, W, D) or (C, B, H, W, D) tensor down to a plain 3D
    numpy array suitable for nib.Nifti1Image. Assumes a single channel.
    """
    arr = tensor
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr)
    # Squeeze out any singleton leading dims (channel, batch)
    while arr.ndim > 3:
        arr = arr[0]
    return arr


def get_affine(tensor, default_spacing=1.0):
    """
    Pulls the affine off a MetaTensor if present, otherwise falls back to a
    plain identity-with-spacing affine so the file still opens with correct
    orientation-agnostic geometry.
    """
    affine = getattr(tensor, "affine", None)
    if affine is not None:
        aff = affine
        if hasattr(aff, "detach"):
            aff = aff.detach().cpu().numpy()
        return np.asarray(aff)
    return np.eye(4) * default_spacing


def save_nii(tensor, path):
    arr = to_numpy_3d(tensor)
    affine = get_affine(tensor)
    nib.save(nib.Nifti1Image(arr.astype(np.float32), affine), path)
    print(f"  saved {path}  shape={arr.shape}")


with initialize(version_base=None, config_path="config"):
    cfg = compose(config_name="base_cfg")

print(OmegaConf.to_yaml(cfg))

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Building longitudinal dataloader...")
train_loader, _, dataset_size = get_longitudinal_loader(cfg.dataset)
print(f"Dataset has {dataset_size} pairs.")

with torch.no_grad():
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= NUM_BATCHES:
            break

        print(f"\nBatch {batch_idx} keys: {list(batch.keys())}")

        ct0_ids = batch.get("ct0_bdmap", ["unknown"] * cfg.dataset.batch_size)
        ct1_ids = batch.get("ct1_bdmap", ["unknown"] * cfg.dataset.batch_size)

        batch_size = batch[KEYS_TO_DUMP[0]].shape[0]
        for sample_idx in range(batch_size):
            ct0_id = ct0_ids[sample_idx] if not isinstance(ct0_ids, str) else ct0_ids
            ct1_id = ct1_ids[sample_idx] if not isinstance(ct1_ids, str) else ct1_ids
            sample_tag = f"batch{batch_idx}_sample{sample_idx}_{ct0_id}_to_{ct1_id}"
            sample_dir = os.path.join(OUTPUT_DIR, sample_tag)
            os.makedirs(sample_dir, exist_ok=True)

            print(f"Sample {sample_idx} ({ct0_id} -> {ct1_id}):")
            for key in KEYS_TO_DUMP:
                if key not in batch:
                    print(f"  [skip] key '{key}' not present in batch")
                    continue
                tensor = batch[key][sample_idx]
                out_path = os.path.join(sample_dir, f"{key}.nii.gz")
                save_nii(tensor, out_path)

print(f"\nDone. Files written under: {os.path.abspath(OUTPUT_DIR)}")