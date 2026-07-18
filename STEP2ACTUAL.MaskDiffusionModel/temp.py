import os
import sys
sys.path.append(os.getcwd())

import hydra
from omegaconf import DictConfig
import torch
import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt

from dataset.dataloader import get_loader

@hydra.main(config_path='config', config_name='base_cfg', version_base=None)
def reconstruct(cfg: DictConfig):
    # 1. Initialize dataloader
    train_dataloader, _, _ = get_loader(cfg.dataset)
    
    # 2. Grab exactly one batch to inspect
    batch = next(iter(train_dataloader))
    
    # 3. Create an output directory
    out_dir = "debug_vis"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Extracted batch with keys: {list(batch.keys())}")
    
    # 4. Extract the first sample in the batch (Index 0)
    # Shape transitions from [B, C, X, Y, Z] -> [C, X, Y, Z]
    tumor_tsdf = batch["tumor_mask"][0].numpy()
    organ_tsdf = batch["organ_mask"][0].numpy()
    coords = batch["heatmap"][0].numpy()
    
    # 5. Squeeze the single channel dimension for the masks 
    # Nibabel expects pure spatial dimensions (X, Y, Z) for 3D volumes
    tumor_tsdf_3d = np.squeeze(tumor_tsdf, axis=0)
    organ_tsdf_3d = np.squeeze(organ_tsdf, axis=0)
    print(tumor_tsdf_3d.shape)
    print(organ_tsdf_3d.shape)
    print(f"HEATMAP: {coords.shape}")
if __name__ == '__main__':
    reconstruct()