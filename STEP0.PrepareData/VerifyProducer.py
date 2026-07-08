import os
import torch
import hydra
import nibabel as nib
import numpy as np
from omegaconf import DictConfig

from vq_gan_3d.model.vqgan import VQGAN

# =============================================================================
# 1. Decoding & Denormalization Helpers
# =============================================================================

def load_vqgan(vqgan_ckpt):
    """Loads the VQGAN model for inference."""
    vqgan = VQGAN.load_from_checkpoint(vqgan_ckpt, weights_only=False).cuda()
    vqgan.eval()
    return vqgan


def decode_latent(latent, vqgan):
    emb_min = vqgan.codebook.embeddings.min()
    emb_max = vqgan.codebook.embeddings.max()
    emb_denom = emb_max - emb_min

    latent_denorm = ((latent + 1.0) / 2.0) * emb_denom + emb_min

    with torch.no_grad():
        decoded = vqgan.decode(latent_denorm, quantize=False)  # ← was True

    decoded = decoded.permute(0, 1, 3, 4, 2).contiguous()
    return decoded


def denormalize_ct(ct_normalized, a_min, a_max, b_min=-1.0, b_max=1.0):
    """Inverts the MONAI intensity scaling back to original HU bounds."""
    return (ct_normalized - b_min) / (b_max - b_min) * (a_max - a_min) + a_min


# =============================================================================
# 2. Main Debug Loop
# =============================================================================

@hydra.main(config_path='config', config_name='base_cfg', version_base=None)
def debug_decode(cfg: DictConfig):
    # --- Setup Paths ---
    # Update this path to the specific batch file you want to test
    batch_path = "/scratch/rpinise1/EvoTumor/Residuals/batch_000002_18a69d50.pt" 

    if not os.path.exists(batch_path):
        raise FileNotFoundError(f"Could not find batch file: {batch_path}. Please update the path.")

    # Enforce saving to the current working directory, not the cache
    out_dir = os.getcwd()
    print(f"Output directory set to: {out_dir}")

    # --- Load Model & Data ---
    print("Loading VQGAN...")
    vqgan = load_vqgan(cfg.producer.vqgan_ckpt)

    print(f"Loading batch data from: {batch_path}")
    batch = torch.load(batch_path, map_location="cpu")

    # The producer casts these to float16 to save space; 
    # cast back to float32 before sending to the VQGAN.
    z0 = batch["z0"].cuda().float()
    residual = batch["residual"].cuda().float()

    # Reconstruct the ct1 latent from the residual
    z1 = z0 + residual

    # --- Decode ---
    print("Decoding z0 (ct0)...")
    ct0_decoded = decode_latent(z0, vqgan).cpu().numpy()

    print("Decoding z1 (ct1)...")
    ct1_decoded = decode_latent(z1, vqgan).cpu().numpy()

    # --- Denormalize & Save to NIfTI ---
    a_min = cfg.dataset.a_min
    a_max = cfg.dataset.a_max
    affine = np.diag([1.0, 1.0, 1.0, 1.0])

    batch_size = z0.shape[0]
    
    for b in range(batch_size):
        # 1. Map CT intensities back to Hounsfield Units
        ct0_hu = denormalize_ct(ct0_decoded[b, 0], a_min, a_max)
        ct1_hu = denormalize_ct(ct1_decoded[b, 0], a_min, a_max)

        ct0_id = batch["ct0_bdmap"][b]
        ct1_id = batch["ct1_bdmap"][b]

        # 2. Save Decoded CT Images
        ct0_path = os.path.join(out_dir, f"debug_b{b}_{ct0_id}_z0_ct.nii.gz")
        ct1_path = os.path.join(out_dir, f"debug_b{b}_{ct1_id}_z1_ct.nii.gz")

        nib.save(nib.Nifti1Image(ct0_hu.astype(np.float32), affine), ct0_path)
        nib.save(nib.Nifti1Image(ct1_hu.astype(np.float32), affine), ct1_path)

        # 3. Save Accompanying Masks (Useful for verifying registration/cropping alignment)
        mask0 = batch["tumor_ct0"][b, 0].numpy()
        mask1 = batch["tumor_ct1_warped"][b, 0].numpy()

        mask0_path = os.path.join(out_dir, f"debug_b{b}_{ct0_id}_tumor0_mask.nii.gz")
        mask1_path = os.path.join(out_dir, f"debug_b{b}_{ct1_id}_tumor1_mask.nii.gz")

        nib.save(nib.Nifti1Image(mask0.astype(np.uint8), affine), mask0_path)
        nib.save(nib.Nifti1Image(mask1.astype(np.uint8), affine), mask1_path)

        print(f"Saved decoded sample {b} (CTs and Masks) to current directory.")

    print("Debug decoding complete.")

if __name__ == "__main__":
    debug_decode()