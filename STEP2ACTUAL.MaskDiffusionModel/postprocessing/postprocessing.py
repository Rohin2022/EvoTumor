import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from monai.transforms import FillHoles, KeepLargestConnectedComponent, Compose

def postprocess_tensor(raw_mask, scale_factor=3, threshold=0.5, num_components=1):
    """
    Handles both (X, Y, Z) and (B, X, Y, Z) formats with NO channel dimension.
    Accepts both PyTorch Tensors and NumPy arrays.
    """
    if isinstance(raw_mask, np.ndarray):
        tensor_mask = torch.from_numpy(raw_mask).float()
    else:
        tensor_mask = raw_mask.float()

    original_dims = tensor_mask.dim()
    if original_dims == 3:
        tensor_mask = tensor_mask.unsqueeze(0)
    elif original_dims == 4:
        pass
    else:
        raise ValueError(f"Expected 3D (X,Y,Z) or 4D (B,X,Y,Z) input, got {original_dims}D")

    tensor_mask = tensor_mask.unsqueeze(1)

    if scale_factor != 1:
        tensor_mask = F.interpolate(
            tensor_mask, 
            scale_factor=scale_factor, 
            mode='trilinear', 
            align_corners=False
        )

    binary_mask = (tensor_mask < threshold).to(torch.uint8)

    postprocess_transforms = Compose([
        FillHoles(),
        KeepLargestConnectedComponent(num_components=num_components)
    ])

    processed_batch = []
    for i in range(binary_mask.shape[0]):
        single_item = binary_mask[i]
        cleaned_item = postprocess_transforms(single_item)
        processed_batch.append(cleaned_item)

    final_tensor = torch.stack(processed_batch, dim=0)

    final_tensor = final_tensor.squeeze(1)

    if original_dims == 3:
        final_tensor = final_tensor.squeeze(0)

    # Return in the same format it was received
    if isinstance(raw_mask, np.ndarray):
        return final_tensor.cpu().numpy().astype(np.uint8)
    return final_tensor


def load_nifti_and_postprocess(raw_nifti_path, output_path, scale_factor=3, threshold=0.5, num_components=1):
    """Wrapper function to handle NIfTI file I/O safely."""
    img = nib.load(raw_nifti_path)
    raw_data = img.get_fdata() # Typically (X, Y, Z)
    affine = img.affine

    # Call our dynamic function
    cleaned_mask_np = postprocess_tensor(raw_data, scale_factor=scale_factor, num_components=num_components, threshold=threshold)

    # Adjust the Affine Matrix (Divide physical spacing by scale factor)
    new_affine = affine.copy()
    new_affine[:3, :3] /= scale_factor

    # Save back to NIfTI
    cleaned_nifti = nib.Nifti1Image(cleaned_mask_np, new_affine)
    nib.save(cleaned_nifti, output_path)
    
    print(f"Original shape: {raw_data.shape} -> New shape: {cleaned_mask_np.shape}")

if __name__ == "__main__":
    input_file = "/projects/bodymaps/Rohin/TumorSynthesis/STEP2.MaskDiffusionModel/inference_masks/step_inference_sample_0_RAW.nii.gz"
    output_file = "/projects/bodymaps/Rohin/TumorSynthesis/STEP2.MaskDiffusionModel/inference_masks/step_inference_sample_0_CLEAN.nii.gz"
    
    if Path(input_file).exists():
        load_nifti_and_postprocess(
            raw_nifti_path=input_file,
            output_path=output_file,
            num_components=1,  # Keeps ONLY the single largest tumor blob
            threshold=0.5,
            scale_factor=3      # Converts > 50% probability into solid tumor
        )
    else:
        print(f"File not found: {input_file}")