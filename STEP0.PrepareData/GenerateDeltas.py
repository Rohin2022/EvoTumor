"""
unigradicon_coregister.py

Co-register two CT volumes using uniGradICON's official segmentation-aware
registration paths, save the resulting transform to disk, and separately
apply that saved transform to an arbitrary list of volumes (CT images and/or
masks).

This wraps the *actual* unigradicon library functions (the same ones used
internally by the `unigradicon-register` / `unigradicon-warp` CLIs), so
behavior matches the official docs exactly:

  - mask_mode="background"      -> mirrors `--fixed_segmentation` /
                                    `--moving_segmentation` WITHOUT
                                    `--loss_function_masking`
                                    (segmentations multiply out the
                                    background before registration)

  - mask_mode="loss"            -> mirrors `--fixed_segmentation` /
                                    `--moving_segmentation` WITH
                                    `--loss_function_masking`
                                    (segmentations are passed into the
                                    network's similarity loss as masks,
                                    typically combined with IO finetuning;
                                    requires io_iterations to be set, since
                                    with zero IO steps the mask is never
                                    used for anything)

Requires:
    pip install unigradicon icon_registration itk torch
"""

import itk
import icon_registration.itk_wrapper as itk_wrapper
from unigradicon import get_model_from_model_zoo, make_sim, preprocess, maybe_cast


def coregister_ct_with_masks(
    ct1_path: str,
    ct2_path: str,
    mask1_path: str,
    mask2_path: str,
    output_transform_path: str,
    modality: str = "ct",
    mask_mode: str = "background",   # "background" or "loss"
    io_iterations=50,                # int, or None to disable IO finetuning
    io_sim: str = "lncc",            # "lncc", "lncc2", or "mind"
    model_name: str = "unigradicon", # "unigradicon" or "multigradicon"
):
    """
    Co-register ct2 (moving) onto ct1 (fixed) using uniGradICON, using the
    official segmentation-guided registration features, and save the
    resulting forward transform (phi_AB) to disk.

    This function does NOT warp any images itself -- use
    `apply_transform_to_volumes` afterward to warp the moving CT, the moving
    mask, or any other volumes that share the moving image's grid.

    Parameters
    ----------
    ct1_path : str
        Path to the fixed/reference CT image.
    ct2_path : str
        Path to the moving CT image (to be warped onto ct1's space).
    mask1_path : str
        Path to the fixed image's segmentation mask.
    mask2_path : str
        Path to the moving image's segmentation mask.
    output_transform_path : str
        Path to write the registration transform (e.g. "trans.hdf5").
    modality : str
        "ct" or "mri". Controls uniGradICON's intensity preprocessing/clamping.
    mask_mode : str
        "background" -> masks out background using segmentations BEFORE
                         registration (equivalent to the CLI without
                         --loss_function_masking). Both masks are required.
        "loss"        -> applies loss function masking using the
                         segmentations DURING IO (equivalent to the CLI
                         with --loss_function_masking). Both masks are
                         required, and io_iterations must not be None --
                         with zero IO steps the masks would never actually
                         be used by the network.
    io_iterations : int or None
        Number of instance-optimization (IO) iterations. None disables IO
        and uses pure network inference.
    io_sim : str
        Similarity measure baked into the network: "lncc", "lncc2", or
        "mind". Matches the --io_sim CLI flag. Note this affects the network
        regardless of mask_mode, since it's set at model-build time -- it's
        not exclusive to mask_mode="loss".
    model_name : str
        "unigradicon" or "multigradicon".

    Returns
    -------
    dict with keys 'phi_AB' (forward transform, used to resample moving
    images into fixed space) and 'phi_BA' (inverse transform).
    """

    if mask_mode not in ("background", "loss"):
        raise ValueError('mask_mode must be "background" or "loss"')

    if mask_mode == "loss" and io_iterations is None:
        raise ValueError(
            'mask_mode="loss" requires io_iterations to be set (e.g. 50); '
            "loss-function masking only takes effect during IO finetuning, "
            "so with io_iterations=None the masks would never be used."
        )

    # --- Build the model exactly like the CLI does ---
    net = get_model_from_model_zoo(model_name, make_sim(io_sim))

    # --- Load images and segmentations ---
    fixed = itk.imread(ct1_path)
    moving = itk.imread(ct2_path)
    fixed_segmentation = itk.imread(mask1_path)
    moving_segmentation = itk.imread(mask2_path)

    if mask_mode == "loss":
        # Mirrors: --fixed_segmentation --moving_segmentation --loss_function_masking
        # Segmentations are NOT multiplied into the images; they are passed
        # as mask_A/mask_B into the network's similarity loss term.
        phi_AB, phi_BA = itk_wrapper.register_pair_with_mask(
            net,
            preprocess(moving, modality),
            preprocess(fixed, modality),
            moving_segmentation,
            fixed_segmentation,
            finetune_steps=None,
        )
    else:
        # Mirrors: --fixed_segmentation --moving_segmentation
        # (without --loss_function_masking)
        # preprocess() multiplies the segmentation into the image, masking
        # out the background BEFORE registration.
        phi_AB, phi_BA = itk_wrapper.register_pair(
            net,
            preprocess(moving, modality, moving_segmentation),
            preprocess(fixed, modality, fixed_segmentation),
            finetune_steps=io_iterations,
        )

    # --- Save the transform to disk (mirrors itk.transformwrite in the CLI) ---
    #itk.transformwrite([phi_AB], output_transform_path)

    return {"phi_AB": phi_AB, "phi_BA": phi_BA}


def apply_transform_to_volumes(
    reference_path: str,
    transform_path: str,
    moving_paths: list[str],
    output_paths: list[str],
    interpolations: list[str] | str = "linear",
):
    """
    Apply a saved uniGradICON transform to a list of moving volumes (CT
    images and/or masks), warping each into the reference (fixed) image's
    space. Mirrors the official `unigradicon-warp` CLI, generalized to an
    arbitrary list of inputs in one call so the network/transform don't need
    to be reloaded per volume.

    Parameters
    ----------
    reference_path : str
        Path to the fixed/reference image whose grid defines the output
        space (same image used as ct1_path / --fixed during registration).
    transform_path : str
        Path to the transform saved by `coregister_ct_with_masks`
        (or any itk.transformwrite-compatible file, e.g. trans.hdf5).
    moving_paths : list[str]
        Paths to the volumes to warp. Order must correspond to output_paths
        and interpolations (if interpolations is a list).
    output_paths : list[str]
        Paths to write each warped volume to. Same length/order as
        moving_paths.
    interpolations : list[str] or str
        Either a single string applied to every volume, or a per-volume
        list the same length as moving_paths. Each entry must be "linear"
        (for CT/MRI intensity images) or "nearest" (for label maps/masks,
        to avoid introducing fractional/interpolated label values).

    Returns
    -------
    list[str]
        The output_paths, for convenience/chaining.
    """

    if len(moving_paths) != len(output_paths):
        raise ValueError("moving_paths and output_paths must be the same length")

    if isinstance(interpolations, str):
        interpolations = [interpolations] * len(moving_paths)
    elif len(interpolations) != len(moving_paths):
        raise ValueError(
            "interpolations must be a single string or a list the same "
            "length as moving_paths"
        )

    for interp in interpolations:
        if interp not in ("linear", "nearest"):
            raise ValueError('each interpolation must be "linear" or "nearest"')

    fixed = itk.imread(reference_path)
    phi_AB = itk.transformread(transform_path)[0]

    for moving_path, output_path, interp in zip(
        moving_paths, output_paths, interpolations
    ):
        moving = itk.imread(moving_path)
        moving_for_warp, maybe_cast_back = maybe_cast(moving)

        if interp == "linear":
            interpolator = itk.LinearInterpolateImageFunction.New(moving_for_warp)
        else:
            interpolator = itk.NearestNeighborInterpolateImageFunction.New(
                moving_for_warp
            )

        warped = itk.resample_image_filter(
            moving_for_warp,
            transform=phi_AB,
            use_reference_image=True,
            reference_image=fixed,
            interpolator=interpolator,
        )
        warped = maybe_cast_back(warped)
        itk.imwrite(warped, output_path)

    return output_paths


if __name__ == "__main__":
    # Example: option 1 from the docs - mask out background before registration
    ct1 = "/projects/bodymaps/Data/image_only/AbdomenAtlasPro/AbdomenAtlasPro/BDMAP_00040739/ct.nii.gz"
    ct2 = "/projects/bodymaps/Data/image_only/AbdomenAtlasPro/AbdomenAtlasPro/BDMAP_00154068/ct.nii.gz"
    mask1 = "/projects/bodymaps/Data/mask_only/AbdomenAtlasPro/AbdomenAtlasPro/BDMAP_00040739/segmentations/liver.nii.gz"
    mask2 = "/projects/bodymaps/Data/mask_only/AbdomenAtlasPro/AbdomenAtlasPro/BDMAP_00154068/segmentations/liver.nii.gz"

    coregister_ct_with_masks(
        ct1_path=ct1,
        ct2_path=ct2,
        mask1_path=mask1,
        mask2_path=mask2,
        output_transform_path="trans.hdf5",
        mask_mode="background",
        io_iterations=None,   # matches `--io_iterations None` in the docs example
    )

    # Warp the moving CT (linear) and the moving mask (nearest-neighbor) in one call
    apply_transform_to_volumes(
        reference_path=ct1,
        transform_path="trans.hdf5",
        moving_paths=[ct2, mask2],
        output_paths=["ct2_registered.nii.gz", "mask2_registered.nii.gz"],
        interpolations=["linear", "nearest"],
    )