from monai.data import MetaTensor
from scipy.ndimage import distance_transform_edt
from monai.transforms import MapTransform
from monai.utils.enums import PostFix
from monai.data.image_reader import ImageReader
from monai.utils import GridSamplePadMode, ensure_tuple, ensure_tuple_rep
from monai.transforms.io.array import LoadImage, SaveImage
from monai.config.type_definitions import NdarrayOrTensor
from monai.utils.enums import TransformBackends
from monai.transforms.transform import Transform, MapTransform
from monai.config import DtypeLike, KeysCollection
from monai.data import DataLoader, Dataset, list_data_collate, DistributedSampler, CacheDataset
from torch.utils.data import WeightedRandomSampler
import pandas as pd
from torch.utils.data import Subset
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
    CenterSpatialCropd,
    Resized,
    SpatialPadd,
    apply_transform,
    RandZoomd,
    RandCropByLabelClassesd,
)
from monai.data import PersistentDataset
import collections.abc
import math
import pickle
import shutil
import sys
import tempfile
import threading
import time
import warnings
from copy import copy, deepcopy
import h5py
import os


import numpy as np
import torch
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union


sys.path.append("..")


DEFAULT_POST_FIX = PostFix.meta()
class LoadAndRegisterPaird(MapTransform):
    """
    Loads tumor_mask and organ_mask for both the fixed (ct0) and moving (ct1)
    timepoints, reads a precomputed ITK transform (written with
    `itk.transformwrite`, e.g. a uniGradICON DisplacementFieldTransform/
    CompositeTransform saved as "{ct0_bdmap}_to_{ct1_bdmap}_{organ}.h5" under
    `transform_root`), and warps the moving (ct1) masks onto the fixed (ct0)
    grid with nearest-neighbor interpolation. Fixed-frame masks are loaded
    as-is, no warp needed.

    Output keys: tumor_mask_0, tumor_mask_1, organ_mask_0, organ_mask_1
    (all in the fixed/ct0 grid).

    *** DIRECTION CHECK NEEDED ***
    Based on your `register_pair` snippet:
        itk_wrapper.register_pair_with_mask(net, ct1_pre, ct0_pre, ...)
    called as (moving=ct1, fixed=ct0). Depending on how your fork of
    icon_registration/uniGradICON defines `register_pair_with_mask`'s return
    value, `phi` may be:
      (a) the transform ITK's ResampleImageFilter expects directly, i.e. a
          mapping from the fixed (reference/output) grid to the moving
          (input) grid — this is what's assumed below, or
      (b) the "forward" moving->fixed transform, in which case you need
          `phi.GetInverseTransform()` (or the inverse displacement field)
          before passing it to resample_image_filter.
    Get this wrong and the warp will be silently incorrect (not error out) —
    please confirm against how `register_pair_with_mask` is used elsewhere
    in your codebase before trusting this. `invert_transform` is left as a
    flag so you can flip it in one place.

    Requires `ct0_bdmap`, `ct1_bdmap`, and `organ` to already be present in
    the data dict as plain strings (not tensors).
    """

    def __init__(
        self,
        tumor_mask_fixed_key: str = "tumor_mask_fixed",
        tumor_mask_moving_key: str = "tumor_mask_moving",
        organ_mask_fixed_key: str = "organ_mask_fixed",
        organ_mask_moving_key: str = "organ_mask_moving",
        transform_root: str = None,
        id_key_1: str = "ct0_bdmap",
        id_key_2: str = "ct1_bdmap",
        organ_key: str = "organ",
        invert_transform: bool = False,
        allow_missing_keys: bool = False,
    ):
        all_keys = [
            tumor_mask_fixed_key,
            tumor_mask_moving_key,
            organ_mask_fixed_key,
            organ_mask_moving_key,
        ]
        super().__init__(keys=all_keys, allow_missing_keys=allow_missing_keys)
        self.tumor_mask_fixed_key = tumor_mask_fixed_key
        self.tumor_mask_moving_key = tumor_mask_moving_key
        self.organ_mask_fixed_key = organ_mask_fixed_key
        self.organ_mask_moving_key = organ_mask_moving_key
        self.transform_root = transform_root
        self.id_key_1 = id_key_1
        self.id_key_2 = id_key_2
        self.organ_key = organ_key
        self.invert_transform = invert_transform

        import itk as _itk
        from monai.data import ITKReader
        self._itk = _itk
        self._itk_reader = ITKReader()

    def _load_transform(self, ct0_bdmap, ct1_bdmap, organ):
        fname = f"{ct0_bdmap}_to_{ct1_bdmap}_{organ}.h5"
        fpath = os.path.join(self.transform_root, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"No precomputed transform found at {fpath}")
        transforms = self._itk.transformread(fpath)
        transform = transforms[0]
        if self.invert_transform:
            transform = transform.GetInverseTransform()
        return transform

    def _warp(self, moving_itk, fixed_itk, transform, interpolator, default_pixel_value):
        return self._itk.resample_image_filter(
            moving_itk,
            transform=transform,
            interpolator=interpolator,
            use_reference_image=True,
            reference_image=fixed_itk,
            default_pixel_value=default_pixel_value,
        )

    def _load_mask(self, itk, path):
        mask_itk = itk.imread(path, itk.UC)
        arr, meta = self._itk_reader.get_data(mask_itk)
        arr = np.squeeze(arr)  # drop any trailing/leading singleton component dim from ITK
        return mask_itk, MetaTensor(arr, affine=meta["affine"]), meta

    def __call__(self, data):
        itk = self._itk
        d = dict(data)

        ct0_bdmap = d[self.id_key_1]
        ct1_bdmap = d[self.id_key_2]
        organ = d[self.organ_key]

        transform = self._load_transform(ct0_bdmap, ct1_bdmap, organ)

        # --- fixed-frame masks (ct0): load only, no warp needed ---
        tumor_fixed_itk, tumor_0_tensor, tumor_0_meta = self._load_mask(
            itk, d[self.tumor_mask_fixed_key]
        )
        organ_fixed_itk, organ_0_tensor, organ_0_meta = self._load_mask(
            itk, d[self.organ_mask_fixed_key]
        )

        d["tumor_mask_0"] = tumor_0_tensor
        d["organ_mask_0"] = organ_0_tensor

        # --- moving-frame masks (ct1): load, then warp onto ct0 grid ---
        tumor_moving_itk = itk.imread(d[self.tumor_mask_moving_key], itk.UC)
        organ_moving_itk = itk.imread(d[self.organ_mask_moving_key], itk.UC)

        tumor_nn_interp = itk.NearestNeighborInterpolateImageFunction.New(tumor_moving_itk)
        warped_tumor_itk = self._warp(
            tumor_moving_itk, tumor_fixed_itk, transform, tumor_nn_interp,
            default_pixel_value=0,
        )
        organ_nn_interp = itk.NearestNeighborInterpolateImageFunction.New(organ_moving_itk)
        warped_organ_itk = self._warp(
            organ_moving_itk, organ_fixed_itk, transform, organ_nn_interp,
            default_pixel_value=0,
        )

        warped_tumor_arr, warped_tumor_meta = self._itk_reader.get_data(warped_tumor_itk)

        warped_organ_arr, warped_organ_meta = self._itk_reader.get_data(warped_organ_itk)

        d["tumor_mask_0"] = tumor_0_tensor
        d["organ_mask_0"] = organ_0_tensor
        d["tumor_mask_1"] = MetaTensor(warped_tumor_arr, affine=warped_tumor_meta["affine"])
        d["organ_mask_1"] = MetaTensor(warped_organ_arr, affine=warped_organ_meta["affine"])

        # NOTE: intentionally not stashing *_meta_dict entries in `d` — they contain
        # raw numpy fields (affine, spacing, etc.) that list_data_collate can choke
        # on during batching, and nothing downstream reads them. The affine is
        # still preserved on each MetaTensor itself if you need it later.

        for key in (
            self.tumor_mask_fixed_key,
            self.tumor_mask_moving_key,
            self.organ_mask_fixed_key,
            self.organ_mask_moving_key,
        ):
            d.pop(key, None)

        del (
            tumor_fixed_itk, organ_fixed_itk,
            tumor_moving_itk, organ_moving_itk,
            warped_tumor_itk, warped_organ_itk,
        )

        return d


class RandZoomd_select(RandZoomd):
    def __call__(self, data):
        d = dict(data)
        name = d['name']
        key = get_key(name)
        if (key not in ['10_03', '10_06', '10_07', '10_08', '10_09', '10_10']):
            return d
        d = super().__call__(d)
        return d


class RandCropByPosNegLabeld_select(RandCropByPosNegLabeld):
    def __call__(self, data):
        d = dict(data)
        name = d['name']
        key = get_key(name)
        # if key in ['10_03', '10_07', '10_08', '04']
        if key in ['10_03', '10_07', '10_08', '04', '05']:
            return d
        d = super().__call__(d)
        return d


class RandCropByLabelClassesd_select(RandCropByLabelClassesd):
    def __call__(self, data):
        d = dict(data)
        name = d['name']
        key = get_key(name)
        # print('key',key)
        # if key in ['10_03', '10_07', '10_08', '04']
        if key not in ['10_03', '10_07', '10_08', '04', '05']:
            return d
        d = super().__call__(d)
        return d


class Compose_Select(Compose):
    def __call__(self, input_):
        name = input_['name']
        key = get_key(name)
        for index, _transform in enumerate(self.transforms):
            # for RandCropByPosNegLabeld and RandCropByLabelClassesd case
            if (key in ['10_03', '10_07', '10_08', '04']) and (index == 8):
                continue
            elif (key not in ['10_03', '10_07', '10_08', '04']) and (index == 9):
                continue
            # for RandZoomd case
            if (key not in ['10_03', '10_06', '10_07', '10_08', '10_09', '10_10']) and (index == 7):
                continue
            input_ = apply_transform(
                _transform, input_, self.map_items, self.unpack_items, self.log_stats)
        return input_


class GenerateTumorHeatmapd(MapTransform):
    """
    Calculates the center of mass of a binary mask and generates a 3D 
    Gaussian heatmap centered on that point.
    """

    def __init__(self, ref_key="tumor_mask", out_key="heatmap", sigma=5.0, allow_missing_keys=False):
        super().__init__([ref_key], allow_missing_keys)
        self.ref_key = ref_key
        self.out_key = out_key
        self.sigma = sigma  # Controls how "wide" the target region is

    def __call__(self, data):
        d = dict(data)
        mask = d[self.ref_key]

        # Ensure it's a tensor for fast math
        mask_tensor = mask if isinstance(
            mask, torch.Tensor) else torch.tensor(mask)

        # Assuming shape is [Channel, X, Y, Z]
        binary_mask = (mask_tensor[0] > 0).float()
        indices = torch.nonzero(binary_mask)

        if len(indices) == 0:
            # Fallback if no tumor is present (blank heatmap)
            heatmap = torch.zeros_like(mask_tensor)
        else:
            # 1. Get exact X, Y, Z centroid
            centroid = indices.float().mean(dim=0)

            # 2. Generate 3D grid
            X, Y, Z = binary_mask.shape
            x_grid, y_grid, z_grid = torch.meshgrid(
                torch.arange(X, device=mask_tensor.device),
                torch.arange(Y, device=mask_tensor.device),
                torch.arange(Z, device=mask_tensor.device),
                indexing='ij'
            )

            # 3. Calculate Gaussian distance
            dist_sq = (x_grid - centroid[0])**2 + (y_grid -
                                                   centroid[1])**2 + (z_grid - centroid[2])**2
            heatmap = torch.exp(-dist_sq / (2 * self.sigma**2))

            # Add channel dimension back -> [1, X, Y, Z]
            heatmap = heatmap.unsqueeze(0)

        d[self.out_key] = heatmap
        return d



class ComputeTSDFd(MapTransform):
    """
    Computes the Truncated Signed Distance Function (TSDF) for binary masks.
    Inside the mask is negative, outside is positive, boundary is 0.
    Output is normalized between [-1, 1].
    """

    def __init__(self, keys, truncation_distance=5.0, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.truncation_distance = truncation_distance

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            mask = d[key]

            # Convert to numpy for fast EDT computation on the CPU DataLoader
            if isinstance(mask, torch.Tensor):
                mask_np = mask.detach().cpu().numpy()
            else:
                mask_np = mask

            # Initialize output tensor
            tsdf_out = np.zeros_like(mask_np, dtype=np.float32)

            # Process each channel independently (usually [C, H, W, D])
            for c in range(mask_np.shape[0]):
                binary_mask = mask_np[c] > 0.5

                # 1. Distance from outside to the boundary (0 inside the mask)
                outside_dist = distance_transform_edt(1 - binary_mask)

                # 2. Distance from inside to the boundary (0 outside the mask)
                inside_dist = distance_transform_edt(binary_mask)

                # 3. Create SDF (positive outside, negative inside)
                sdf = outside_dist - inside_dist

                # 4. Truncate at margins and normalize to [-1, 1] range
                tsdf = np.clip(sdf, -self.truncation_distance,
                               self.truncation_distance)
                tsdf = tsdf / self.truncation_distance

                tsdf_out[c] = tsdf

            # Return tensor in same device/format it arrived in
            d[key] = torch.from_numpy(tsdf_out) if isinstance(
                mask, torch.Tensor) else tsdf_out

        return d



def get_loader(args):
    MASK_KEYS = ["tumor_mask_0", "tumor_mask_1", "organ_mask_0", "organ_mask_1"]


    deterministic_transforms = [
        LoadAndRegisterPaird(
            tumor_mask_fixed_key="tumor_mask_fixed",
            tumor_mask_moving_key="tumor_mask_moving",
            organ_mask_fixed_key="organ_mask_fixed",
            organ_mask_moving_key="organ_mask_moving",
            transform_root=args.transform_root_path,
        ),
        EnsureChannelFirstd(keys=MASK_KEYS, channel_dim="no_channel"),

        # 1. Restructure the full volume first
        Orientationd(keys=MASK_KEYS, axcodes="RAS"),
        Spacingd(
            keys=MASK_KEYS,
            pixdim=(args.space_x, args.space_y, args.space_z),
            mode=("nearest", "nearest", "nearest", "nearest"),
        ),
        SpatialPadd(
            keys=MASK_KEYS,
            spatial_size=(args.roi_x, args.roi_y, args.roi_z),
            mode="constant",
        ),

        # 2. Heatmap, built only from tumor_mask_1, before cropping
        GenerateTumorHeatmapd(ref_key="tumor_mask_1", out_key="heatmap", sigma=8.0),
    ]

    # ----------------------------------------------------------------------
    # STOCHASTIC: anything randomized (crop) plus everything downstream of it
    # (TSDF must run on the post-crop patch, not the full volume, so it stays
    # here even though it's not random itself — caching it would freeze the
    # TSDF to whatever the first random crop happened to produce).
    # ----------------------------------------------------------------------
    stochastic_transforms = [
        # 3. Crop and pad (heatmap sliced identically to the masks)
        RandCropByLabelClassesd(
            keys=MASK_KEYS + ["heatmap"],
            label_key="tumor_mask_1",
            spatial_size=(args.roi_x, args.roi_y, args.roi_z),
            ratios=[1, 10000],
            num_classes=2,
            num_samples=args.num_samples,
        ),
        SpatialPadd(
            keys=MASK_KEYS + ["heatmap"],
            spatial_size=(args.roi_x, args.roi_y, args.roi_z),
            mode="constant",
        ),
        CenterSpatialCropd(
            keys=MASK_KEYS + ["heatmap"],
            roi_size=(args.roi_x, args.roi_y, args.roi_z),
        ),

        # 4. TSDF on all four cropped masks; heatmap stays a raw 0-to-1 Gaussian
        ComputeTSDFd(keys=MASK_KEYS),

        # 5. Finalize
        ToTensord(keys=MASK_KEYS + ["heatmap"]),
    ]
    # breakpoint()
    if args.phase == 'train':
        # training dict part

        train_input = pd.read_csv(os.path.join(
            args.data_txt_path, args.dataset_list, f'{args.datafile}'))

        train_input.dropna(inplace=True)

        def parseOrganName(organName):
            if organName == "gallbladder":
                return "gall_bladder"
            return organName

        # --- tumor mask paths (fixed = ct0, moving = ct1) ---
        train_input["tumor_mask_fixed"] = train_input.apply(
            lambda row: os.path.join(args.segmentations_root_path, str(
                row["ct0_bdmap"]), "segmentations", f"{row['organ']}_lesion.nii.gz"),
            axis=1
        )
        train_input["tumor_mask_moving"] = train_input.apply(
            lambda row: os.path.join(args.segmentations_root_path, str(
                row["ct1_bdmap"]), "segmentations", f"{row['organ']}_lesion.nii.gz"),
            axis=1
        )

        # --- organ mask paths (fixed = ct0, moving = ct1) ---
        train_input["organ_mask_fixed"] = train_input.apply(
            lambda row: os.path.join(args.organ_segmentations_root_path, str(
                row["ct0_bdmap"]), "segmentations", f"{parseOrganName(row['organ'])}.nii.gz"),
            axis=1
        )
        train_input["organ_mask_moving"] = train_input.apply(
            lambda row: os.path.join(args.organ_segmentations_root_path, str(
                row["ct1_bdmap"]), "segmentations", f"{parseOrganName(row['organ'])}.nii.gz"),
            axis=1
        )

        organ_mapping = {
            'spleen': 0,
            'bladder': 1,
            'gallbladder': 2,
            'esophagus': 3,
            'stomach': 4,
            'duodenum': 5,
            'colon': 6,
            'prostate': 7,
            'uterus': 8
        }

        # 1. Drop invalid rows first
        train_input = train_input[train_input["organ"].isin(list(organ_mapping.keys()))]

        # 2. Keep organ as a string for path-building/transform_root lookups
        #    (LoadAndRegisterPaird's organ_key expects the raw organ string, since
        #    the transform filename is "{ct0_bdmap}_to_{ct1_bdmap}_{organ}.h5")
        train_input["organ_name"] = train_input["organ"]

        train_input["organ_id"] = train_input["organ"].map(organ_mapping).astype(int)


        organ_counts = train_input['organ'].value_counts()
        train_input['sample_weight'] = train_input['organ'].apply(
            lambda o: 1.0 / np.sqrt(organ_counts[o])
        )

        # 4. Normalize numeric features if needed (normalized_time_delta is already
        #    normalized per your columns, so nothing else to standardize here unless
        #    you want to re-derive stats from unix_delta or similar)

        # 5. Select only the columns the pipeline actually needs, plus keep the
        #    path/id columns MONAI's transform will read
        keep_cols = [
            "ct0_bdmap", "ct1_bdmap",
            "normalized_time_delta",
            "organ_name",
            "tumor_mask_fixed", "tumor_mask_moving",
            "organ_mask_fixed", "organ_mask_moving", 
            "sample_weight",
            "organ_id"
        ]
        train_input = train_input[keep_cols]

        # 6. Rename to the keys LoadAndRegisterPaird / downstream code expect
        train_input = train_input.rename(columns={
            "organ_name": "organ",
            "normalized_time_delta": "delta_t",
        })

        # 7. Convert to dictionary records for MONAI
        data_dicts_train = train_input.to_dict("records")
        print('train len {}'.format(len(data_dicts_train)))

        persistent_cache_dir = os.path.join(args.persistent_cache_dir)
        os.makedirs(persistent_cache_dir, exist_ok=True)

        train_dataset = PersistentDataset(
            data=data_dicts_train,
            transform=Compose(deterministic_transforms),
            cache_dir=persistent_cache_dir,
        )

        # stochastic transforms applied on top, per-epoch, not cached
        train_dataset = Dataset(data=train_dataset, transform=Compose(stochastic_transforms))

        if args.dist:
            train_sampler = DistributedSampler(
                dataset=train_dataset, even_divisible=True, shuffle=True)
        else:
            sample_weights = [d["sample_weight"] for d in data_dicts_train]
            train_sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(train_dataset),
                replacement=True,
            )

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.num_workers, collate_fn=list_data_collate,
            sampler=train_sampler, pin_memory=True, persistent_workers=True,
        )
        return train_loader, train_sampler, len(train_dataset)



def get_key(name):
    # input: name
    # output: the corresponding key
    dataset_index = int(name[0:2])
    if dataset_index == 10:
        template_key = name[0:2] + '_' + name[17:19]
    else:
        template_key = name[0:2]
    return template_key


if __name__ == "__main__":
    train_loader, test_loader = partial_label_dataloader()
    for index, item in enumerate(test_loader):
        print(item['image'].shape, item['label'].shape, item['task_id'])
        input()
