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


class LoadImageh5d(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        reader: Optional[Union[ImageReader, str]] = None,
        dtype: DtypeLike = np.float32,
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = DEFAULT_POST_FIX,
        overwriting: bool = False,
        image_only: bool = False,
        ensure_channel_first: bool = False,
        simple_keys: bool = False,
        allow_missing_keys: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self._loader = LoadImage(
            reader, image_only, dtype, ensure_channel_first, simple_keys, *args, **kwargs)
        if not isinstance(meta_key_postfix, str):
            raise TypeError(
                f"meta_key_postfix must be a str but is {type(meta_key_postfix).__name__}.")
        self.meta_keys = ensure_tuple_rep(
            None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(
            meta_key_postfix, len(self.keys))
        self.overwriting = overwriting

    def register(self, reader: ImageReader):
        self._loader.register(reader)

    def __call__(self, data, reader: Optional[ImageReader] = None):
        d = dict(data)
        for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
            data = self._loader(d[key], reader)
            if self._loader.image_only:
                d[key] = data
            else:
                if not isinstance(data, (tuple, list)):
                    raise ValueError(
                        "loader must return a tuple or list (because image_only=False was used).")
                d[key] = data[0]
                if not isinstance(data[1], dict):
                    raise ValueError("metadata must be a dict.")
                meta_key = meta_key or f"{key}_{meta_key_postfix}"
                if meta_key in d and not self.overwriting:
                    raise KeyError(
                        f"Metadata with key {meta_key} already exists and overwriting=False.")
                d[meta_key] = data[1]
        # post_label_pth = d['post_label']
        # with h5py.File(post_label_pth, 'r') as hf:
        #     data = hf['post_label'][()]
        # d['post_label'] = data[0]
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
            print(f"CENTROID: {centroid}")

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


def get_loader(args):
    train_transforms = Compose(
        [
            # 1. Load data
            LoadImageh5d(keys=["tumor_mask", "organ_mask"]),
            EnsureChannelFirstd(keys=["tumor_mask", "organ_mask"]),

            # 2. Restructure the full volume FIRST
            Orientationd(keys=["tumor_mask", "organ_mask"], axcodes="RAS"),
            Spacingd(
                keys=["tumor_mask", "organ_mask"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("nearest", "nearest"),
            ),
            SpatialPadd(
                keys=["tumor_mask", "organ_mask"],
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                mode='constant'
            ),

            # 3. GENERATE THE HEATMAP BEFORE CROPPING
            # You can adjust sigma. 5.0 means the "hotspot" radius is roughly 10-15 voxels wide
            GenerateTumorHeatmapd(ref_key="tumor_mask",
                                  out_key="heatmap", sigma=8.0),

            # 4. Crop and Augment (Heatmap gets sliced exactly like the masks)
            RandCropByLabelClassesd(
                keys=["tumor_mask", "organ_mask", "heatmap"],  # Added heatmap
                label_key="tumor_mask",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                ratios=[1, 10000],
                num_classes=2,
                num_samples=args.num_samples,
            ),

            SpatialPadd(
                keys=["tumor_mask", "organ_mask", "heatmap"],
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                mode="constant",
            ),
            CenterSpatialCropd(
                keys=["tumor_mask", "organ_mask", "heatmap"],
                roi_size=(args.roi_x, args.roi_y, args.roi_z),
            ),

            RandRotate90d(
                keys=["tumor_mask", "organ_mask", "heatmap"],  # Added heatmap
                prob=0.20,
                max_k=3,
            ),

            # 5. Compute TSDF on the cropped mask patches
            # NOTE: Heatmap is NOT included here. We want it to stay a 0-to-1 Gaussian.
            ComputeTSDFd(keys=["tumor_mask", "organ_mask"]),



            # 6. Finalize
            ToTensord(keys=["tumor_mask", "organ_mask", "heatmap"]),
        ]
    )
    val_transforms = Compose(
        [
            LoadImageh5d(keys=["tumor_mask", "organ_mask"]),
            EnsureChannelFirstd(keys=["tumor_mask", "organ_mask"]),

            # 1. Restructure the full volume
            Orientationd(keys=["tumor_mask", "organ_mask"], axcodes="RAS"),
            Spacingd(
                keys=["tumor_mask", "organ_mask"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("nearest", "nearest"),
            ),

            # Generates the heatmap on the newly cropped, smaller volume.
            GenerateTumorHeatmapd(
                ref_key="tumor_mask",
                out_key="heatmap", 
                sigma=8.0
            ),

            # 2. CROP FOREGROUND FIRST 
            # Cut away the empty background before doing heavy math.
            # (Only the masks exist at this point)
            CropForegroundd(
                keys=["tumor_mask", "organ_mask", "heatmap"], 
                source_key="organ_mask"
            ),

            
            
            # 4. PAD TO ROI SIZE
            # Now that all three keys exist, pad them together to guarantee 
            # they are at least the size of your patch crop.
            SpatialPadd(
                keys=["tumor_mask", "organ_mask", "heatmap"],
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                mode="constant",
            ),
          
            # 5. EXTRACT THE PATCH
            RandCropByLabelClassesd(
                keys=["tumor_mask", "organ_mask", "heatmap"],
                label_key="tumor_mask",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                ratios=[1, 10000],      
                num_classes=2,
                num_samples=args.num_samples, 
            ),

            # 6. CALCULATE TSDF ON THE SMALL PATCH
            ComputeTSDFd(keys=["tumor_mask", "organ_mask"]),

            # 7. FINALIZE
            ToTensord(keys=["tumor_mask", "organ_mask", "heatmap"]),
        ]
    )
    # breakpoint()

    # breakpoint()
    if args.phase == 'train':
        # training dict part

        train_input = pd.read_csv(os.path.join(
            args.data_txt_path, args.dataset_list, f'{args.datafile}'))

        train_input["tumor_mask"] = train_input.apply(
            lambda row: os.path.join(args.segmentations_root_path, str(
                row["bdmap_id"]), "segmentations", f"{row['organ']}_lesion.nii.gz"),
            axis=1
        )

        def parseOrganName(organName):
            if (organName == "gallbladder"):
                return 'gall_bladder'
            return organName

        train_input["organ_mask"] = train_input.apply(
            lambda row: os.path.join(args.organ_segmentations_root_path, str(
                row["bdmap_id"]), "segmentations", f"{parseOrganName(row['organ'])}.nii.gz"),
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
        train_input = train_input[train_input["organ"].isin(
            list(organ_mapping.keys()))]
        train_input = train_input[train_input["volume_ml"] > 0.0]

        # 2. CALCULATE WEIGHTS FIRST (While volume_ml is still in true mL)
        vol_cutoff = float(train_input['volume_ml'].quantile(0.98))
        train_input['capped_volume'] = np.clip(
            train_input['volume_ml'], a_min=0, a_max=vol_cutoff)
        train_input['volume_bin'] = pd.cut(
            train_input['capped_volume'], bins=5, labels=False)

        organ_counts = train_input['organ'].value_counts()
        volume_counts = train_input['volume_bin'].value_counts()

        def compute_dual_weight(row):
            f_organ = organ_counts[row['organ']]
            f_vol = volume_counts[row['volume_bin']]
            return (1.0 / np.sqrt(f_organ)) * (1.0 / np.sqrt(f_vol))

        train_input['sample_weight'] = train_input.apply(
            compute_dual_weight, axis=1)

        # 3. NOW MAP ORGAN STRINGS TO INTEGERS
        train_input["organ"] = train_input.apply(
            lambda row: organ_mapping[row["organ"]], axis=1)

        # 4. NOW NORMALIZE NUMERIC FEATURES FOR THE MODEL
        import json
        from pandas.api.types import is_numeric_dtype

        stats_file = "dataset_norm_stats.json"
        exclude_cols = ['volume_bin', 'sample_weight', 'organ',
                        'tumor_mask', 'organ_mask', 'capped_volume', 'column_task']

        if os.path.exists(stats_file):
            print("Loading existing normalization statistics...")
            with open(stats_file, "r") as f:
                normalization_stats = json.load(f)
        else:
            print("Generating new normalization statistics...")
            normalization_stats = {}
            for key in train_input.columns:
                if key in exclude_cols or not is_numeric_dtype(train_input[key]):
                    continue

                normalization_stats[key] = {
                    "mean": float(train_input[key].mean()),
                    "std": float(train_input[key].std() + 1e-6)
                }

            with open(stats_file, "w") as f:
                json.dump(normalization_stats, f, indent=4)
            print("Saved new normalization data.")

        # Apply the normalization (using loaded or newly generated stats)
        for key, stats in normalization_stats.items():
            if key in train_input.columns:
                train_input[key] = (train_input[key] -
                                    stats["mean"]) / stats["std"]

        # 5. CONVERT TO DICTIONARY RECORDS FOR MONAI
        data_dicts_train = train_input.to_dict("records")
        print('train len {}'.format(len(data_dicts_train)))

        persistent_cache_dir = os.path.join(
            args.data_root_path, "monai_persistent_cache")
        os.makedirs(persistent_cache_dir, exist_ok=True)

        train_dataset = Dataset(
            data=data_dicts_train, transform=train_transforms)

        if args.dist:
            train_sampler = DistributedSampler(
                dataset=train_dataset, even_divisible=True, shuffle=True)
        else:
            # Extract weights in the exact order of the dataset
            sample_weights = [d["sample_weight"] for d in data_dicts_train]

            train_sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(train_dataset),
                replacement=True
            )
        # breakpoint()
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.num_workers,
                                  collate_fn=list_data_collate, sampler=train_sampler, pin_memory=True, persistent_workers=True)
        return train_loader, train_sampler, len(train_dataset)
        # return train_loader

    if args.phase == 'validation':
        # 1. Read the validation CSV
        val_input = pd.read_csv(os.path.join(
            args.data_txt_path, args.dataset_list, f'{args.datafile}'))

        # 2. Generate file paths
        def parseOrganName(organName):
            if (organName == "gallbladder"):
                return 'gall_bladder'
            return organName

        val_input["organ_mask"] = val_input.apply(
            lambda row: os.path.join(args.organ_segmentations_root_path, str(
                row["bdmap_id"]), "segmentations", f"{parseOrganName(row['organ'])}.nii.gz"),
            axis=1
        )

        val_input["tumor_mask"] = val_input.apply(
            lambda row: os.path.join(args.segmentations_root_path, str(
                row["bdmap_id"]), "segmentations", f"{row['organ']}_lesion.nii.gz"),
            axis=1
        )

        # 3. Map string labels to integers
        organ_mapping = {
            'spleen': 0, 'bladder': 1, 'gallbladder': 2, 'esophagus': 3,
            'stomach': 4, 'duodenum': 5, 'colon': 6, 'prostate': 7, 'uterus': 8
        }
        val_input = val_input[val_input["organ"].isin(
            list(organ_mapping.keys()))]
        val_input["organ"] = val_input.apply(
            lambda row: organ_mapping[row["organ"]], axis=1)

        # 4. Apply Normalization (Strictly LOADING stats, never generating them)
        import json
        stats_file = "dataset_norm_stats.json"

        if not os.path.exists(stats_file):
            raise FileNotFoundError(
                "Normalization stats missing. Run training phase first to generate dataset_norm_stats.json.")

        with open(stats_file, "r") as f:
            normalization_stats = json.load(f)

        for key, stats in normalization_stats.items():
            if key in val_input.columns:
                val_input[key] = (
                    val_input[key] - stats["mean"]) / stats["std"]

        # 5. Convert to MONAI dictionary format
        data_dicts_val = val_input.to_dict("records")
        print('val len {}'.format(len(data_dicts_val)))

        # 6. Build the Dataset and Loader (No Weighted Sampler needed)
        if args.cache_dataset:
            val_dataset = CacheDataset(
                data=data_dicts_val, transform=val_transforms, cache_rate=args.cache_rate)
        else:
            val_dataset = Dataset(data=data_dicts_val,
                                  transform=val_transforms)

        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=list_data_collate)

        return val_loader, val_transforms, len(val_dataset)


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
