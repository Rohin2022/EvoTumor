from monai.transforms import MapTransform
from monai.utils.enums import PostFix
from monai.data.image_reader import ImageReader
from monai.utils import GridSamplePadMode, ensure_tuple, ensure_tuple_rep
from monai.transforms.io.array import LoadImage, SaveImage
from monai.config.type_definitions import NdarrayOrTensor
from monai.utils.enums import TransformBackends
import os
from monai.transforms.transform import Transform, MapTransform
from monai.config import DtypeLike, KeysCollection
from monai.data import DataLoader, Dataset, list_data_collate, DistributedSampler, CacheDataset, MetaTensor
from torch.utils.data import Subset
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    SelectItemsd,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    SelectItemsd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
    CenterSpatialCropd,
    Resized,
    SpatialPadd,
    CastToTyped,
    apply_transform,
    RandZoomd,
    RandCropByLabelClassesd,
    RandSpatialCropd,
    GaussianSmooth,
)
from monai.data import PersistentDataset
import collections.abc
import math
import pickle
import shutil
import sys
import pandas as pd
import tempfile
import threading
import time
import warnings
from copy import copy, deepcopy
import h5py
import os

import numpy as np
import torch
import torch.nn.functional as F
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
        return d





# ---------------------------------------------------------------------------
# NEW: longitudinal-specific transforms
# ---------------------------------------------------------------------------
class LoadAndRegisterPaird(MapTransform):
    """
    Loads the fixed (ct0) and moving (ct1) images from disk via ITK, reads a
    precomputed ITK transform (written with `itk.transformwrite`, e.g. a
    uniGradICON DisplacementFieldTransform/CompositeTransform saved as
    "{ct0_bdmap}_{ct1_bdmap}.h5" under `transform_root`), resamples the
    moving image onto the fixed image's grid, and hands both back to MONAI
    as arrays + meta dicts (via ITKReader) so the rest of the pipeline
    (Orientationd, Spacingd, ScaleIntensityRanged, ToTensord, RandSpatialCropd)
    works unmodified.
 
    This has to be a combined load+warp step (rather than warping tensors
    after MONAI's own LoadImage) because ITK resampling needs real itk.Image
    objects with correct spacing/origin/direction, not bare arrays.
 
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
    in your codebase (e.g. wherever the precomputed transforms currently get
    applied for the original tumor-generation pipeline) before trusting this.
    I've left `invert_transform` as a flag so you can flip it in one place.
 
    Requires `ct0_bdmap` and `ct1_bdmap` to already be present in the data
    dict as plain strings (not tensors).
    """
 
    def __init__(
        self,
        fixed_key: str = "image_fixed",
        moving_key: str = "image_moving",
        # extra keys already living in the fixed (ct0) frame — loaded as-is,
        # no warp needed (e.g. organ_mask_fixed, tumor_mask_fixed)
        fixed_extra_keys: Sequence[str] = (),
        # extra keys living in the moving (ct1) frame — loaded then warped
        # into the fixed grid with nearest-neighbor interpolation, since
        # these are label maps and must stay binary/integer valued
        # (e.g. organ_mask_moving, tumor_mask_moving)
        moving_extra_keys: Sequence[str] = (),
        transform_root: str = None,
        id_key_1: str = "ct0_bdmap",
        id_key_2: str = "ct1_bdmap",
        organ_key: str = "organ",
        invert_transform: bool = False,
        allow_missing_keys: bool = False,
    ):
        all_keys = [fixed_key, moving_key] + list(fixed_extra_keys) + list(moving_extra_keys)
        super().__init__(keys=all_keys, allow_missing_keys=allow_missing_keys)
        self.fixed_key = fixed_key
        self.moving_key = moving_key
        self.fixed_extra_keys = list(fixed_extra_keys)
        self.moving_extra_keys = list(moving_extra_keys)
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
 
    def __call__(self, data):
        itk = self._itk
        d = dict(data)
 
        fixed_path = d[self.fixed_key]
        moving_path = d[self.moving_key]
        ct0_bdmap = d[self.id_key_1]
        ct1_bdmap = d[self.id_key_2]
        organ = d[self.organ_key]
 
        fixed_itk = itk.imread(fixed_path, itk.F)
        moving_itk = itk.imread(moving_path, itk.F)
        transform = self._load_transform(ct0_bdmap, ct1_bdmap, organ)
 
        # --- main image: linear interpolation ---
        linear_interp = itk.LinearInterpolateImageFunction.New(moving_itk)
        warped_itk = self._warp(
            moving_itk, fixed_itk, transform, linear_interp,
            default_pixel_value=-1000.0,  # air/HU floor; adjust if your images aren't raw HU at this stage
        )
 
        fixed_arr, fixed_meta = self._itk_reader.get_data(fixed_itk)
        warped_arr, warped_meta = self._itk_reader.get_data(warped_itk)



        fixed_arr = MetaTensor(fixed_arr[None], affine=fixed_meta["affine"])
        warped_arr = MetaTensor(warped_arr[None], affine=warped_meta["affine"])
 
        d[self.fixed_key] = fixed_arr
        d[f"{self.fixed_key}_meta_dict"] = fixed_meta
        d[self.moving_key] = warped_arr
        d[f"{self.moving_key}_meta_dict"] = warped_meta
 
        # --- fixed-frame extras (masks already aligned with ct0): load only, no warp ---
        for key in self.fixed_extra_keys:
            mask_path = d[key]
            mask_itk = itk.imread(mask_path, itk.UC)
            mask_arr, mask_meta = self._itk_reader.get_data(mask_itk)

            d[key] = MetaTensor(mask_arr[None], affine=mask_meta["affine"])
            d[f"{key}_meta_dict"] = mask_meta

        for key in self.moving_extra_keys:
            mask_path = d[key]
            mask_itk = itk.imread(mask_path, itk.UC)
            nn_interp = itk.NearestNeighborInterpolateImageFunction.New(mask_itk)
            warped_mask_itk = self._warp(
                mask_itk, fixed_itk, transform, nn_interp, default_pixel_value=0,
            )
            mask_arr, mask_meta = self._itk_reader.get_data(warped_mask_itk)
            d[key] = MetaTensor(mask_arr[None], affine=mask_meta["affine"])
            d[f"{key}_meta_dict"] = mask_meta
 
        del fixed_itk, moving_itk, warped_itk

        return d
 
 
class TumorMaskUnionDilated(MapTransform):
    """
    mask_union = maximum(tumor_mask_fixed, tumor_mask_moving)
    mask_union = gaussian_blur(mask_union)
 
    Both input masks are expected to already be in the same (fixed) grid —
    i.e. run this *after* LoadAndRegisterPaird has warped the moving-frame
    tumor mask into the fixed frame. The blur softens/dilates the binary
    union slightly rather than doing a hard morphological dilation; tune
    `sigma` (in voxels, per MONAI's GaussianSmooth) to taste.
    """
 
    def __init__(
        self,
        tumor_mask_fixed_key: str = "tumor_mask_fixed",
        tumor_mask_moving_key: str = "tumor_mask_moving",
        output_key: str = "tumor_mask_union",
        binary_union_mask_key: str = "binary_tumor_mask_union",
        sigma: float = 1.5,
        allow_missing_keys: bool = False,
    ):
        super().__init__(
            keys=[tumor_mask_fixed_key, tumor_mask_moving_key],
            allow_missing_keys=allow_missing_keys,
        )
        self.tumor_mask_fixed_key = tumor_mask_fixed_key
        self.tumor_mask_moving_key = tumor_mask_moving_key
        self.output_key = output_key
        self.binary_tumor_mask_union = binary_union_mask_key
        self.smoother = GaussianSmooth(sigma=sigma)
 
    def __call__(self, data):
        d = dict(data)
 
        mask0 = (d[self.tumor_mask_fixed_key] > 0.5).float()
        mask1 = (d[self.tumor_mask_moving_key] > 0.5).float()
 
        mask_union = torch.maximum(mask0, mask1)
        mask_union = self.smoother(mask_union)
 
        d[self.output_key] = mask_union
        d[self.binary_tumor_mask_union] = (mask_union > 0.5).float()
        return d
 
import datetime

def _log_cache_event(name, data):
    """Prints whenever a deterministic transform actually executes (i.e. cache miss)."""
    ct0 = data.get("ct0_bdmap", "?")
    ct1 = data.get("ct1_bdmap", "?")
    ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[CACHE-BUILD {ts}] Running '{name}' for pair ({ct0} -> {ct1}) "
          f"— this should print exactly once per pair per persistent cache.",
          flush=True)


class CacheRunLogger(Transform):
    """
    Wraps a deterministic transform and logs every time it is actually
    executed. Must subclass Transform (not just be a plain callable) or
    PersistentDataset._pre_transform will bail out on the first wrapped
    transform in the Compose list and cache nothing (see prior debugging).

    Use this to confirm: deterministic transforms run once per item when
    building/populating the persistent cache, and never again afterward
    (subsequent epochs should only exercise the stochastic crop stage).
    """
    def __init__(self, transform):
        self.transform = transform
        self.name = transform.__class__.__name__

    def __call__(self, data):
        if isinstance(data, list):
            for d in data:
                _log_cache_event(self.name, d)
            return [self.transform(d) for d in data]
        _log_cache_event(self.name, data)
        return self.transform(data)


class LongitudinalAugDataset(torch.utils.data.Dataset):
    def __init__(self, base, transform):
        self.base = base
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        return self.transform(self.base[idx])


ORGAN_MAPPING = {
    'spleen': 0,
    'bladder': 1,
    'gallbladder': 2,
    'esophagus': 3,
    'stomach': 4,
    'duodenum': 5,
    'colon': 6,
    'prostate': 7,
    'uterus': 8,
}


def _parse_organ_name(organ_name):
    if organ_name == "gallbladder":
        return "gall_bladder"
    return organ_name

IMAGE_KEYS = ["ct0", "ct1"]
WARPED_MASK_KEYS = ["organ_mask_fixed", "tumor_mask_fixed", "organ_mask_moving", "tumor_mask_moving"]
DERIVED_MASK_KEYS = ["binary_tumor_union_mask"]
MASK_KEYS = WARPED_MASK_KEYS + DERIVED_MASK_KEYS
 
# Keys that exist immediately after LoadAndRegisterPaird (before
# TumorMaskUnionDilated creates tumor_mask_union/binary_tumor_union_mask).
# Orientationd/Spacingd must only ever see these 6 — NOT the derived keys,
# which don't exist yet at that point in the pipeline.
LOADED_KEYS = IMAGE_KEYS + WARPED_MASK_KEYS  # 6 keys
 
# Full final key set, used for CropForegroundd/SpatialPadd/ToTensord/the
# stochastic crop, once everything (including derived masks) exists.
ALL_SPATIAL_KEYS = IMAGE_KEYS + MASK_KEYS + ["tumor_mask_union"]  # 8 keys
 
 
def get_longitudinal_loader(args):
    """
    Builds a train loader that returns paired (fixed, warped-moving) 128^3
    crops, plus per-timepoint organ/tumor masks (moving-frame masks warped
    into the fixed grid) and a blurred + binary tumor-mask union, for
    residual / longitudinal diffusion training.
 
    Expects args to include (in addition to the usual data_root_path etc.):
      - args.pair_csv_path: CSV with columns ct0_bdmap, ct1_bdmap, organ
        (+ any tabular radiomics columns you're conditioning on). `organ` is
        assumed the same for both timepoints of a pair, matching the
        single-timepoint pipeline's organ_mask/tumor_mask path convention.
      - args.segmentations_root_path / args.organ_segmentations_root_path:
        same layout as the original get_loader (.../<bdmap_id>/segmentations/*.nii.gz)
      - args.transform_root_path: directory of precomputed ITK transforms,
        one per pair (see LoadAndRegisterPaird docstring)
      - args.roi_x / roi_y / roi_z: crop size (128 in your case)
      - args.num_samples: number of crops per volume for RandCropByLabelClassesd
      - args.tumor_mask_union_sigma (optional, default 1.5): blur sigma
        for the dilated tumor union mask
    """
    deterministic_transforms = Compose(
        [
            CacheRunLogger(LoadAndRegisterPaird(
                fixed_key="ct0",
                moving_key="ct1",
                fixed_extra_keys=["organ_mask_fixed", "tumor_mask_fixed"],
                moving_extra_keys=["organ_mask_moving", "tumor_mask_moving"],
                transform_root=args.transform_root_path,
            )),
            Orientationd(keys=LOADED_KEYS, axcodes="RAS"),
            Spacingd(
                keys=LOADED_KEYS,
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "bilinear", "nearest", "nearest", "nearest", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=IMAGE_KEYS,
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            CastToTyped(keys=WARPED_MASK_KEYS, dtype=np.uint8),
            TumorMaskUnionDilated(
                tumor_mask_fixed_key="tumor_mask_fixed",
                tumor_mask_moving_key="tumor_mask_moving",
                output_key="tumor_mask_union",
                binary_union_mask_key="binary_tumor_union_mask",
                sigma=getattr(args, "tumor_mask_union_sigma", 1.5),
            ),
            CastToTyped(keys=DERIVED_MASK_KEYS, dtype=np.uint8),
            CropForegroundd(
                keys=ALL_SPATIAL_KEYS,
                source_key="binary_tumor_union_mask",
                select_fn=lambda x: x > 0.5,
                margin=64,
                allow_smaller=False,
            ),
            SpatialPadd(
                keys=ALL_SPATIAL_KEYS,
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                mode="constant",
            ),
            ToTensord(keys=ALL_SPATIAL_KEYS),
            SelectItemsd(keys=ALL_SPATIAL_KEYS + ["ct0_bdmap", "ct1_bdmap", "normalized_time_delta"]),
        ]
    )
 
    stochastic_transforms = Compose(
        [
            RandCropByLabelClassesd(
                keys=ALL_SPATIAL_KEYS,
                label_key="binary_tumor_union_mask",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                ratios=[1, 100_000],
                num_classes=2,
                num_samples=args.num_samples,
                image_key="ct0",
                image_threshold=-1,
            ),
        ]
    )


    pair_df = pd.read_csv(os.path.join(args.tumor_csv_path,args.dataset_list,args.pair_csv_path))
    pair_df = pair_df[pair_df["organ"].isin(list(ORGAN_MAPPING.keys()))]

    def build_image_path(bdmap_id):
        return os.path.join(args.data_root_path, str(bdmap_id), "ct.nii.gz")

    def build_organ_mask_path(row, bdmap_id_col):
        return os.path.join(
            args.organ_segmentations_root_path, str(row[bdmap_id_col]),
            "segmentations", f"{_parse_organ_name(row['organ'])}.nii.gz",
        )

    def build_tumor_mask_path(row, bdmap_id_col):
        return os.path.join(
            args.segmentations_root_path, str(row[bdmap_id_col]),
            "segmentations", f"{row['organ']}_lesion.nii.gz",
        )

    pair_df["ct0"] = pair_df["ct0_bdmap"].apply(build_image_path)
    pair_df["ct1"] = pair_df["ct1_bdmap"].apply(build_image_path)
    pair_df["organ_mask_fixed"] = pair_df.apply(lambda r: build_organ_mask_path(r, "ct0_bdmap"), axis=1)
    pair_df["organ_mask_moving"] = pair_df.apply(lambda r: build_organ_mask_path(r, "ct1_bdmap"), axis=1)
    pair_df["tumor_mask_fixed"] = pair_df.apply(lambda r: build_tumor_mask_path(r, "ct0_bdmap"), axis=1)
    pair_df["tumor_mask_moving"] = pair_df.apply(lambda r: build_tumor_mask_path(r, "ct1_bdmap"), axis=1)

    data_dicts = pair_df.to_dict("records")

    if args.persistent_cache:
        os.makedirs(args.persistent_cache_dir, exist_ok=True)
        cached_dataset = PersistentDataset(
            data=data_dicts,
            transform=deterministic_transforms,
            cache_dir=args.persistent_cache_dir,
        )
        train_dataset = LongitudinalAugDataset(cached_dataset, transform=stochastic_transforms)
    else:
        train_dataset = Dataset(
            data=data_dicts,
            transform=Compose([deterministic_transforms, stochastic_transforms]),
        )

    train_sampler = DistributedSampler(
        dataset=train_dataset, even_divisible=True, shuffle=True) if args.dist else None

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, collate_fn=list_data_collate,
        sampler=train_sampler, pin_memory=True, persistent_workers=False,
    )
    return train_loader, train_sampler, len(train_dataset)