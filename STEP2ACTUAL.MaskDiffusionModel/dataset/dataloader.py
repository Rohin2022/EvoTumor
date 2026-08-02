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
    DeleteItemsd,
    SelectItemsd,
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
class LoadPairedMasksd(MapTransform):
    """
    Loads tumor_mask and organ_mask for both timepoints (ct0 and ct1)
    with no registration/warping applied — each mask is loaded as-is,
    on its own native grid.

    Output keys: tumor_mask_0, tumor_mask_1, organ_mask_0, organ_mask_1
    """

    def __init__(
        self,
        tumor_mask_fixed_key: str = "tumor_mask_fixed",
        tumor_mask_moving_key: str = "tumor_mask_moving",
        organ_mask_fixed_key: str = "organ_mask_fixed",
        organ_mask_moving_key: str = "organ_mask_moving",
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

        import itk as _itk
        from monai.data import ITKReader
        self._itk = _itk
        self._itk_reader = ITKReader()

    def _load_mask(self, path):
        mask_itk = self._itk.imread(path, self._itk.UC)
        arr, meta = self._itk_reader.get_data(mask_itk)
        arr = np.squeeze(arr)  # drop any trailing/leading singleton component dim from ITK
        return MetaTensor(arr, affine=meta["affine"])

    def __call__(self, data):
        d = dict(data)

        # --- timepoint 0 (ct0): load only, no warp ---
        d["tumor_mask_0"] = self._load_mask(d[self.tumor_mask_fixed_key])
        d["organ_mask_0"] = self._load_mask(d[self.organ_mask_fixed_key])

        # --- timepoint 1 (ct1): load only, no warp ---
        d["tumor_mask_1"] = self._load_mask(d[self.tumor_mask_moving_key])
        d["organ_mask_1"] = self._load_mask(d[self.organ_mask_moving_key])

        for key in (
            self.tumor_mask_fixed_key,
            self.tumor_mask_moving_key,
            self.organ_mask_fixed_key,
            self.organ_mask_moving_key,
        ):
            d.pop(key, None)

        return d


from monai.transforms.io.array import LoadImage, SaveImage
from monai.data.image_reader import ImageReader


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
            print("EMPTY HEATMAP", data["ct1_bdmap"])
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
            d[key] = torch.from_numpy(tsdf_out)

        return d


class LogMissingClass1d(MapTransform):
    """
    Passthrough diagnostic transform. Checks whether `label_key` has any
    nonzero (class-1 / tumor) voxels, and if not, logs the record's
    ct0_bdmap/ct1_bdmap/organ/delta_t to a file. Does not modify the data
    in any way -- safe to drop in anywhere in the pipeline, including
    directly before RandCropByLabelClassesd, to see exactly which records
    are about to trigger MONAI's "no available indices of class 1 to crop"
    warning.
 
    Usage: insert into `deterministic_transforms` (to check once, on the
    cached/full volume) or into `stochastic_transforms` right before
    RandCropByLabelClassesd (to check every epoch, in case upstream
    randomness -- e.g. registration -- could vary presence between runs).
    """
 
    def __init__(self, label_key="tumor_mask_1", allow_missing_keys=False,
                 log_path="missing_class1_log.csv", tag=""):
        super().__init__([label_key], allow_missing_keys)
        self.label_key = label_key
        self.log_path = log_path
        self.tag = tag
        self._wrote_header = os.path.exists(log_path)
 
    def __call__(self, data):
        d = dict(data)
        mask = d[self.label_key]
        mask_tensor = mask if isinstance(mask, torch.Tensor) else torch.tensor(mask)
 
        n_class1 = (mask_tensor > 0).sum().item()
 
        if n_class1 == 0:
            ct0 = d.get("ct0_bdmap", "<unknown>")
            ct1 = d.get("ct1_bdmap", "<unknown>")
            organ = d.get("organ", "<unknown>")
            delta_t = d.get("delta_t", "<unknown>")
 
            print(f"[LogMissingClass1d{(':' + self.tag) if self.tag else ''}] "
                  f"NO CLASS-1 VOXELS  ct0={ct0}  ct1={ct1}  organ={organ}  delta_t={delta_t}")
 
            # append to a simple CSV log so you get a persistent record across
            # a full epoch / multiple runs without needing to scrape stdout
            write_header = not self._wrote_header
            with open(self.log_path, "a") as f:
                if write_header:
                    f.write("tag,ct0_bdmap,ct1_bdmap,organ,delta_t,n_class1_voxels\n")
                    self._wrote_header = True
                f.write(f"{self.tag},{ct0},{ct1},{organ},{delta_t},{n_class1}\n")
 
        # passthrough -- never modifies the data
        return d

class CreateUnionMaskd(MapTransform):
    """
    Computes the union of an organ mask and tumor mask for a given timepoint
    and stores it under `out_key`. Used purely as a scratch key to drive
    CropForegroundd's bounding-box computation (union of both structures),
    not intended to be carried downstream as model input.
    """
    def __init__(self, organ_key: str, tumor_key: str, out_key: str):
        super().__init__(keys=[organ_key, tumor_key])
        self.organ_key = organ_key
        self.tumor_key = tumor_key
        self.out_key = out_key

    def __call__(self, data):
        d = dict(data)
        organ = d[self.organ_key]
        tumor = d[self.tumor_key]

        if isinstance(organ, torch.Tensor):
            union = torch.logical_or(organ > 0, tumor > 0).to(organ.dtype)
        else:
            union = np.logical_or(organ > 0, tumor > 0).astype(organ.dtype)

        d[self.out_key] = union
        return d


_BDMAP_MAP_CACHE = {}  # csv_path -> {bdmap_id (str): patient_id (str)}
 
 
def _load_bdmap_mapping(args):
    csv_path = args.bdmap_mapping_csv
    if csv_path in _BDMAP_MAP_CACHE:
        return _BDMAP_MAP_CACHE[csv_path]
 
    bdmap_col = getattr(args, "bdmap_col", "bdmap_id")
    patient_col = getattr(args, "patient_col", "patient_id")
 
    df = pd.read_csv(csv_path, dtype=str)
    missing_cols = {bdmap_col, patient_col} - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"bdmap mapping CSV {csv_path} is missing expected column(s) "
            f"{missing_cols}. Found columns: {list(df.columns)}"
        )
 
    mapping = dict(zip(df[bdmap_col], df[patient_col]))
    _BDMAP_MAP_CACHE[csv_path] = mapping
    return mapping
 
 
def bdmap_to_patient(bdmap_id, args):
    """
    Returns the patient id for a given ct bdmap id, using the mapping CSV
    at args.bdmap_mapping_csv (cached in-process after first load).
 
    Raises KeyError if bdmap_id isn't found -- a missing id/patient mapping
    means downstream tumor/organ mask paths would also be wrong, so this
    fails loudly rather than silently returning None. If you'd rather skip
    unmapped rows instead of raising, filter train_input on this before
    calling, or catch KeyError at the call site.
    """
    mapping = _load_bdmap_mapping(args)
    bdmap_id = str(bdmap_id)
    if bdmap_id not in mapping:
        raise KeyError(
            f"bdmap_id '{bdmap_id}' not found in mapping CSV "
            f"{args.bdmap_mapping_csv}"
        )
    return mapping[bdmap_id]
 

import json
from .onc_condition_encoder import OncConditionEncoder
 
# ----------------------------------------------------------------------
# get_onc_conditioning
# ----------------------------------------------------------------------
 
_ONC_ENCODER_CACHE = {}  # onc_conditioning_dir -> OncConditionEncoder
 
 
def _get_onc_encoder(args):
    onc_dir = args.onc_conditioning_dir
    if onc_dir in _ONC_ENCODER_CACHE:
        return _ONC_ENCODER_CACHE[onc_dir]
 
    # Use the first .json file found (alphabetically, for determinism) as
    # the vocab reference. All conditioning JSONs in this folder are
    # expected to share identical schema/vocab since build_conditioning.py
    # derives it from the same fixed canonicalization folder for every
    # patient -- so which file is picked as reference doesn't matter.
    candidates = sorted(
        f for f in os.listdir(onc_dir)
        if f.endswith(".json") and not f.endswith(".log")
    )
    if not candidates:
        raise FileNotFoundError(
            f"No conditioning JSON files found in {onc_dir} to use as a "
            f"vocab reference for OncConditionEncoder."
        )
 
    ref_path = os.path.join(onc_dir, candidates[0])
    with open(ref_path) as f:
        reference_conditioning = json.load(f)
 
    encoder = OncConditionEncoder(reference_conditioning)
    _ONC_ENCODER_CACHE[onc_dir] = encoder
    print(f"[get_onc_conditioning] Built onc_cond encoder from reference "
          f"'{ref_path}'. {encoder.describe()}")
    return encoder



def get_onc_conditioning(patient_id, args):
    """
    Loads the conditioning JSON for `patient_id` from
    args.onc_conditioning_dir/{patient_id}.json and encodes it into a
    fixed-order onc_cond vector via OncConditionEncoder.
 
    Returns a numpy float32 array of shape (encoder.total_dim,) --
    numpy rather than a torch tensor so it round-trips cleanly through
    pandas .apply()/PersistentDataset caching (both pickle-based); it gets
    converted to a torch tensor downstream by MONAI's collate_fn like any
    other array-valued key.
 
    If the patient's conditioning JSON is missing, this WARNS and returns
    an all-zero vector of the correct shape (equivalent to "no known
    oncology conditioning for this patient") rather than raising, so a
    handful of patients missing RadGPT notes doesn't kill the whole
    data-prep run. Tighten this to raise instead if you'd rather catch
    missing conditioning data explicitly during data prep.
    """
    encoder = _get_onc_encoder(args)
 
    cond_path = os.path.join(args.onc_conditioning_dir, f"{patient_id}.json")
    if not os.path.exists(cond_path):
        warnings.warn(
            f"[get_onc_conditioning] No conditioning JSON for patient "
            f"'{patient_id}' at {cond_path} -- using all-zero onc_cond."
        )
        return np.zeros(encoder.total_dim, dtype=np.float32)
 
    with open(cond_path) as f:
        conditioning = json.load(f)
 
    return encoder.encode(conditioning).numpy().astype(np.float32)
 


from monai.transforms import MapTransform, RandomizableTransform
 
 
class ZeroDeltaTIdentityd(RandomizableTransform, MapTransform):
    def __init__(
        self,
        tumor_mask_0_key: str = "tumor_mask_0",
        tumor_mask_1_key: str = "tumor_mask_1",
        organ_mask_0_key: str = "organ_mask_0",
        organ_mask_1_key: str = "organ_mask_1",
        delta_t_key: str = "delta_t",
        prob: float = 0.1,
        flag_key: str = "is_zero_delta_t_synthetic",
        allow_missing_keys: bool = False,
    ):
        """
        Args:
            tumor_mask_0_key / organ_mask_0_key: source (ct0) keys to copy from.
            tumor_mask_1_key / organ_mask_1_key: target (ct1) keys to overwrite.
            delta_t_key: dict key holding the scalar delta_t value.
            prob: probability of hijacking each sample (args.zero_delta_t_prob).
            flag_key: dict key to record whether this sample was hijacked
                (bool), for downstream logging/analysis/loss-weighting.
                Harmless extra key -- ignored by transforms that don't
                reference it.
            allow_missing_keys: passed through to MapTransform.
        """
        MapTransform.__init__(
            self,
            keys=[tumor_mask_0_key, tumor_mask_1_key, organ_mask_0_key, organ_mask_1_key],
            allow_missing_keys=allow_missing_keys,
        )
        RandomizableTransform.__init__(self, prob=prob)
 
        self.tumor_mask_0_key = tumor_mask_0_key
        self.tumor_mask_1_key = tumor_mask_1_key
        self.organ_mask_0_key = organ_mask_0_key
        self.organ_mask_1_key = organ_mask_1_key
        self.delta_t_key = delta_t_key
        self.flag_key = flag_key
 
    def __call__(self, data):
        d = dict(data)
        self.randomize(None)
 
        d[self.flag_key] = bool(self._do_transform)
 
        if not self._do_transform:
            return d
 
        # --- hijack masks: t0 <- t1 (deep copy so later in-place transforms
        # on tumor_mask_1/organ_mask_1 don't also mutate tumor_mask_0/organ_mask_0) ---
        d[self.tumor_mask_0_key] = deepcopy(d[self.tumor_mask_1_key])
        d[self.organ_mask_0_key] = deepcopy(d[self.organ_mask_1_key])
 
        # --- override delta_t = 0 ---
        d[self.delta_t_key] = 0.0
 
        # onc_cond intentionally left untouched -- see module docstring.
 
        return d


class PrintNonTensorTypes:
    """Debug transform: prints types of any non-tensor items in the sample."""
    def __call__(self, data):
        if isinstance(data, dict):
            for k, v in data.items():
                if not torch.is_tensor(v):
                    print(f"[NON-TENSOR] key={k!r} type={type(v)} value={v!r}")
        elif isinstance(data, (list, tuple)):
            for i, v in enumerate(data):
                if not torch.is_tensor(v):
                    print(f"[NON-TENSOR] index={i} type={type(v)} value={v!r}")
        else:
            if not torch.is_tensor(data):
                print(f"[NON-TENSOR] type={type(data)} value={data!r}")
        return data

def get_loader(args):

    MASK_KEYS = ["tumor_mask_0", "tumor_mask_1", "organ_mask_0", "organ_mask_1"]

    if(args.include_img):
        deterministic_transforms = [
            LoadImageh5d(keys=["ct0"]),
            LoadPairedMasksd(
                tumor_mask_fixed_key="tumor_mask_fixed",
                tumor_mask_moving_key="tumor_mask_moving",
                organ_mask_fixed_key="organ_mask_fixed",
                organ_mask_moving_key="organ_mask_moving",
            ),
            EnsureChannelFirstd(keys=MASK_KEYS + ["ct0"], channel_dim="no_channel"),
            Orientationd(keys=MASK_KEYS + ["ct0"], axcodes="RAS"),
            Spacingd(
                keys=MASK_KEYS + ["ct0"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("nearest", "nearest", "nearest", "nearest","bilinear"),
            ),

            # Build per-timepoint union masks (organ ∪ tumor) purely to drive cropping
            CreateUnionMaskd(organ_key="organ_mask_0", tumor_key="tumor_mask_0", out_key="union_mask_0"),
            CreateUnionMaskd(organ_key="organ_mask_1", tumor_key="tumor_mask_1", out_key="union_mask_1"),

            # Crop each timepoint's mask set to the union bounding box, margin=40
            CropForegroundd(
                keys=["tumor_mask_0", "organ_mask_0", "union_mask_0","ct0"],
                source_key="union_mask_0",
                select_fn=lambda x: x > 0,
                allow_smaller=True,
                margin=40,
            ),
            CropForegroundd(
                keys=["tumor_mask_1", "organ_mask_1", "union_mask_1","ct0"],
                source_key="union_mask_1",
                select_fn=lambda x: x > 0,
                allow_smaller=True,
                margin=40,
            ),

            DeleteItemsd(keys=["union_mask_0", "union_mask_1",
                        "foreground_start_coord", "foreground_end_coord"]),

            SpatialPadd(
                keys=MASK_KEYS,
                spatial_size=(175, 175, 175),
                mode="constant",
            ),
            CenterSpatialCropd(roi_size=(175, 175, 175), keys=MASK_KEYS),



            LogMissingClass1d(label_key="tumor_mask_1", tag="post_pad_pre_crop",
                            log_path="missing_class1_log.csv"),
        ]
        


        # ----------------------------------------------------------------------
        # STOCHASTIC: anything randomized (crop) plus everything downstream of it
        # ----------------------------------------------------------------------
        stochastic_transforms = [
            # Random crop biased toward tumor, using tumor_mask_1 as the driving key
            ZeroDeltaTIdentityd(
                tumor_mask_0_key="tumor_mask_0",
                tumor_mask_1_key="tumor_mask_1",
                organ_mask_0_key="organ_mask_0",
                organ_mask_1_key="organ_mask_1",
                delta_t_key="delta_t",
                prob=args.zero_delta_t_prob,
            ),
            RandCropByLabelClassesd(
                keys=MASK_KEYS + ["ct0"],
                label_key="tumor_mask_1",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                ratios=[1, 10000],
                num_classes=2,
                num_samples=args.num_samples,
            ),

            GenerateTumorHeatmapd(ref_key="tumor_mask_1", out_key="heatmap", sigma=8.0),


            SpatialPadd(
                keys=MASK_KEYS + ["heatmap", "ct0"],
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                mode="constant",
            ),
            CenterSpatialCropd(
                keys=MASK_KEYS + ["heatmap", "ct0"],
                roi_size=(args.roi_x, args.roi_y, args.roi_z),
            ),

            ComputeTSDFd(keys=MASK_KEYS),

            ToTensord(keys=MASK_KEYS + ["heatmap", "delta_t", "organ_id","onc_cond", "ct0"]),
            SelectItemsd(keys=MASK_KEYS+["heatmap","onc_cond","delta_t", "organ_id", "ct0"]),

        ]
    else:
        deterministic_transforms = [
            LoadPairedMasksd(
                tumor_mask_fixed_key="tumor_mask_fixed",
                tumor_mask_moving_key="tumor_mask_moving",
                organ_mask_fixed_key="organ_mask_fixed",
                organ_mask_moving_key="organ_mask_moving",
            ),
            EnsureChannelFirstd(keys=MASK_KEYS, channel_dim="no_channel"),
            Orientationd(keys=MASK_KEYS, axcodes="RAS"),
            Spacingd(
                keys=MASK_KEYS,
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("nearest", "nearest", "nearest", "nearest"),
            ),

            # Build per-timepoint union masks (organ ∪ tumor) purely to drive cropping
            CreateUnionMaskd(organ_key="organ_mask_0", tumor_key="tumor_mask_0", out_key="union_mask_0"),
            CreateUnionMaskd(organ_key="organ_mask_1", tumor_key="tumor_mask_1", out_key="union_mask_1"),

            # Crop each timepoint's mask set to the union bounding box, margin=40
            CropForegroundd(
                keys=["tumor_mask_0", "organ_mask_0", "union_mask_0"],
                source_key="union_mask_0",
                select_fn=lambda x: x > 0,
                allow_smaller=True,
                margin=40,
            ),
            CropForegroundd(
                keys=["tumor_mask_1", "organ_mask_1", "union_mask_1"],
                source_key="union_mask_1",
                select_fn=lambda x: x > 0,
                allow_smaller=True,
                margin=40,
            ),

            DeleteItemsd(keys=["union_mask_0", "union_mask_1",
                        "foreground_start_coord", "foreground_end_coord"]),

            SpatialPadd(
                keys=MASK_KEYS,
                spatial_size=(175, 175, 175),
                mode="constant",
            ),
            CenterSpatialCropd(roi_size=(175, 175, 175), keys=MASK_KEYS),



            LogMissingClass1d(label_key="tumor_mask_1", tag="post_pad_pre_crop",
                            log_path="missing_class1_log.csv"),
        ]
        


        # ----------------------------------------------------------------------
        # STOCHASTIC: anything randomized (crop) plus everything downstream of it
        # ----------------------------------------------------------------------
        stochastic_transforms = [
            # Random crop biased toward tumor, using tumor_mask_1 as the driving key
            ZeroDeltaTIdentityd(
                tumor_mask_0_key="tumor_mask_0",
                tumor_mask_1_key="tumor_mask_1",
                organ_mask_0_key="organ_mask_0",
                organ_mask_1_key="organ_mask_1",
                delta_t_key="delta_t",
                prob=args.zero_delta_t_prob,
            ),
            RandCropByLabelClassesd(
                keys=MASK_KEYS,
                label_key="tumor_mask_1",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                ratios=[1, 10000],
                num_classes=2,
                num_samples=args.num_samples,
            ),

            GenerateTumorHeatmapd(ref_key="tumor_mask_1", out_key="heatmap", sigma=8.0),


            SpatialPadd(
                keys=MASK_KEYS + ["heatmap"],
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                mode="constant",
            ),
            CenterSpatialCropd(
                keys=MASK_KEYS + ["heatmap"],
                roi_size=(args.roi_x, args.roi_y, args.roi_z),
            ),

            ComputeTSDFd(keys=MASK_KEYS),

            ToTensord(keys=MASK_KEYS + ["heatmap", "delta_t", "organ_id","onc_cond"]),
            SelectItemsd(keys=MASK_KEYS+["heatmap","onc_cond","delta_t", "organ_id"]),

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

        train_input["patient_id"] = train_input.apply(
            lambda row: bdmap_to_patient(row["ct0_bdmap"], args), axis=1
        )

        train_input["onc_cond"] = train_input.apply(
            lambda row: get_onc_conditioning(row["patient_id"], args), axis=1
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

        train_input["ct0"] = train_input.apply(
            lambda row: os.path.join(args.image_root_path, str(
                row["ct0_bdmap"]), "ct.nii.gz"),
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
        organ_weight = train_input['organ'].apply(
            lambda o: 1.0 / np.sqrt(organ_counts[o])
        )

        # NOTE: still "normalized_time_delta" here — the rename to "delta_t"
        # doesn't happen until step 6, below.
        delta_t_clipped = train_input['normalized_time_delta'].clip(-3.0, 3.0)
        bin_width = 0.5
        train_input['delta_t_bin'] = (delta_t_clipped / bin_width).round() * bin_width

        delta_t_bin_counts = train_input['delta_t_bin'].value_counts()
        delta_t_weight = train_input['delta_t_bin'].apply(
            lambda b: 1.0 / np.sqrt(delta_t_bin_counts[b])
        )

        train_input['sample_weight'] = organ_weight * delta_t_weight
        train_input = train_input.drop(columns=['delta_t_bin'])

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
            "organ_id",
            "onc_cond",
            "ct0"
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
