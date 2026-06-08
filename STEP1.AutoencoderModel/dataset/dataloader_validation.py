from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandSpatialCropd,
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
import sys
from copy import copy, deepcopy
import h5py, os
import numpy as np
import torch
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

sys.path.append("..") 

from torch.utils.data import Subset

from monai.data import DataLoader, Dataset, list_data_collate, DistributedSampler, CacheDataset
from monai.config import DtypeLike, KeysCollection
from monai.transforms.transform import Transform, MapTransform
from monai.utils.enums import TransformBackends
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.io.array import LoadImage, SaveImage
from monai.utils import GridSamplePadMode, ensure_tuple, ensure_tuple_rep
from monai.data.image_reader import ImageReader
from monai.utils.enums import PostFix
DEFAULT_POST_FIX = PostFix.meta()


class LoadImaged_BodyMap(MapTransform):
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
        self._loader = LoadImage(reader, image_only, dtype, ensure_channel_first, simple_keys, *args, **kwargs)
        if not isinstance(meta_key_postfix, str):
            raise TypeError(f"meta_key_postfix must be a str but is {type(meta_key_postfix).__name__}.")
        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.overwriting = overwriting


    def register(self, reader: ImageReader):
        self._loader.register(reader)


    def __call__(self, data, reader: Optional[ImageReader] = None):
        d = dict(data)
        # print(d['image'])
        for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
            try:
                data = self._loader(d[key], reader)
            except Exception as e:
                print("=" * 80)
                print("FAILED CASE:", d.get("name"))
                print("IMAGE PATH:", d.get(key))
                print("ERROR:", repr(e))
                print("=" * 80)
                raise

            if self._loader.image_only:
                d[key] = data
            else:
                if not isinstance(data, (tuple, list)):
                    raise ValueError("loader must return a tuple or list (because image_only=False was used).")
                d[key] = data[0]
                if not isinstance(data[1], dict):
                    raise ValueError("metadata must be a dict.")
                meta_key = meta_key or f"{key}_{meta_key_postfix}"
                if meta_key in d and not self.overwriting:
                    raise KeyError(f"Metadata with key {meta_key} already exists and overwriting=False.")
                d[meta_key] = data[1]

        organ_lbl, meta_information = self.label_transfer(d['label'], d['image'].shape)
        
        if hasattr(d['image'], "meta"):
            from monai.data.meta_tensor import MetaTensor
            
            # If meta_information exists and is a dictionary/MetaTensor meta, use it.
            # Otherwise, safely fallback to the image's metadata.
            chosen_meta = deepcopy(meta_information) if meta_information is not None else deepcopy(d['image'].meta)

            d['label'] = MetaTensor(
                organ_lbl, 
                meta=chosen_meta
            )
        else:
            d['label'] = organ_lbl
            
        d['label_meta_dict'] = meta_information
        return d

    def label_transfer(self, lbl_dir, shape):
        organ_lbl = np.zeros(shape)

        organ_mapping = {
            # Healthy Organs
            'spleen': 1,
            'bladder': 2,
            'gall_bladder': 3,
            'esophagus': 4,
            'stomach': 5,
            'duodenum': 6,
            'colon': 7,
            'prostate': 8,
            'uterus': 9,
            
            # Lesions
            'spleen_lesion': 10,
            'bladder_lesion': 11,
            'gallbladder_lesion': 12,
            'esophagus_lesion': 13,
            'stomach_lesion': 14,
            'duodenum_lesion': 15,
            'colon_lesion': 16,
            'prostate_lesion': 17,
            'uterus_lesion': 18
        }
        
        meta_information = None # Initialize in case no files are found in the directory

        for organ_name, label_idx in organ_mapping.items():
            file_path = os.path.join(lbl_dir, f"{organ_name}.nii.gz")
            
            if os.path.exists(file_path):
                try:
                    array, meta_information = self._loader(file_path)
                except Exception as e:
                    print(f"Failed organ file: {file_path}")
                    raise e
                organ_lbl[array > 0] = label_idx


        return organ_lbl, meta_information



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
        self._loader = LoadImage(reader, image_only, dtype, ensure_channel_first, simple_keys, *args, **kwargs)
        if not isinstance(meta_key_postfix, str):
            raise TypeError(f"meta_key_postfix must be a str but is {type(meta_key_postfix).__name__}.")
        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.overwriting = overwriting


    def register(self, reader: ImageReader):
        self._loader.register(reader)


    def __call__(self, data, reader: Optional[ImageReader] = None):
        d = dict(data)
        for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
            try:
                loaded = self._loader(d[key], reader)
            except Exception as e:
                print(f"Failed to load {d.get('name', d[key])}: {e}")
                raise  # re-raise so DataLoader skips this sample properly

            if self._loader.image_only:
                d[key] = loaded
            else:
                if not isinstance(loaded, (tuple, list)):
                    raise ValueError("loader must return a tuple or list (because image_only=False was used).")
                d[key] = loaded[0]
                if not isinstance(loaded[1], dict):
                    raise ValueError("metadata must be a dict.")
                meta_key = meta_key or f"{key}_{meta_key_postfix}"
                if meta_key in d and not self.overwriting:
                    raise KeyError(f"Metadata with key {meta_key} already exists and overwriting=False.")
                d[meta_key] = loaded[1]
        return d
import time
import random
from torch.utils.data import Dataset as TorchDataset
from collections import defaultdict

class SafeDataset(TorchDataset):
    def __init__(self, original_dataset, log_every=100):
        self.dataset = original_dataset
        self.log_every = log_every
        
        # Timing accumulators
        self._call_count = 0
        self._total_load_time = 0
        self._total_transform_time = 0
        self._times = []

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        for _ in range(10):
            try:
                t0 = time.perf_counter()
                result = self.dataset[index]
                t1 = time.perf_counter()
                
                elapsed = t1 - t0
                self._times.append(elapsed)
                self._call_count += 1
                
                # Log every N samples
                if self._call_count % 5 == 0:
                    avg = sum(self._times) / len(self._times)
                    worst = max(self._times)
                    best = min(self._times)
                    print(
                        f"[DataLoader] samples={self._call_count} | "
                        f"avg={avg*1000:.1f}ms | "
                        f"min={best*1000:.1f}ms | "
                        f"max={worst*1000:.1f}ms | "
                        f"last={elapsed*1000:.1f}ms"
                    )
                    # Reset rolling window to avoid stale stats
                    self._times = self._times[-100:]
                
                return result
                
            except Exception as e:
                print(f"============= FAILED TO LOAD: {index} =============")
                print(e)
                print("===================================================")
                index = random.randint(0, len(self.dataset) - 1)
        
        raise RuntimeError("Failed to load 10 consecutive images. Check your data path.")

    def __getattr__(self, attr):
        # 1. Ignore internal python dunder methods during unpickling
        if attr.startswith('__') and attr.endswith('__'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")
            
        # 2. Prevent recursion if 'dataset' hasn't been initialized yet
        if 'dataset' not in self.__dict__:
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{attr}' (dataset not initialized)")
            
        return getattr(self.dataset, attr)
        
            
def get_loader(args):
    train_transforms = Compose(
        [
            LoadImaged_BodyMap(keys=["image"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "nearest"),
            ), # process h5 to here
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z), mode=["minimum", "constant"]),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z), 
                pos=20,
                neg=1,
                num_samples=args.num_samples,
                image_key="image",
                image_threshold=-1,
            ),
            RandRotate90d(
                keys=["image", "label"],
                prob=0.10,
                max_k=3,
            ),
            ToTensord(keys=["image", "label"]),
        ]
    )

    val_transforms = Compose(
        [
            LoadImageh5d(keys=["image"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "nearest"),
            ), 
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z), mode='constant'),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                pos=2,
                neg=0,
                num_samples=args.num_samples,
                image_key="image",
                image_threshold=-1,
            ),
            ToTensord(keys=["image", "label"]),
        ]
    )

        if args.phase == 'train':        
            train_img=[]
            train_lbl=[]
            train_name=[]
            for line in open(os.path.join(args.data_txt_path,  args.dataset_list+'.txt')):
                name = line.strip()
                train_img.append(os.path.join(args.data_root_path, args.img_path, name + '/ct.nii.gz'))
                train_lbl.append(os.path.join(args.data_root_path, args.seg_path, name + '/segmentations/'))
                train_name.append(name)

            
            data_dicts_train = [{'image': image, 'label': label, 'name': name}
                        for image, label, name in zip(train_img, train_lbl, train_name)]
            print('train len {}'.format(len(data_dicts_train)))
        # data_dicts_train=data_dicts_train[:10]
        # breakpoint()
    
        train_dataset = Dataset(data=data_dicts_train, transform=train_transforms)

        train_dataset = SafeDataset(train_dataset)

        #train_sampler = DistributedSampler(dataset=train_dataset, even_divisible=True, shuffle=True) if args.dist else None
        train_sampler = None

        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.num_workers, 
                                    collate_fn=list_data_collate, sampler=train_sampler, pin_memory=True, persistent_workers=True, prefetch_factor=4)
        return train_loader, train_sampler, len(train_dataset)    
    
    if args.phase == 'validation':
        val_img = []
        val_lbl = []
        val_name = []
        for item in args.dataset_list:
            for line in open(os.path.join(args.data_txt_path,  item, 'real_tumor_val_0.txt')):
                name = line.strip().split()[1].split('.')[0]
                val_img.append(os.path.join(args.data_root_path, line.strip().split()[0]))
                val_lbl.append(os.path.join(args.data_root_path, line.strip().split()[1]))
                val_name.append(name)
        data_dicts_val = [{'image': image, 'name': name}
                    for image, label, name in zip(val_img, val_lbl, val_name)]
        print('val len {}'.format(len(data_dicts_val)))
    
        val_dataset = Dataset(data=data_dicts_val, transform=val_transforms)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=list_data_collate)
        return val_loader, val_transforms, len(val_dataset)
    

if __name__ == "__main__":
    train_loader, test_loader = partial_label_dataloader()
    for index, item in enumerate(test_loader):
        print(item['image'].shape, item['label'].shape, item['task_id'])
        input()