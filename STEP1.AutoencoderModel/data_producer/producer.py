import os
import time
import random
import uuid
import torch
import multiprocessing as mp
import hydra
import json
import ctypes
from omegaconf import DictConfig

from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
    SpatialPadd,
)
from monai.transforms.transform import MapTransform
from monai.transforms.io.array import LoadImage
from monai.utils import ensure_tuple, ensure_tuple_rep
from monai.data.image_reader import ImageReader
from monai.utils.enums import PostFix
from copy import deepcopy
import numpy as np
from typing import Optional, Union
from monai.config import DtypeLike, KeysCollection

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

        organ_lbl, meta_information = self.label_transfer(d['segmentations'], d['image'].shape)
        
        if hasattr(d['image'], "meta"):
            from monai.data.meta_tensor import MetaTensor
            chosen_meta = deepcopy(meta_information) if meta_information is not None else deepcopy(d['image'].meta)
            d['label'] = MetaTensor(
                organ_lbl, 
                meta=chosen_meta
            )
        else:
            d['label'] = organ_lbl
            
        d['label_meta_dict'] = meta_information
        return d

    def label_transfer(self, segmentations_list, shape):
        organ_lbl = np.zeros(shape)

        organ_mapping = {
            'spleen': 1, 'bladder': 2, 'gall_bladder': 3, 'esophagus': 4,
            'stomach': 5, 'duodenum': 6, 'colon': 7, 'prostate': 8, 'uterus': 9,
            'spleen_lesion': 10, 'bladder_lesion': 11, 'gallbladder_lesion': 12,
            'esophagus_lesion': 13, 'stomach_lesion': 14, 'duodenum_lesion': 15,
            'colon_lesion': 16, 'prostate_lesion': 17, 'uterus_lesion': 18
        }
        
        meta_information = None 

        for seg_info in segmentations_list:
            file_path = seg_info["file_path"]
            organ_name = seg_info["organ_name"]
            label_idx = organ_mapping[organ_name]
            
            try:
                array, meta_information = self._loader(file_path)
            except Exception as e:
                print(f"Failed organ file: {file_path}")
                raise e
            
            if hasattr(array, "as_tensor"):
                array_np = array.as_tensor().numpy()
            elif hasattr(array, "numpy"):
                array_np = array.numpy()
            else:
                array_np = array

            organ_lbl[array_np > 0] = label_idx

        return organ_lbl, meta_information


def init_worker(cfg: DictConfig):
    global transform_pipeline
    
    transform_pipeline = Compose([
        LoadImaged_BodyMap(keys=["image"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(cfg.dataset.space_x, cfg.dataset.space_y, cfg.dataset.space_z), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=cfg.dataset.a_min, a_max=cfg.dataset.a_max, b_min=cfg.dataset.b_min, b_max=cfg.dataset.b_max, clip=True),
        SpatialPadd(keys=["image", "label"], spatial_size=(cfg.dataset.roi_x, cfg.dataset.roi_y, cfg.dataset.roi_z), mode=["minimum", "constant"]),
        RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", spatial_size=(cfg.dataset.roi_x, cfg.dataset.roi_y, cfg.dataset.roi_z), pos=10, neg=1, num_samples=cfg.producer.num_samples, image_key="image", image_threshold=-1),
        RandRotate90d(keys=["image", "label"], prob=0.10, max_k=3),
        ToTensord(keys=["image", "label"]),
    ])

def continuous_worker_loop(worker_id, shared_json_str, cfg: DictConfig):
    random.seed(os.getpid() + int(time.time()))

    print(f"[Worker {worker_id}] Unpacking data manifest from shared memory...")
    all_data_dicts = json.loads(shared_json_str.value)
    
    init_worker(cfg)
    local_batch_buffer = []
    
    scratch_dir = cfg.producer.scratch_dir
    max_batches = cfg.producer.max_batches
    chunk_size = cfg.producer.chunk_size
    
    consecutive_failures = 0
    
    while True: 
        random.shuffle(all_data_dicts)
        
        for data_dict in all_data_dicts:
            # 1. Termination Check
            if random.random() < 0.1: 
                current_files = [f for f in os.listdir(scratch_dir) if f.endswith(".pt")]
                if len(current_files) >= max_batches:
                    print(f"[Worker {worker_id}] Cache target reached ({len(current_files)}/{max_batches} files). Exiting cleanly.")
                    return
            
            # 2. Process Data
            try:
                outputs = transform_pipeline(data_dict)
                if not isinstance(outputs, list):
                    outputs = [outputs]
                    
                for crop in outputs:
                    pure_image = crop["image"].as_tensor()
                    pure_label = crop["label"].as_tensor()

                    # --- New: Multi-Hot Crop Classification ---
                    # 1. Find all unique integer values present in this crop
                    unique_vals = torch.unique(pure_label).long()
                    
                    # 2. Initialize a 19-element binary vector (0 to 18)
                    class_presence = torch.zeros(19, dtype=torch.uint8)
                    
                    # 3. Filter out any unexpected values to prevent out-of-bounds indexing
                    valid_indices = unique_vals[(unique_vals >= 0) & (unique_vals <= 18)]
                    
                    # 4. Set the present indices to 1
                    class_presence[valid_indices] = 1
                    # ------------------------------------------

                    local_batch_buffer.append({
                        "image": pure_image.clone().to(torch.float16), 
                        "label": pure_label.clone().to(torch.int8),
                        "contents": class_presence  # Now saves the full binary map!
                    })

                    if worker_id % 24 == 0 and len(local_batch_buffer) % 16 == 0:
                        print(f"[Worker {worker_id}] Buffer filling: {len(local_batch_buffer)} / {chunk_size}")
                
                consecutive_failures = 0
                    
            except Exception as e:
                print("============= FAILED TO LOAD/TRANSFORM =============")
                print(f"[Worker {worker_id}] Failed on {data_dict.get('name')}")
                print(e)
                print("====================================================")
                
                consecutive_failures += 1
                if consecutive_failures >= 10:
                    raise RuntimeError(f"[Worker {worker_id}] Failed 10 consecutive times. Weka storage might be offline.")
                continue
                
            # 3. Save batched chunk when full
            if len(local_batch_buffer) >= chunk_size:
                
                current_files = [f for f in os.listdir(scratch_dir) if f.endswith(".pt")]
                if len(current_files) >= max_batches:
                    print(f"[Worker {worker_id}] Cache met right before write. Discarding buffer and exiting.")
                    return

                process_buffer = local_batch_buffer[:chunk_size]
                local_batch_buffer = local_batch_buffer[chunk_size:] 
                
                batch_dict = {
                    "image": torch.stack([d["image"] for d in process_buffer]),
                    "label": torch.stack([d["label"] for d in process_buffer]),
                    "contents": torch.stack([d["contents"] for d in process_buffer]) # Stacks into shape [256, 19]
                }
                
                file_id = f"batch_w{worker_id}_{uuid.uuid4().hex[:8]}.pt"
                temp_path = os.path.join(scratch_dir, f"_{file_id}.tmp")
                final_path = os.path.join(scratch_dir, file_id)
                
                torch.save(batch_dict, temp_path)
                os.rename(temp_path, final_path)


def start_producer(cfg: DictConfig):
    scratch_dir = cfg.producer.scratch_dir
    os.makedirs(scratch_dir, exist_ok=True)
    
    manifest_path = os.path.join('dataset_manifest.json')
    
    print(f"Reading raw manifest text from {manifest_path}...")
    with open(manifest_path, 'r') as f:
        raw_text = f.read()
        
    print("Allocating shared memory space for manifest string...")
    shared_json_str = mp.Array(ctypes.c_char, raw_text.encode('utf-8'), lock=False)
    
    del raw_text

    num_workers = cfg.producer.num_workers
    print(f"Starting {num_workers} continuous workers writing chunks of {cfg.producer.chunk_size} to {scratch_dir}...")
    
    workers = []
    mp.set_start_method('spawn', force=True) 
    
    for i in range(num_workers):
        p = mp.Process(target=continuous_worker_loop, args=(i, shared_json_str, cfg))
        p.start()
        workers.append(p)
        
    for p in workers:
        p.join()


@hydra.main(config_path='../config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig, args=None):
    start_producer(cfg)

if __name__ == "__main__":
    run()