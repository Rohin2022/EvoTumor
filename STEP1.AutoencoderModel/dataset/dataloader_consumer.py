import os
import time
import random
import torch
from torch.utils.data import IterableDataset, DataLoader

# ==========================================
# 1. ATOMIC CACHE CONSUMER DATASET
# ==========================================
class CacheConsumerDataset(IterableDataset):
    def __init__(self, cache_dir):
        super().__init__()
        self.cache_dir = cache_dir

    def __iter__(self):
        """
        Infinite generator that safely pulls .pt batches from the scratch directory.
        Works flawlessly across multiple GPUs and DataLoader workers via atomic locking.
        """
        while True:
            # 1. Look for valid, completely written .pt files
            try:
                valid_files = [
                    f for f in os.listdir(self.cache_dir) 
                    if f.endswith('.pt') and not f.startswith('_')
                ]
            except FileNotFoundError:
                valid_files = []

            # 2. If the CPU producer is catching up, snooze for a second
            if not valid_files:
                time.sleep(1.0)
                continue
            
            # Shuffle so different GPUs/workers don't instantly fight for the exact same file index
            random.shuffle(valid_files)
            file_processed = False
            
            for file_name in valid_files:
                original_path = os.path.join(self.cache_dir, file_name)
                locked_path = os.path.join(self.cache_dir, f"_locked_{file_name}")
                
                # 3. ATOMIC LOCK: Try to claim the file by renaming it.
                # If another GPU or worker beats us to it, this throws an error and we skip.
                try:
                    os.rename(original_path, locked_path)
                except OSError:
                    continue
                    
                # 4. Lock Acquired! Load the data and destroy the file
# 4. Lock Acquired! Load the data and destroy the file
                try:
                    batch_data = torch.load(locked_path, map_location='cpu', weights_only=True)
                    #os.remove(locked_path)
                    
                    images = batch_data["image"]
                    labels = batch_data["label"]
                    contents = batch_data["contents"]
                    
                    num_crops = images.shape[0]
                    
                    shuffle_idx = torch.randperm(num_crops)
                    
                    images = images[shuffle_idx]
                    labels = labels[shuffle_idx]
                    contents = contents[shuffle_idx]
                    
                    for i in range(num_crops):
                        yield {
                            "image": images[i].to(torch.float32),
                            "label": labels[i].to(torch.int64),
                            "contents": contents[i].to(torch.float32) 
                        }
                    
                except Exception as e:
                    print(f"[Consumer Error] Failed to read {locked_path}: {e}")
                    if os.path.exists(locked_path):
                        os.remove(locked_path)
            
            if not file_processed:
                time.sleep(0.5)

# ==========================================
# 2. GET_LOADER FUNCTION
# ==========================================
def get_loader(args):
    """
    Returns the training loader connected directly to the pre-computed Weka cache.
    """
    if args.phase == 'train':
        print(f"Connecting Consumer to Cache Directory: {args.scratch_dir}")
        
        # Instantiate the consumer dataset
        train_dataset = CacheConsumerDataset(cache_dir=args.scratch_dir)
        
        # Note: Because this is an IterableDataset, we DO NOT need a DistributedSampler.
        # DDP (Distributed Data Parallel) will automatically assign a DataLoader to each H100.
        # The atomic `os.rename` lock prevents the 4 GPUs from duplicating data natively!
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size,     # E.g., 16 or 32 per GPU
            num_workers=args.num_workers,   # E.g., 4 or 8
            pin_memory=True,                # Crucial for fast CPU -> GPU transfers
            prefetch_factor=2               # Keeps the queue full
        )
        
        # In an infinite IterableDataset, length is technically infinite. 
        # You will control epochs mathematically in your training loop.
        dataset_len = float('inf') 
        
        return train_loader
        
    elif args.phase == 'validation':
        raise NotImplementedError("Validation loader is handled in dataloader_validation.py")

if __name__ == "__main__":
    # Dummy args class for quick local testing
    class Args:
        phase = 'train'
        scratch_dir = "/scratch/rpinise1/MultiTumorSynthesis/AutoencoderCache"
        batch_size = 8
        num_workers = 2

    args = Args()
    train_loader, _, _ = get_loader(args)
    
    print("Testing stream...")
    for index, batch in enumerate(train_loader):
        print(f"Batch {index}: Image {batch['image'].shape}, Label {batch['label'].shape}, Contents {batch['contents'].shape}")
        if index >= 5:
            break