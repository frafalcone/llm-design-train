import torch
from torch.utils.data import Dataset, DataLoader, Subset
import os
import numpy


class CustomDataset(Dataset):
    def __init__(self, bin_path, max_length):
        self.bin_path = bin_path
        self.max_length = max_length
        
        file_size = os.path.getsize(bin_path)
        self.num_tokens = file_size // numpy.dtype(numpy.int32).itemsize
        self.data = numpy.memmap(bin_path, dtype=numpy.int32, mode='r')

    def __len__(self):
        return max(0, (self.num_tokens - 1) // self.max_length)

    def __getitem__(self, idx):
        start = idx * self.max_length
        end = start + self.max_length
        
        x = torch.from_numpy(self.data[start:end].astype(numpy.int64))
        y = torch.from_numpy(self.data[start+1:end+1].astype(numpy.int64))
        
        return x, y

def create_dataloader(bin_path, data_configuration, shuffle=True, drop_last=True, pin_memory=True, percentage=1.0):
    context_size = data_configuration.get("context_size", 1024)
    batch = data_configuration.get("batch", 32)
    num_workers = data_configuration.get("num_workers", 0)
    
    dataset = CustomDataset(bin_path, context_size)
    
    if percentage < 1.0:
        indices = list(range(int(len(dataset) * percentage)))
        dataset = Subset(dataset, indices)

    effective_pin_memory = pin_memory and torch.cuda.is_available()

    return DataLoader(
        dataset,
        batch_size=batch,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=effective_pin_memory,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )