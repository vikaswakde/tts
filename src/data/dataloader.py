"""
Data loading utilities for TTS system.
Handles batch creation and collation of samples.
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict, List
import numpy as np

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function to create batches.
    Pads sequences to the same length within a batch.
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        Dictionary containing batched and padded tensors
    """
    # Get maximum lengths in batch
    max_text_len = max(x["text_length"] for x in batch)
    max_mel_len = max(x["mel_length"] for x in batch)
    
    # Initialize tensors
    text_padded = torch.zeros(len(batch), max_text_len, dtype=torch.long)
    mel_padded = torch.zeros(len(batch), batch[0]["mel"].size(0), max_mel_len)
    
    # Store lengths for packing/masking
    text_lengths = torch.zeros(len(batch), dtype=torch.long)
    mel_lengths = torch.zeros(len(batch), dtype=torch.long)
    
    # Pad sequences
    for i, sample in enumerate(batch):
        text = sample["text"]
        mel = sample["mel"]
        text_len = sample["text_length"]
        mel_len = sample["mel_length"]
        
        # Pad text
        text_padded[i, :text_len] = text
        
        # Pad mel spectrogram
        mel_padded[i, :, :mel_len] = mel
        
        # Store lengths
        text_lengths[i] = text_len
        mel_lengths[i] = mel_len
    
    return {
        "text_padded": text_padded,
        "mel_padded": mel_padded,
        "text_lengths": text_lengths,
        "mel_lengths": mel_lengths,
        "batch_size": torch.tensor(len(batch))
    }

def create_dataloader(dataset,
                     batch_size: int,
                     num_workers: int = 0,
                     shuffle: bool = True) -> DataLoader:
    """
    Create a DataLoader for the dataset.
    
    Args:
        dataset: Dataset instance
        batch_size: Number of samples per batch
        num_workers: Number of worker processes
        shuffle: Whether to shuffle the data
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False,  # True if using GPU
        drop_last=True     # Drop incomplete batches
    ) 