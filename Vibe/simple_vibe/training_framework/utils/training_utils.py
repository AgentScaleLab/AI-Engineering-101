"""
Utility functions for training optimization
"""
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import os


def get_device():
    """Get the best available device for training"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def create_dataloader(dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True):
    """
    Create an optimized dataloader with best practices
    """
    if torch.cuda.is_available():
        # Use pinned memory and multiple workers for GPU training
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0)  # Keep workers alive between epochs if num_workers > 0
        )
    else:
        # For CPU training, fewer workers may be better
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory if num_workers > 0 else False,
            persistent_workers=(num_workers > 0) if num_workers > 0 else False
        )

    return dataloader


def calculate_accuracy(outputs, targets):
    """Calculate accuracy for classification tasks"""
    _, predicted = torch.max(outputs.data, 1)
    total = targets.size(0)
    correct = (predicted == targets).sum().item()
    return 100 * correct / total


def format_time(seconds):
    """Format time in seconds to a human-readable string"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def print_model_info(model):
    """Print model information including parameter count"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")