"""
Dataset loading utilities with support for ImageFolder and CIFAR-10
Includes OpenCV-based preprocessing alternatives
"""

import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import functional as F
from PIL import Image
from typing import Optional, Tuple, Callable
import torch


class OpenCVTransform:
    """OpenCV-based image preprocessing transform"""
    
    def __init__(self, size: Tuple[int, int] = (224, 224), mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                 std: Tuple[float, float, float] = (0.229, 0.224, 0.225)):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32) * 255.0
        self.std = np.array(std, dtype=np.float32) * 255.0
    
    def __call__(self, image: np.ndarray) -> torch.Tensor:
        # Resize
        image = cv2.resize(image, self.size)
        
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize
        image = image.astype(np.float32)
        image = (image - self.mean) / self.std
        
        # Convert to CHW format
        image = image.transpose(2, 0, 1)
        
        return torch.from_numpy(image)


def get_transforms(img_size: int = 224, use_opencv: bool = False, augment: bool = True):
    """
    Get data transforms for training and validation
    
    Args:
        img_size: Target image size
        use_opencv: Use OpenCV-based transforms instead of torchvision
        augment: Apply data augmentation (for training)
    
    Returns:
        Tuple of (train_transform, val_transform)
    """
    if use_opencv:
        if augment:
            # OpenCV-based augmentation
            train_transform = transforms.Compose([
                transforms.Lambda(lambda x: np.array(x)),
                transforms.Lambda(lambda x: cv2.resize(x, (img_size, img_size))),
                transforms.Lambda(lambda x: cv2.flip(x, 1) if np.random.rand() > 0.5 else x),
                OpenCVTransform(size=(img_size, img_size))
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Lambda(lambda x: np.array(x)),
                OpenCVTransform(size=(img_size, img_size))
            ])
        
        val_transform = transforms.Compose([
            transforms.Lambda(lambda x: np.array(x)),
            OpenCVTransform(size=(img_size, img_size))
        ])
    else:
        # Standard torchvision transforms
        if augment:
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        val_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    return train_transform, val_transform


def load_cifar10_dataset(data_dir: str = './data', img_size: int = 224, batch_size: int = 32,
                         use_opencv: bool = False, num_workers: int = 4):
    """
    Load CIFAR-10 dataset programmatically
    
    Args:
        data_dir: Directory to save/load CIFAR-10 data
        img_size: Target image size
        batch_size: Batch size for DataLoader
        use_opencv: Use OpenCV transforms
        num_workers: Number of workers for DataLoader
    
    Returns:
        Tuple of (train_loader, val_loader, class_names)
    """
    train_transform, val_transform = get_transforms(img_size, use_opencv, augment=True)
    
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    
    val_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    return train_loader, val_loader, class_names


def load_imagefolder_dataset(data_dir: str, img_size: int = 224, batch_size: int = 32,
                             use_opencv: bool = False, num_workers: int = 4,
                             val_split: float = 0.2):
    """
    Load dataset from folder structure: data/train/<class>/*, data/val/<class>/*
    If only train folder exists, split it into train/val
    
    Args:
        data_dir: Root directory containing 'train' and optionally 'val' folders
        img_size: Target image size
        batch_size: Batch size for DataLoader
        use_opencv: Use OpenCV transforms
        num_workers: Number of workers for DataLoader
        val_split: Validation split ratio if val folder doesn't exist
    
    Returns:
        Tuple of (train_loader, val_loader, class_names)
    """
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    train_transform, val_transform = get_transforms(img_size, use_opencv, augment=True)
    
    # Load train dataset
    if not os.path.exists(train_dir):
        raise ValueError(f"Train directory not found: {train_dir}")
    
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    
    # Get class names from train dataset
    class_names = train_dataset.classes
    
    # Load or create validation dataset
    if os.path.exists(val_dir):
        val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    else:
        # Split train dataset
        from torch.utils.data import random_split, Subset
        total_size = len(train_dataset)
        val_size = int(total_size * val_split)
        train_size = total_size - val_size
        
        # Create indices for split
        indices = torch.randperm(total_size, generator=torch.Generator().manual_seed(42)).tolist()
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Create subset datasets
        train_subset = Subset(train_dataset, train_indices)
        
        # Create validation dataset with val transform
        val_dataset_full = datasets.ImageFolder(train_dir, transform=val_transform)
        val_dataset = Subset(val_dataset_full, val_indices)
        train_dataset = train_subset
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, class_names

