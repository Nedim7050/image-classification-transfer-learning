"""
Inference utilities for image classification
Supports single image prediction with top-k results
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights, efficientnet_b0, EfficientNet_B0_Weights
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Dict
import os


def load_model(model_path: str, device: str = 'cpu', num_classes: Optional[int] = None) -> Tuple[nn.Module, List[str], str]:
    """
    Load trained model from checkpoint
    
    Args:
        model_path: Path to model checkpoint
        device: Device to load model on
        num_classes: Number of classes (will be inferred from checkpoint if not provided)
    
    Returns:
        Tuple of (model, class_names, model_name)
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model info from checkpoint
    model_name = checkpoint.get('model_name', 'resnet50')
    num_classes = num_classes or checkpoint.get('num_classes', 10)
    class_names = checkpoint.get('class_names', [f'class_{i}' for i in range(num_classes)])
    
    # Create model
    if model_name.lower() == 'resnet50':
        model = resnet50(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif model_name.lower() == 'efficientnet':
        model = efficientnet_b0(weights=None)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Load weights
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, class_names, model_name


def preprocess_image(image_path: str, img_size: int = 224, use_opencv: bool = False) -> torch.Tensor:
    """
    Preprocess image for inference
    
    Args:
        image_path: Path to image file
        img_size: Target image size
        use_opencv: Use OpenCV for preprocessing
    
    Returns:
        Preprocessed image tensor
    """
    if use_opencv:
        # OpenCV-based preprocessing
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image = cv2.resize(image, (img_size, img_size))
        
        # Normalize
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = image.astype(np.float32) / 255.0
        image = (image - mean) / std
        
        # Convert to CHW format
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).float()
    else:
        # PIL-based preprocessing
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = transform(image)
    
    return image.unsqueeze(0)  # Add batch dimension


def predict(image_path: str, model_path: str, topk: int = 3, device: str = 'cpu',
           img_size: int = 224, use_opencv: bool = False) -> List[Tuple[str, float]]:
    """
    Predict top-k classes for an image
    
    Args:
        image_path: Path to image file
        model_path: Path to model checkpoint
        topk: Number of top predictions to return
        device: Device to run inference on
        img_size: Image size for preprocessing
        use_opencv: Use OpenCV for preprocessing
    
    Returns:
        List of tuples (class_name, probability) sorted by probability (descending)
    """
    # Load model
    model, class_names, _ = load_model(model_path, device)
    
    # Preprocess image
    image_tensor = preprocess_image(image_path, img_size, use_opencv)
    image_tensor = image_tensor.to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        top_probs, top_indices = torch.topk(probabilities, min(topk, len(class_names)))
    
    # Format results
    results = []
    for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
        results.append((class_names[idx], float(prob)))
    
    return results


def predict_batch(image_paths: List[str], model_path: str, topk: int = 3,
                 device: str = 'cpu', img_size: int = 224, batch_size: int = 32,
                 use_opencv: bool = False) -> List[List[Tuple[str, float]]]:
    """
    Predict top-k classes for a batch of images
    
    Args:
        image_paths: List of paths to image files
        model_path: Path to model checkpoint
        topk: Number of top predictions to return per image
        device: Device to run inference on
        img_size: Image size for preprocessing
        batch_size: Batch size for processing
        use_opencv: Use OpenCV for preprocessing
    
    Returns:
        List of prediction results for each image
    """
    # Load model
    model, class_names, _ = load_model(model_path, device)
    
    all_results = []
    
    # Process in batches
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        
        # Preprocess batch
        for img_path in batch_paths:
            img_tensor = preprocess_image(img_path, img_size, use_opencv)
            batch_images.append(img_tensor)
        
        batch_tensor = torch.cat(batch_images, dim=0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(batch_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probabilities, min(topk, len(class_names)), dim=1)
        
        # Format results
        for probs, indices in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
            results = [(class_names[idx], float(prob)) for prob, idx in zip(probs, indices)]
            all_results.append(results)
    
    return all_results


def predict_from_array(image_array: np.ndarray, model_path: str, topk: int = 3,
                      device: str = 'cpu', img_size: int = 224) -> List[Tuple[str, float]]:
    """
    Predict from numpy array (useful for Streamlit or web apps)
    
    Args:
        image_array: numpy array of image (RGB format, uint8)
        model_path: Path to model checkpoint
        topk: Number of top predictions to return
        device: Device to run inference on
        img_size: Image size for preprocessing
    
    Returns:
        List of tuples (class_name, probability) sorted by probability (descending)
    """
    # Load model
    model, class_names, _ = load_model(model_path, device)
    
    # Preprocess
    if len(image_array.shape) == 3:
        # Single image
        image = Image.fromarray(image_array)
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform(image).unsqueeze(0).to(device)
    else:
        raise ValueError("Image array must be 3D (H, W, C)")
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        top_probs, top_indices = torch.topk(probabilities, min(topk, len(class_names)))
    
    # Format results
    results = []
    for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
        results.append((class_names[idx], float(prob)))
    
    return results

