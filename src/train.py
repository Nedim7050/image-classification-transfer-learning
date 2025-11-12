"""
Training script for image classification with transfer learning
Supports CLI arguments, checkpointing, early stopping, and TensorBoard logging
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import load_cifar10_dataset, load_imagefolder_dataset
from src.utils import calculate_accuracy, save_checkpoint, get_model_summary


def create_model(model_name: str = 'resnet50', num_classes: int = 10, pretrained: bool = True,
                freeze_backbone: bool = False, unfreeze_last_n: int = 0):
    """
    Create model with transfer learning
    
    Args:
        model_name: Model architecture ('resnet50' or 'efficientnet')
        num_classes: Number of output classes
        pretrained: Use pretrained weights
        freeze_backbone: Freeze all backbone layers
        unfreeze_last_n: Unfreeze last N layers (0 = all frozen if freeze_backbone=True)
    
    Returns:
        Model and device
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_name.lower() == 'resnet50':
        from torchvision.models import resnet50, ResNet50_Weights
        
        if pretrained:
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            model = resnet50(weights=None)
        
        # Replace classifier
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
            
            # Unfreeze last N layers
            if unfreeze_last_n > 0:
                layers = list(model.children())
                for layer in layers[-unfreeze_last_n:]:
                    for param in layer.parameters():
                        param.requires_grad = True
        
        # Always unfreeze classifier
        for param in model.fc.parameters():
            param.requires_grad = True
    
    elif model_name.lower() == 'efficientnet':
        try:
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
            
            if pretrained:
                model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            else:
                model = efficientnet_b0(weights=None)
            
            # Replace classifier
            num_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_features, num_classes)
            
            # Freeze backbone if requested
            if freeze_backbone:
                for param in model.features.parameters():
                    param.requires_grad = False
                
                # Unfreeze last N layers
                if unfreeze_last_n > 0:
                    layers = list(model.features.children())
                    for layer in layers[-unfreeze_last_n:]:
                        for param in layer.parameters():
                            param.requires_grad = True
            
            # Always unfreeze classifier
            for param in model.classifier.parameters():
                param.requires_grad = True
        
        except ImportError:
            print("EfficientNet not available, falling back to ResNet50")
            return create_model('resnet50', num_classes, pretrained, freeze_backbone, unfreeze_last_n)
    
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose 'resnet50' or 'efficientnet'")
    
    model = model.to(device)
    return model, device


def train_epoch(model, train_loader, criterion, optimizer, device, use_amp=False, scaler=None):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        if use_amp:
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        acc = calculate_accuracy(outputs, labels)
        running_acc += acc
        total += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.2f}%'})
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = running_acc / len(train_loader)
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device, use_amp=False):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            if use_amp:
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            acc = calculate_accuracy(outputs, labels)
            running_acc += acc
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.2f}%'})
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = running_acc / len(val_loader)
    
    return epoch_loss, epoch_acc, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description='Train image classification model with transfer learning')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Directory containing data (or where CIFAR-10 will be downloaded)')
    parser.add_argument('--use_cifar10', action='store_true',
                       help='Use CIFAR-10 dataset instead of ImageFolder')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Image size for training')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--use_opencv', action='store_true',
                       help='Use OpenCV transforms instead of torchvision')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='resnet50',
                       choices=['resnet50', 'efficientnet'],
                       help='Model architecture')
    parser.add_argument('--num_classes', type=int, default=10,
                       help='Number of classes')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained weights')
    parser.add_argument('--freeze_backbone', action='store_true',
                       help='Freeze backbone layers')
    parser.add_argument('--unfreeze_last_n', type=int, default=0,
                       help='Unfreeze last N layers (0 = all frozen if freeze_backbone=True)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu). Auto-detect if not specified')
    parser.add_argument('--use_amp', action='store_true',
                       help='Use Automatic Mixed Precision for faster training')
    
    # Checkpointing and logging
    parser.add_argument('--save_dir', type=str, default='./models',
                       help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='Directory for TensorBoard logs')
    parser.add_argument('--early_stopping_patience', type=int, default=None,
                       help='Early stopping patience (None to disable)')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Load dataset
    print("\nLoading dataset...")
    if args.use_cifar10:
        train_loader, val_loader, class_names = load_cifar10_dataset(
            args.data_dir, args.img_size, args.batch_size, args.use_opencv, args.num_workers
        )
        args.num_classes = 10
    else:
        train_loader, val_loader, class_names = load_imagefolder_dataset(
            args.data_dir, args.img_size, args.batch_size, args.use_opencv, args.num_workers
        )
        args.num_classes = len(class_names)
    
    print(f"Number of classes: {args.num_classes}")
    print(f"Class names: {class_names}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create model
    print("\nCreating model...")
    model, device = create_model(
        args.model, args.num_classes, args.pretrained,
        args.freeze_backbone, args.unfreeze_last_n
    )
    get_model_summary(model, (3, args.img_size, args.img_size))
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Mixed precision
    scaler = GradScaler() if args.use_amp else None
    
    # TensorBoard writer
    writer = SummaryWriter(args.log_dir)
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    # Resume from checkpoint
    start_epoch = 0
    best_acc = 0.0
    patience_counter = 0
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint.get('best_acc', 0.0)
        print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    print("\nStarting training...")
    print(f"{'='*60}")
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, args.use_amp, scaler)
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device, args.use_amp)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # TensorBoard logging
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Save checkpoint
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_acc': best_acc,
            'val_acc': val_acc,
            'class_names': class_names,
            'model_name': args.model,
            'num_classes': args.num_classes
        }, checkpoint_path, is_best)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  Best Val Acc: {best_acc:.2f}%")
        
        # Early stopping
        if args.early_stopping_patience and patience_counter >= args.early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            print(f"No improvement for {patience_counter} epochs")
            break
    
    writer.close()
    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {os.path.join(args.save_dir, 'best_model.pth')}")
    print(f"TensorBoard logs: {args.log_dir}")
    print(f"Run: tensorboard --logdir {args.log_dir}")


if __name__ == '__main__':
    main()

