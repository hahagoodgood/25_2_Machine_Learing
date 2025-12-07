"""
Training script for COVID-19 Image Classification
Supports VGG16, ResNet50, and DenseNet121 models
"""
import os
import argparse
import time
import copy
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import config
from dataset import get_data_loaders
from models import get_model
from utils import set_seed, get_device, save_checkpoint, count_parameters


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    
    def __init__(self, patience=10, min_delta=0.0, mode='max'):
        """
        Args:
            patience (int): How many epochs to wait after last improvement
            min_delta (float): Minimum change to qualify as improvement
            mode (str): 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'max':
            if score < self.best_score + self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
        else:  # mode == 'min'
            if score > self.best_score - self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, writer=None):
    """
    Train for one epoch
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        writer: TensorBoard writer
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    # Log to TensorBoard
    if writer:
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Accuracy/train', accuracy, epoch)
    
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device, epoch, writer=None):
    """
    Validate the model
    
    Args:
        model: Neural network model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        epoch: Current epoch number
        writer: TensorBoard writer
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
    
    avg_loss = running_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    # Log to TensorBoard
    if writer:
        writer.add_scalar('Loss/val', avg_loss, epoch)
        writer.add_scalar('Accuracy/val', accuracy, epoch)
    
    return avg_loss, accuracy


def test(model, test_loader, device):
    """
    Test the model
    
    Args:
        model: Neural network model
        test_loader: Test data loader
        device: Device to test on
        
    Returns:
        tuple: (accuracy, all_labels, all_predictions, all_probabilities)
    """
    model.eval()
    correct = 0
    total = 0
    
    all_labels = []
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            
            pbar.set_postfix({'acc': 100. * correct / total})
    
    accuracy = 100. * correct / total
    
    return accuracy, all_labels, all_predictions, all_probabilities


def train_model(model_name, epochs=None, batch_size=None, learning_rate=None, device=None):
    """
    Main training function
    
    Args:
        model_name (str): Name of the model to train
        epochs (int, optional): Number of epochs
        batch_size (int, optional): Batch size
        learning_rate (float, optional): Learning rate
        device (torch.device, optional): Device to train on
    """
    # Set defaults from config
    if epochs is None:
        epochs = config.NUM_EPOCHS
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if learning_rate is None:
        learning_rate = config.LEARNING_RATE
    if device is None:
        device = get_device()
    
    # Set random seed for reproducibility
    set_seed(config.RANDOM_SEED)
    
    print('='*80)
    print(f'Training {model_name.upper()} Model')
    print('='*80)
    print(f'Configuration:')
    print(f'  Epochs: {epochs}')
    print(f'  Batch size: {batch_size}')
    print(f'  Learning rate: {learning_rate}')
    print(f'  Device: {device}')
    print(f'  Early stopping patience: {config.EARLY_STOPPING_PATIENCE}')
    print('='*80)
    
    # Load data
    print('\nLoading data...')
    train_loader, val_loader, test_loader, class_weights = get_data_loaders(
        config.DATASET_DIR,
        batch_size=batch_size
    )
    
    # Create model
    print(f'\nCreating {model_name} model...')
    model = get_model(model_name, num_classes=config.NUM_CLASSES, pretrained=True)
    model = model.to(device)
    
    # Count parameters
    count_parameters(model)
    
    # Loss function with class weights for imbalanced dataset
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=config.LR_SCHEDULER_FACTOR,
        patience=config.LR_SCHEDULER_PATIENCE,
        min_lr=config.LR_SCHEDULER_MIN_LR,
        verbose=True
    )
    
    # TensorBoard writer
    writer = SummaryWriter(os.path.join(config.TENSORBOARD_DIR, model_name))
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE, mode='max')
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    print('\n' + '='*80)
    print('Starting training...')
    print('='*80)
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        print(f'\nEpoch {epoch}/{epochs}')
        print('-' * 40)
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch, writer
        )
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch summary
        print(f'\nEpoch {epoch} Summary:')
        print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        print(f'  Learning Rate: {current_lr:.2e}')
        
        # Save best model
        if val_acc > best_val_acc:
            print(f'  ✓ Validation accuracy improved from {best_val_acc:.2f}% to {val_acc:.2f}%')
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            
            # Save checkpoint
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f'{model_name}_best.pth')
            save_checkpoint({
                'epoch': epoch,
                'model_name': model_name,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'history': history
            }, checkpoint_path)
        
        # Early stopping
        early_stopping(val_acc)
        if early_stopping.early_stop:
            print(f'\nEarly stopping triggered after {epoch} epochs')
            break
    
    # Training complete
    training_time = time.time() - start_time
    print('\n' + '='*80)
    print('Training completed!')
    print(f'Total training time: {training_time / 60:.2f} minutes')
    print(f'Best validation accuracy: {best_val_acc:.2f}%')
    print('='*80)
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    # Test the model
    print('\nEvaluating on test set...')
    test_acc, test_labels, test_preds, test_probs = test(model, test_loader, device)
    print(f'Test Accuracy: {test_acc:.2f}%')
    
    # Save final checkpoint with test results
    final_checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f'{model_name}_final.pth')
    save_checkpoint({
        'model_name': model_name,
        'model_state_dict': model.state_dict(),
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'history': history,
        'training_time': training_time
    }, final_checkpoint_path)
    
    writer.close()
    
    return model, history, test_acc


def main():
    parser = argparse.ArgumentParser(description='Train COVID-19 Classification Model')
    parser.add_argument('--model', type=str, required=True,
                       choices=['vgg16', 'resnet50', 'densenet121'],
                       help='Model architecture to train')
    parser.add_argument('--epochs', type=int, default=None,
                       help=f'Number of epochs (default: {config.NUM_EPOCHS})')
    parser.add_argument('--batch_size', type=int, default=None,
                       help=f'Batch size (default: {config.BATCH_SIZE})')
    parser.add_argument('--lr', type=float, default=None,
                       help=f'Learning rate (default: {config.LEARNING_RATE})')
    
    args = parser.parse_args()
    
    # Train the model
    train_model(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )


if __name__ == '__main__':
    main()
