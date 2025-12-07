"""
Utility functions for COVID-19 Image Classification Project
"""
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import json


def set_seed(seed=42):
    """
    Set random seeds for reproducibility
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """
    Get the appropriate device (GPU if available, else CPU)
    
    Returns:
        torch.device: Device object
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f'Using GPU: {torch.cuda.get_device_name(0)}')
        print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
    else:
        print('Using CPU')
    return device


def save_checkpoint(state, filepath):
    """
    Save model checkpoint
    
    Args:
        state (dict): State dictionary containing model and optimizer states
        filepath (str): Path to save checkpoint
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)
    print(f'Checkpoint saved to {filepath}')


def load_checkpoint(filepath, model, optimizer=None):
    """
    Load model checkpoint
    
    Args:
        filepath (str): Path to checkpoint file
        model (nn.Module): Model to load weights into
        optimizer (Optimizer, optional): Optimizer to load state into
        
    Returns:
        dict: Checkpoint information (epoch, best_acc, etc.)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f'Checkpoint not found at {filepath}')
    
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f'Checkpoint loaded from {filepath}')
    return checkpoint


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None, title='Confusion Matrix'):
    """
    Plot confusion matrix
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        class_names (list): List of class names
        save_path (str, optional): Path to save the plot
        title (str): Plot title
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title(title, fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Confusion matrix saved to {save_path}')
    
    plt.show()
    return cm


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path=None):
    """
    Plot training and validation loss/accuracy curves
    
    Args:
        train_losses (list): Training losses per epoch
        val_losses (list): Validation losses per epoch
        train_accs (list): Training accuracies per epoch
        val_accs (list): Validation accuracies per epoch
        save_path (str, optional): Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Training curves saved to {save_path}')
    
    plt.show()


def calculate_metrics(y_true, y_pred, class_names):
    """
    Calculate comprehensive classification metrics
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        class_names (list): List of class names
        
    Returns:
        dict: Dictionary containing various metrics
    """
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'overall_accuracy': report['accuracy'],
        'per_class_metrics': {}
    }
    
    for i, class_name in enumerate(class_names):
        metrics['per_class_metrics'][class_name] = {
            'precision': report[class_name]['precision'],
            'recall': report[class_name]['recall'],
            'f1-score': report[class_name]['f1-score'],
            'support': report[class_name]['support']
        }
    
    return metrics


def save_metrics(metrics, filepath):
    """
    Save metrics to JSON file
    
    Args:
        metrics (dict): Metrics dictionary
        filepath (str): Path to save metrics
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f'Metrics saved to {filepath}')


def plot_roc_curves(y_true, y_probs, class_names, save_path=None):
    """
    Plot ROC curves for multi-class classification
    
    Args:
        y_true (array-like): True labels
        y_probs (array-like): Predicted probabilities (n_samples, n_classes)
        class_names (list): List of class names
        save_path (str, optional): Path to save the plot
    """
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    plt.figure(figsize=(10, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (class_name, color) in enumerate(zip(class_names, colors[:n_classes])):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=color, linewidth=2,
                label=f'{class_name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Multi-Class Classification', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'ROC curves saved to {save_path}')
    
    plt.show()


def count_parameters(model):
    """
    Count total and trainable parameters in a model
    
    Args:
        model (nn.Module): PyTorch model
        
    Returns:
        tuple: (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    print(f'Non-trainable parameters: {total_params - trainable_params:,}')
    
    return total_params, trainable_params
