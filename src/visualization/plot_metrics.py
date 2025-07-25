import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_training_history(history: dict, save_path: Path = None):
    """
    Plot training and validation metrics over epochs.
    
    Args:
        history: Dictionary containing training history
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot metrics
    plt.subplot(1, 2, 2)
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    for metric in metrics:
        values = [epoch_metrics[metric] for epoch_metrics in history['val_metrics']]
        plt.plot(values, label=metric.capitalize())
    
    plt.title('Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Training history plot saved to {save_path}")
    
    plt.close()

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path = None):
    """
    Plot confusion matrix for model predictions.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save the plot
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Confusion matrix plot saved to {save_path}")
    
    plt.close()

def plot_attention_weights(text: str, attention_weights: np.ndarray, save_path: Path = None):
    """
    Plot attention weights for a given text.
    
    Args:
        text: Input text
        attention_weights: Attention weights for each token
        save_path: Path to save the plot
    """
    tokens = text.split()
    plt.figure(figsize=(12, 4))
    
    # Plot attention weights
    plt.bar(range(len(tokens)), attention_weights)
    plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
    plt.title('Attention Weights')
    plt.xlabel('Tokens')
    plt.ylabel('Attention Weight')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Attention weights plot saved to {save_path}")
    
    plt.close()

def plot_model_comparison(metrics: dict, save_path: Path = None):
    """
    Plot comparison of different models' performance.
    
    Args:
        metrics: Dictionary containing model metrics
        save_path: Path to save the plot
    """
    models = list(metrics.keys())
    metric_names = ['accuracy', 'precision', 'recall', 'f1']
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(models))
    width = 0.2
    
    for i, metric in enumerate(metric_names):
        values = [metrics[model][metric] for model in models]
        plt.bar(x + i*width, values, width, label=metric.capitalize())
    
    plt.title('Model Performance Comparison')
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.xticks(x + width*1.5, models, rotation=45)
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Model comparison plot saved to {save_path}")
    
    plt.close()

def plot_feature_importance(feature_importance: dict, save_path: Path = None):
    """
    Plot feature importance scores.
    
    Args:
        feature_importance: Dictionary containing feature importance scores
        save_path: Path to save the plot
    """
    features = list(feature_importance.keys())
    importance = list(feature_importance.values())
    
    # Sort by importance
    sorted_idx = np.argsort(importance)
    features = [features[i] for i in sorted_idx]
    importance = [importance[i] for i in sorted_idx]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(features)), importance)
    plt.yticks(range(len(features)), features)
    plt.title('Feature Importance')
    plt.xlabel('Importance Score')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Feature importance plot saved to {save_path}")
    
    plt.close()

def main():
    # Create visualization directory
    vis_dir = Path(__file__).parent.parent.parent / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    
    # Example usage
    history = {
        'train_loss': [0.5, 0.4, 0.3],
        'val_loss': [0.45, 0.35, 0.25],
        'val_metrics': [
            {'accuracy': 0.8, 'precision': 0.75, 'recall': 0.85, 'f1': 0.8},
            {'accuracy': 0.85, 'precision': 0.8, 'recall': 0.9, 'f1': 0.85},
            {'accuracy': 0.9, 'precision': 0.85, 'recall': 0.95, 'f1': 0.9}
        ]
    }
    
    # Plot training history
    plot_training_history(history, save_path=vis_dir / "training_history.png")
    
    # Example confusion matrix
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 0])
    plot_confusion_matrix(y_true, y_pred, save_path=vis_dir / "confusion_matrix.png")
    
    # Example model comparison
    metrics = {
        'BERT': {'accuracy': 0.85, 'precision': 0.82, 'recall': 0.88, 'f1': 0.85},
        'BiLSTM': {'accuracy': 0.78, 'precision': 0.75, 'recall': 0.81, 'f1': 0.78},
        'Hybrid': {'accuracy': 0.92, 'precision': 0.9, 'recall': 0.94, 'f1': 0.92}
    }
    plot_model_comparison(metrics, save_path=vis_dir / "model_comparison.png")
    
    # Example feature importance
    feature_importance = {
        'BERT': 0.4,
        'BiLSTM': 0.3,
        'Attention': 0.2,
        'TF-IDF': 0.1
    }
    plot_feature_importance(feature_importance, save_path=vis_dir / "feature_importance.png")

if __name__ == "__main__":
    main() 