import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self,
                 model: nn.Module,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 learning_rate: float = 2e-5,
                 num_epochs: int = 10,
                 early_stopping_patience: int = 3):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate
        )
        
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(input_ids, attention_mask)
            loss = self.criterion(outputs['logits'], labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def evaluate(self, eval_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs['logits'], labels)
                
                total_loss += loss.item()
                
                preds = torch.argmax(outputs['logits'], dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_labels, all_preds)
        metrics['loss'] = total_loss / len(eval_loader)
        
        return total_loss / len(eval_loader), metrics
    
    def _calculate_metrics(self, labels: List[int], preds: List[int]) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='weighted'
        )
        accuracy = accuracy_score(labels, preds)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_training_steps: int) -> Dict[str, List[float]]:
        """Train the model with early stopping."""
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': []
        }
        
        for epoch in range(self.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")
            
            # Training
            train_loss = self.train_epoch(train_loader)
            history['train_loss'].append(train_loss)
            
            # Validation
            val_loss, val_metrics = self.evaluate(val_loader)
            history['val_loss'].append(val_loss)
            history['val_metrics'].append(val_metrics)
            
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}")
            logger.info(f"Val Metrics: {val_metrics}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    logger.info("Early stopping triggered")
                    break
            
            scheduler.step()
        
        return history
    
    def predict(self, test_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions on test data."""
        self.model.eval()
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Predicting"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                probs = self.model.predict(input_ids, attention_mask)
                preds = torch.argmax(probs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        return np.array(all_preds), np.array(all_probs) 