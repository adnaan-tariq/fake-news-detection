import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from typing import Dict, List, Union
import pandas as pd
import numpy as np

class FakeNewsDataset(Dataset):
    def __init__(self,
                 texts: List[str],
                 labels: List[int],
                 tokenizer: BertTokenizer,
                 max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def create_data_loaders(
    df: pd.DataFrame,
    text_column: str,
    label_column: str,
    tokenizer: BertTokenizer,
    batch_size: int = 32,
    max_length: int = 512,
    train_size: float = 0.8,
    val_size: float = 0.1,
    random_state: int = 42
) -> Dict[str, torch.utils.data.DataLoader]:
    """Create train, validation, and test data loaders."""
    # Split data
    train_df = df.sample(frac=train_size, random_state=random_state)
    remaining_df = df.drop(train_df.index)
    val_df = remaining_df.sample(frac=val_size/(1-train_size), random_state=random_state)
    test_df = remaining_df.drop(val_df.index)
    
    # Create datasets
    train_dataset = FakeNewsDataset(
        texts=train_df[text_column].tolist(),
        labels=train_df[label_column].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    val_dataset = FakeNewsDataset(
        texts=val_df[text_column].tolist(),
        labels=val_df[label_column].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    test_dataset = FakeNewsDataset(
        texts=test_df[text_column].tolist(),
        labels=test_df[label_column].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    } 