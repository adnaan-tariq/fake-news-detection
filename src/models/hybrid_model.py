import torch
import torch.nn as nn
from transformers import BertModel
from typing import Tuple, Dict

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attention_weights = torch.softmax(self.attention(x), dim=1)
        attended = torch.sum(attention_weights * x, dim=1)
        return attended, attention_weights

class HybridFakeNewsDetector(nn.Module):
    def __init__(self,
                 bert_model_name: str = "bert-base-uncased",
                 lstm_hidden_size: int = 256,
                 lstm_num_layers: int = 2,
                 dropout_rate: float = 0.3,
                 num_classes: int = 2):
        super().__init__()
        
        # BERT encoder
        self.bert = BertModel.from_pretrained(bert_model_name)
        bert_hidden_size = self.bert.config.hidden_size
        
        # BiLSTM layer
        self.lstm = nn.LSTM(
            input_size=bert_hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention layer
        self.attention = AttentionLayer(lstm_hidden_size * 2)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_hidden_size * 2, lstm_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_hidden_size, num_classes)
        )
        
    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Get BERT embeddings
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        bert_embeddings = bert_outputs.last_hidden_state
        
        # Process through BiLSTM
        lstm_output, _ = self.lstm(bert_embeddings)
        
        # Apply attention
        attended, attention_weights = self.attention(lstm_output)
        
        # Classification
        logits = self.classifier(attended)
        
        return {
            'logits': logits,
            'attention_weights': attention_weights
        }
    
    def predict(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        """Get model predictions."""
        outputs = self.forward(input_ids, attention_mask)
        return torch.softmax(outputs['logits'], dim=1)
    
    def get_attention_weights(self, input_ids: torch.Tensor,
                            attention_mask: torch.Tensor) -> torch.Tensor:
        """Get attention weights for interpretability."""
        outputs = self.forward(input_ids, attention_mask)
        return outputs['attention_weights'] 