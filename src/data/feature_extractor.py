import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from transformers import BertTokenizer, BertModel
from typing import Tuple, Dict, List
import pandas as pd
from tqdm import tqdm

class FeatureExtractor:
    def __init__(self, bert_model_name: str = "bert-base-uncased"):
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert_model = BertModel.from_pretrained(bert_model_name)
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        self.count_vectorizer = CountVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
    def get_bert_embeddings(self, texts: List[str], 
                          batch_size: int = 32,
                          max_length: int = 512) -> np.ndarray:
        """Extract BERT embeddings for a list of texts."""
        self.bert_model.eval()
        embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size)):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize and prepare input
                encoded = self.bert_tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors='pt'
                )
                
                # Get BERT embeddings
                outputs = self.bert_model(**encoded)
                # Use [CLS] token embeddings as sentence representation
                batch_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
                embeddings.append(batch_embeddings)
                
        return np.vstack(embeddings)
    
    def get_tfidf_features(self, texts: List[str]) -> np.ndarray:
        """Extract TF-IDF features from texts."""
        return self.tfidf_vectorizer.fit_transform(texts).toarray()
    
    def get_count_features(self, texts: List[str]) -> np.ndarray:
        """Extract Count Vectorizer features from texts."""
        return self.count_vectorizer.fit_transform(texts).toarray()
    
    def extract_all_features(self, texts: List[str],
                           use_bert: bool = True,
                           use_tfidf: bool = True,
                           use_count: bool = True) -> Dict[str, np.ndarray]:
        """Extract all features from texts."""
        features = {}
        
        if use_bert:
            features['bert'] = self.get_bert_embeddings(texts)
        if use_tfidf:
            features['tfidf'] = self.get_tfidf_features(texts)
        if use_count:
            features['count'] = self.get_count_features(texts)
            
        return features
    
    def extract_features_from_dataframe(self, 
                                      df: pd.DataFrame,
                                      text_column: str,
                                      **kwargs) -> Dict[str, np.ndarray]:
        """Extract features from a dataframe's text column."""
        texts = df[text_column].tolist()
        return self.extract_all_features(texts, **kwargs) 