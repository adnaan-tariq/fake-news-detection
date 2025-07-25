import re
import emoji
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from typing import List, Union
import pandas as pd

class TextPreprocessor:
    def __init__(self):
        # Download required NLTK data
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def remove_urls(self, text: str) -> str:
        """Remove URLs from text."""
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub('', text)
    
    def remove_emojis(self, text: str) -> str:
        """Remove emojis from text."""
        return emoji.replace_emoji(text, replace='')
    
    def remove_special_chars(self, text: str) -> str:
        """Remove special characters and numbers."""
        return re.sub(r'[^a-zA-Z\s]', '', text)
    
    def remove_extra_spaces(self, text: str) -> str:
        """Remove extra spaces."""
        return re.sub(r'\s+', ' ', text).strip()
    
    def lemmatize_text(self, text: str) -> str:
        """Lemmatize text."""
        # Simple word tokenization using split
        tokens = text.split()
        return ' '.join([self.lemmatizer.lemmatize(token) for token in tokens])
    
    def remove_stopwords(self, text: str) -> str:
        """Remove stopwords from text."""
        # Simple word tokenization using split
        tokens = text.split()
        return ' '.join([token for token in tokens if token.lower() not in self.stop_words])
    
    def correct_spelling(self, text: str) -> str:
        """Correct spelling in text."""
        return str(TextBlob(text).correct())
    
    def preprocess_text(self, text: str, 
                       remove_urls: bool = True,
                       remove_emojis: bool = True,
                       remove_special_chars: bool = True,
                       remove_stopwords: bool = True,
                       lemmatize: bool = True,
                       correct_spelling: bool = False) -> str:
        """Apply all preprocessing steps to text."""
        if not isinstance(text, str):
            return ""
            
        text = text.lower()
        
        if remove_urls:
            text = self.remove_urls(text)
        if remove_emojis:
            text = self.remove_emojis(text)
        if remove_special_chars:
            text = self.remove_special_chars(text)
        if remove_stopwords:
            text = self.remove_stopwords(text)
        if lemmatize:
            text = self.lemmatize_text(text)
        if correct_spelling:
            text = self.correct_spelling(text)
            
        text = self.remove_extra_spaces(text)
        return text
    
    def preprocess_dataframe(self, df: pd.DataFrame, 
                           text_column: str,
                           **kwargs) -> pd.DataFrame:
        """Preprocess text column in a dataframe."""
        df = df.copy()
        df[text_column] = df[text_column].apply(
            lambda x: self.preprocess_text(x, **kwargs)
        )
        return df 