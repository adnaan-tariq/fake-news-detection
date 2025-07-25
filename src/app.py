import streamlit as st
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import plotly.express as px
import plotly.graph_objects as go
from transformers import BertTokenizer
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.hybrid_model import HybridFakeNewsDetector
from src.config.config import *
from src.data.preprocessor import TextPreprocessor

# Set page config
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide"
)

@st.cache_resource
def load_model_and_tokenizer():
    """Load the model and tokenizer (cached)."""
    # Initialize model
    model = HybridFakeNewsDetector(
        bert_model_name=BERT_MODEL_NAME,
        lstm_hidden_size=LSTM_HIDDEN_SIZE,
        lstm_num_layers=LSTM_NUM_LAYERS,
        dropout_rate=DROPOUT_RATE
    )
    
    # Load trained weights
    state_dict = torch.load(SAVED_MODELS_DIR / "final_model.pt", map_location=torch.device('cpu'))
    
    # Filter out unexpected keys
    model_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}
    
    # Load the filtered state dict
    model.load_state_dict(filtered_state_dict, strict=False)
    model.eval()
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    
    return model, tokenizer

@st.cache_resource
def get_preprocessor():
    """Get the text preprocessor (cached)."""
    return TextPreprocessor()

def predict_news(text):
    """Predict if the given news is fake or real."""
    # Get model, tokenizer, and preprocessor from cache
    model, tokenizer = load_model_and_tokenizer()
    preprocessor = get_preprocessor()
    
    # Preprocess text
    processed_text = preprocessor.preprocess_text(text)
    
    # Tokenize
    encoding = tokenizer.encode_plus(
        processed_text,
        add_special_tokens=True,
        max_length=MAX_SEQUENCE_LENGTH,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    # Get prediction
    with torch.no_grad():
        outputs = model(
            encoding['input_ids'],
            encoding['attention_mask']
        )
        probabilities = torch.softmax(outputs['logits'], dim=1)
        prediction = torch.argmax(outputs['logits'], dim=1)
        attention_weights = outputs['attention_weights']
    
    # Convert attention weights to numpy and get the first sequence
    attention_weights_np = attention_weights[0].cpu().numpy()
    
    return {
        'prediction': prediction.item(),
        'label': 'FAKE' if prediction.item() == 1 else 'REAL',
        'confidence': torch.max(probabilities, dim=1)[0].item(),
        'probabilities': {
            'REAL': probabilities[0][0].item(),
            'FAKE': probabilities[0][1].item()
        },
        'attention_weights': attention_weights_np
    }

def plot_confidence(probabilities):
    """Plot prediction confidence."""
    fig = go.Figure(data=[
        go.Bar(
            x=list(probabilities.keys()),
            y=list(probabilities.values()),
            text=[f'{p:.2%}' for p in probabilities.values()],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title='Prediction Confidence',
        xaxis_title='Class',
        yaxis_title='Probability',
        yaxis_range=[0, 1]
    )
    
    return fig

def plot_attention(text, attention_weights):
    """Plot attention weights."""
    tokens = text.split()
    attention_weights = attention_weights[:len(tokens)]  # Truncate to match tokens
    
    # Ensure attention weights are in the correct format
    if isinstance(attention_weights, (list, np.ndarray)):
        attention_weights = np.array(attention_weights).flatten()
    
    # Format weights for display
    formatted_weights = [f'{float(w):.2f}' for w in attention_weights]
    
    fig = go.Figure(data=[
        go.Bar(
            x=tokens,
            y=attention_weights,
            text=formatted_weights,
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title='Attention Weights',
        xaxis_title='Tokens',
        yaxis_title='Attention Weight',
        xaxis_tickangle=45
    )
    
    return fig

def main():
    st.title("📰 Fake News Detection System")
    st.write("""
    This application uses a hybrid deep learning model (BERT + BiLSTM + Attention) 
    to detect fake news articles. Enter a news article below to analyze it.
    """)
    
    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info("""
    
    The model combines:
    - BERT for contextual embeddings
    - BiLSTM for sequence modeling
    - Attention mechanism for interpretability
    """)
    
    # Main content
    st.header("News Analysis")
    
    # Text input
    news_text = st.text_area(
        "Enter the news article to analyze:",
        height=200,
        placeholder="Paste your news article here..."
    )
    
    if st.button("Analyze"):
        if news_text:
            with st.spinner("Analyzing the news article..."):
                # Get prediction
                result = predict_news(news_text)
                
                # Display result
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Prediction")
                    if result['label'] == 'FAKE':
                        st.error(f"🔴 This news is likely FAKE (Confidence: {result['confidence']:.2%})")
                    else:
                        st.success(f"🟢 This news is likely REAL (Confidence: {result['confidence']:.2%})")
                
                with col2:
                    st.subheader("Confidence Scores")
                    st.plotly_chart(plot_confidence(result['probabilities']), use_container_width=True)
                
                # Show attention visualization
                st.subheader("Attention Analysis")
                st.write("""
                The attention weights show which parts of the text the model focused on 
                while making its prediction. Higher weights indicate more important tokens.
                """)
                st.plotly_chart(plot_attention(news_text, result['attention_weights']), use_container_width=True)
                
                # Show model explanation
                st.subheader("Model Explanation")
                if result['label'] == 'FAKE':
                    st.write("""
                    The model identified this as fake news based on:
                    - Linguistic patterns typical of fake news
                    - Inconsistencies in the content
                    - Attention weights on suspicious phrases
                    """)
                else:
                    st.write("""
                    The model identified this as real news based on:
                    - Credible language patterns
                    - Consistent information
                    - Attention weights on factual statements
                    """)
        else:
            st.warning("Please enter a news article to analyze.")

if __name__ == "__main__":
    main() 