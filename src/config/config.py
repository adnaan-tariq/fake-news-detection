from pathlib import Path
import torch

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
SAVED_MODELS_DIR = MODEL_DIR / "saved"
CHECKPOINTS_DIR = MODEL_DIR / "checkpoints"

# Data parameters
MAX_SEQUENCE_LENGTH = 256
VOCAB_SIZE = 15000
EMBEDDING_DIM = 128
BATCH_SIZE = 8
TEST_SIZE = 0.2
VAL_SIZE = 0.1
RANDOM_STATE = 42
MAX_SAMPLES = 10000

# Model parameters
BERT_MODEL_NAME = "bert-base-uncased"
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 1
DROPOUT_RATE = 0.3
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
EARLY_STOPPING_PATIENCE = 2

# Training parameters

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0
PIN_MEMORY = False

# Feature extraction
USE_TFIDF = True
USE_BERT = True
USE_LSTM = True

# Evaluation metrics
METRICS = ["accuracy", "precision", "recall", "f1"] 