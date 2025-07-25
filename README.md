# TruthCheck: Fake News Detection with Fine-Tuned BERT

TruthCheck is an **open source** fake news detection system leveraging a hybrid deep learning architecture. It combines a pre-trained **BERT**-base-uncased model with a BiLSTM and attention mechanism, **fully fine-tuned** on a curated **_dataset of real and fake news_**. The project includes robust preprocessing, feature extraction, model training, evaluation, and a Streamlit web app for interactive predictions.

---

## 🚀 Features
- **Hybrid Model:** BERT-base-uncased + BiLSTM + Attention
- **Full Fine-Tuning:** All layers of BERT and additional layers are trainable and optimized on the fake news dataset
- **Comprehensive Preprocessing:** Cleaning, tokenization, lemmatization, and more
- **Training & Evaluation:** Scripts for training, validation, and test evaluation
- **Interactive App:** Streamlit web app for real-time news classification
- **Live Demo:** Deployed on Hugging Face Spaces for immediate testing
- **Ready for Deployment:** Easily extendable for research or production

## 🌐 Live Demo

**Try TruthCheck now:** [https://huggingface.co/spaces/adnaan05/TruthCheck](https://huggingface.co/spaces/adnaan05/TruthCheck)

- **Platform:** Hugging Face Spaces
- **Framework:** Streamlit
- **Status:** Live and accessible
- **Features:** Real-time fake news detection with confidence scores


---

## 🧠 Model Details
- **Base Model:** [BERT-base-uncased](https://huggingface.co/bert-base-uncased)
- **Architecture:**
  - BERT encoder (pre-trained, all layers fine-tuned)
  - BiLSTM layer for sequential context
  - Attention mechanism for interpretability
  - Fully connected classification head
- **Fine-Tuning Technique:**
  - All BERT layers are unfrozen and updated during training (full fine-tuning)
  - Additional layers (BiLSTM, attention, classifier) are trained from scratch

---

## 📂 Project Structure
```
.
├── data/
│   ├── raw/           # Raw datasets
│   └── processed/     # Processed data
├── models/
│   ├── saved/        # Saved model checkpoints
│   └── checkpoints/  # Training checkpoints
├── app.py                  # Streamlit app entry point
├── src/
│   ├── app.py              # Main app logic
│   ├── config/             # Configuration files
│   ├── data/               # Data processing code (not datasets)
│   ├── models/             # Model architecture and training code
│   ├── visualization/      # Plotting and visualization scripts
│   └── train.py            # Model training script
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── .gitignore
```

---

## 📥 Datasets Used (with References)

- **Kaggle Fake and Real News Dataset:**  
  [Kaggle Dataset Link](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- **LIAR Dataset:**  
  [https://www.cs.ucsb.edu/~william/data/liar_dataset.zip](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip)

---

## 📥 Download Data and Model

**Raw and Processed Datasets:**  
[Google Drive Link](https://drive.google.com/drive/folders/1tAhWhhhDes5uCdcnMLmJdFBSGWFFl55M?usp=sharing)

**Trained Model(s):**  
[Google Drive Link](https://drive.google.com/drive/folders/1VEFa0y_vW6AzT5x0fRwmX8shoBhUGd7K?usp=sharing)

### **Instructions:**
1. Download the datasets and place them in the `data/` directory:
    - `data/raw/` for raw files
    - `data/processed/` for processed files
2. Download the trained model (e.g., `final_model.pt` or `best_model.pt`) and place it in `models/saved/`.

---

## ⚙️ Setup & Deployment Instructions

1. **Clone the repository:**
    ```bash
    git clone https://github.com/adnaan-tariq/fake-news-detection.git
    cd fake-news-detection
    ```
2. **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```
3. **Install dependencies:**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

---

## 🏃‍♂️ Usage

### **Train the Model**
If you want to train from scratch (after placing the data as described above):
```bash
python -m src.train
```

### **Run the Streamlit App**
```bash
streamlit run app.py
```
- Open [http://localhost:8501](http://localhost:8501) in your browser.

### **Test the Model**
- The app and scripts will use the model in `models/saved/final_model.pt` by default.
- For custom inference, see the example in `src/app.py` or ask for a sample script.

---

## 📊 Results
- **Validation Accuracy:** ~93%
- **Validation F1 Score:** ~0.93
- (See training logs and visualizations for more details.)

---

## 📦 Open Source & Reproducibility Policy
- **All code is original or properly credited.**
- **All code for the ML model training process is included and open source.**
- **References to all datasets used are provided above.**
- **Source code for the full web application is included.**
- **Setup and deployment instructions are provided in this README.**
- **This project is fully open-sourced and reproducible, in line with hackathon requirements.**

---

## 📦 Data & Model Policy
- **Data and model files are NOT included in this repository.**
- Please download them from the provided Google Drive links above.

---

## 🤝 Contributing
Pull requests and suggestions are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License
This project is licensed under the MIT License.

---

## 🙋‍♂️ Contact
For questions or support, contact [Adnan Tariq](mailto:adnantariq966@gmail.com). 
