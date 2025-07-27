# TruthCheck: Fake News Detection with Fine-Tuned BERT

![MIT License](https://img.shields.io/badge/License-MIT-green.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Framework](https://img.shields.io/badge/Framework-Streamlit-orange)
![Model](https://img.shields.io/badge/Model-BERT--BiLSTM--Attention-ff69b4)
![Status](https://img.shields.io/badge/Status-Production-brightgreen)
![Last Commit](https://img.shields.io/github/last-commit/adnaan-tariq/fake-news-detection)
![Repo Size](https://img.shields.io/github/repo-size/adnaan-tariq/fake-news-detection)
![Open Issues](https://img.shields.io/github/issues/adnaan-tariq/fake-news-detection)
![Pull Requests](https://img.shields.io/github/issues-pr/adnaan-tariq/fake-news-detection)
![Forks](https://img.shields.io/github/forks/adnaan-tariq/fake-news-detection?style=social)
![Stars](https://img.shields.io/github/stars/adnaan-tariq/fake-news-detection?style=social)
![Contributors](https://img.shields.io/github/contributors/adnaan-tariq/fake-news-detection)
[![Live Demo](https://img.shields.io/badge/🧪%20Try%20on-HuggingFace-blue?logo=huggingface)](https://huggingface.co/spaces/adnaan05/TruthCheck)

## 🚀 Features
- 🤖 **Hybrid Model:** BERT + BiLSTM + Attention
- 🔄 **Full Fine-Tuning:** All layers of BERT and additional layers
- 🧹 **Robust Preprocessing:** Tokenization, lemmatization, cleaning
- 🧪 **Real-time Prediction App:** Built with Streamlit
- 🌍 **Live Demo on Hugging Face Spaces**
- 🚀 **Plug-and-Play Deployment:** Research-ready, production-capable

## 🌐 Live Demo

👉 [**Click to Launch TruthCheck**](https://huggingface.co/spaces/adnaan05/TruthCheck)

- **Platform:** Hugging Face Spaces  
- **Framework:** Streamlit  
- **Status:** ✅ Live  
- **Features:** Real-time fake news detection with confidence scores

## 🧠 Model Details
- **Base Model:** [BERT-base-uncased](https://huggingface.co/bert-base-uncased)
- **Architecture:**
  - 🧠 BERT Encoder (pre-trained, fine-tuned)
  - 🔁 BiLSTM for sequence representation
  - 👁️ Attention for interpretability
  - 🧮 Fully connected classifier
- **Training:**
  - BERT weights are unfrozen and trained
  - Additional layers initialized and trained from scratch

## 📂 Project Structure
```
.
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── saved/
│   └── checkpoints/
├── app.py
├── src/
│   ├── app.py
│   ├── config/
│   ├── data/
│   ├── models/
│   ├── visualization/
│   └── train.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 📥 Datasets Used
- 🗂️ [Kaggle Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- 🗃️ [LIAR Dataset (UCSB)](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip)

## 📦 Download Data and Model

- **Datasets:** [📁 Google Drive](https://drive.google.com/drive/folders/1tAhWhhhDes5uCdcnMLmJdFBSGWFFl55M?usp=sharing)
- **Trained Models:** [📁 Google Drive](https://drive.google.com/drive/folders/1VEFa0y_vW6AzT5x0fRwmX8shoBhUGd7K?usp=sharing)

### Instructions:
- Place raw/processed data under `data/`
- Place final model file (e.g. `final_model.pt`) in `models/saved/`

## ⚙️ Setup & Installation

### 🔧 Step 1: Clone the Repository
```bash
git clone https://github.com/adnaan-tariq/fake-news-detection.git
cd fake-news-detection
```

### 🧱 Step 2: Create & Activate a Virtual Environment

#### 💻 On Windows
```bash
python -m venv venv
.\venv\Scripts\activate
```

#### 🍎 On macOS/Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

### 📦 Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 🏃‍♂️ Usage

### 🧠 Train the Model
```bash
python -m src.train
```

### 🌐 Run the Streamlit App
```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

## 📊 Results
- **Validation Accuracy:** ~93%
- **F1 Score:** ~0.93  
📈 See `src/visualization/` for training logs and plots.

## 🤝 Contributing

We 💖 contributions from the open-source community!

### How to Contribute:
1. Fork this repository
2. Create a new branch:  
   `git checkout -b feature-name`
3. Make your changes and commit:
   ```bash
   git add .
   git commit -m "Add feature: description"
   ```
4. Push and open a pull request:
   ```bash
   git push origin feature-name
   ```

### You can:
- Fix bugs 🐞
- Improve documentation 📚
- Add new features 🌟
- Optimize model or code 🧠
- Improve UI/UX or frontend 🎨

> 🙌 All contributions will be **acknowledged** and **appreciated**!

## 💡 Open Source & Reproducibility

- ✅ Fully open-sourced under MIT License
- ✅ Training, evaluation, and app code included
- ✅ Datasets referenced externally
- ✅ Reproducible results with provided files
- ✅ Built for real-world use and hackathons

## 📄 License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

## 📬 Contact

For feedback, questions, or collaborations:
- 📧 [Muhammad Adnan Tariq](mailto:adnantariq966@gmail.com)
- 📧 [Muhammad Khaqan Nasir](mailto:khaqannasir01@gmail.com)

## 💖 Hackathon Submission

Made with ❤️ by **Muhammad Adnan Tariq** and **Muhammad Khaqan Nasir**  
🎯 Submitted to: **_Build Real ML Web Apps: No Wrappers, Just Real Models_**  
🏢 Hosted by: [Devpost](https://devpost.com)  
🧠 Powered by: **BERT + BiLSTM + Attention + Streamlit**  
🚀 Built with a passion for **truth**, **tech**, and **open-source**.
