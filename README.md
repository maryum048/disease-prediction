# disease-prediction
AI-powered disease prediction system using machine learning to predict multiple diseases based on symptoms and medical data
# 🏥 MediSense AI — Disease Prediction Web App

> **AI-powered symptom analysis using BERT embeddings + Flask**

![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=flat-square&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-black?style=flat-square&logo=flask)
![BERT](https://img.shields.io/badge/BERT-MiniLM--L6--v2-orange?style=flat-square)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4-f89939?style=flat-square&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📌 Overview

**MediSense AI** is an end-to-end machine learning web application that predicts diseases from natural language symptom descriptions. The user types their symptoms in plain English, and the BERT-powered model returns the most likely diagnosis along with confidence scores.

**⚠️ Disclaimer:** This is an educational project only. Not a substitute for professional medical advice.

---

## ✨ Features

- 🤖 **BERT Embeddings** — Uses `all-MiniLM-L6-v2` for rich semantic text understanding
- 🧹 **Full NLP Pipeline** — Text cleaning → synonym normalization → tokenization → stopword removal → lemmatization
- 📊 **Top-3 Predictions** — Returns ranked predictions with confidence percentages
- 🎯 **50+ Diseases** — Covers a wide range of conditions from Malaria to Diabetes
- 💻 **Professional UI** — Dark-themed, responsive Flask frontend
- ⚡ **REST API** — Clean `/predict` endpoint for easy integration
- ✅ **Input Validation** — Server-side and client-side validation

---

## 🧠 Model Architecture

```
Raw Symptom Text
      │
      ▼
┌─────────────────────┐
│  Text Preprocessing │  ← Clean, normalize, tokenize, lemmatize
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│  BERT Embeddings    │  ← sentence-transformers (all-MiniLM-L6-v2)
│  384-dim vectors    │     (fallback: TF-IDF if BERT unavailable)
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│ Logistic Regression │  ← Multinomial, trained on disease dataset
│    Classifier       │
└─────────────────────┘
      │
      ▼
  Disease + Confidence Score
```

---

## 📁 Project Structure

```
disease-predictor/
│
├── app.py                      # Flask application (main entry)
│
├── model/
│   ├── preprocessing.py        # NLP pipeline (clean, tokenize, lemmatize)
│   ├── train_model.py          # Model training script — run once
│   ├── predictor.py            # Inference / prediction class
│   └── saved_model/            # Auto-generated after training
│       ├── classifier.pkl
│       ├── label_encoder.pkl
│       ├── preprocessor.pkl
│       └── config.pkl
│
├── data/
│   └── disease_dataset.csv     # Training dataset (50+ diseases)
│
├── templates/
│   └── index.html              # Frontend UI
│
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/medisense-ai.git
cd medisense-ai
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Model
```bash
python model/train_model.py
```
Expected output:
```
✅ BERT model loaded
⚙️  Generating BERT embeddings...
✓ Test Accuracy: 91.23%
✓ F1 Score: 0.9105
✅ Training Complete! Model saved.
```

### 5. Run the App
```bash
python app.py
```

Open your browser: **http://localhost:5000**

---

## 🔌 API Reference

### POST `/predict`

Predict disease from symptom description.

**Request:**
```json
{
  "symptoms": "high fever chills body ache sweating headache fatigue"
}
```

**Response:**
```json
{
  "success": true,
  "disease": "Malaria",
  "confidence": 87.34,
  "top_predictions": [
    { "disease": "Malaria",       "confidence": 87.34 },
    { "disease": "Dengue Fever",  "confidence": 7.21  },
    { "disease": "Typhoid Fever", "confidence": 3.10  }
  ],
  "symptoms_detected": ["fever", "chill", "ache", "sweat", "headach", "fatig"]
}
```

### GET `/diseases`
Returns list of all diseases the model can predict.

### GET `/health`
Health check — returns model loading status.

---

## 🖼️ Screenshots

> *(Add screenshots of your app here after running)*

| Input Screen | Results Screen |
|---|---|
| ![input](screenshots/input.png) | ![results](screenshots/results.png) |

---

## 📊 Dataset

The model is trained on a curated dataset of **50+ diseases** with associated symptom descriptions including:

| Category | Examples |
|---|---|
| Infectious | Malaria, Dengue, Typhoid, Tuberculosis, COVID |
| Neurological | Migraine, Meningitis, Encephalitis, Stroke |
| Metabolic | Diabetes Type 1 & 2, Hypothyroidism |
| Respiratory | Pneumonia, Asthma, Bronchitis, Influenza |
| Cardiovascular | Heart Disease, Arrhythmia |
| Gastrointestinal | IBS, Appendicitis, Food Poisoning, Hepatitis |

**To add more data:** Simply add rows to `data/disease_dataset.csv` and retrain.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Backend | Python, Flask |
| ML Model | Scikit-learn (Logistic Regression) |
| Embeddings | BERT (sentence-transformers, MiniLM-L6-v2) |
| NLP | NLTK (tokenization, lemmatization) |
| Frontend | HTML5, CSS3, Vanilla JS |
| Serialization | Pickle |

---

## 🔮 Future Improvements

- [ ] Add more diseases and larger dataset
- [ ] Deploy to Hugging Face Spaces or Render
- [ ] Add patient history tracking
- [ ] Integrate with real medical databases
- [ ] Add multilingual support (Urdu/Hindi)
- [ ] Build a mobile app with Flutter

---

## 👩‍💻 Author

Maryum Afzal
- GitHub: https://github.com/maryum048/disease-prediction/blob/main/README.md?plain=1


---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---


