"""
Predictor Module — Disease Prediction Inference
================================================
Loads saved model and runs predictions on new symptom inputs.
"""

import os
import pickle
import numpy as np

from typing import Dict, List, Tuple

# Try BERT
try:
    from sentence_transformers import SentenceTransformer
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.preprocessing import SymptomPreprocessor


class DiseasePredictor:

    def __init__(self, model_dir: str = None):
        if model_dir is None:
            model_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),  # ← fix: sirf ek dirname
                'saved_model'                                # ← fix: 'model' hata diya
            )
        self.model_dir = model_dir
        self.is_loaded = False
        self._load_model()

    def _load_model(self):
        try:
            with open(os.path.join(self.model_dir, 'config.pkl'), 'rb') as f:
                self.config = pickle.load(f)

            with open(os.path.join(self.model_dir, 'classifier.pkl'), 'rb') as f:
                self.classifier = pickle.load(f)

            with open(os.path.join(self.model_dir, 'label_encoder.pkl'), 'rb') as f:
                self.label_encoder = pickle.load(f)

            with open(os.path.join(self.model_dir, 'preprocessor.pkl'), 'rb') as f:
                self.preprocessor = pickle.load(f)

            self.use_bert = self.config.get('use_bert', False)

            if self.use_bert and BERT_AVAILABLE:
                bert_name = self.config.get('bert_model_name', 'all-MiniLM-L6-v2')
                self.bert_model = SentenceTransformer(bert_name)
            else:
                self.use_bert = False
                tfidf_path = os.path.join(self.model_dir, 'tfidf_extractor.pkl')
                if os.path.exists(tfidf_path):
                    with open(tfidf_path, 'rb') as f:
                        self.tfidf_extractor = pickle.load(f)
                else:
                    raise FileNotFoundError("TF-IDF extractor not found!")

            self.is_loaded = True
            print("✅ Model loaded successfully")

        except FileNotFoundError as e:
            self.is_loaded = False
            raise FileNotFoundError(
                f"Model not found at: {self.model_dir}\n"
                "Please run: python model/train_model.py"
            )

    def _get_features(self, text: str) -> np.ndarray:
        processed = self.preprocessor.preprocess(text)

        if self.use_bert and BERT_AVAILABLE:
            return self.bert_model.encode([processed])
        else:
            features = self.tfidf_extractor.transform([processed])
            return features.toarray()

    def predict(self, symptoms_text: str) -> dict:

        if not self.is_loaded:
            return {
                'error': 'Model not loaded',
                'disease': None,
                'confidence': 0,
                'top_predictions': [],
                'symptoms_detected': []
            }

        if not symptoms_text or len(symptoms_text.strip()) < 3:
            return {
                'error': 'Please describe your symptoms in more detail.',
                'disease': None,
                'confidence': 0,
                'top_predictions': [],
                'symptoms_detected': []
            }

        try:
            features = self._get_features(symptoms_text)

            proba = self.classifier.predict_proba(features)[0]
            prediction_idx = np.argmax(proba)

            top3_indices = np.argsort(proba)[::-1][:3]
            top_predictions = [
                {
                    'disease': self.label_encoder.classes_[i],
                    'confidence': round(float(proba[i]) * 100, 2)
                }
                for i in top3_indices
            ]

            symptoms_detected = self.preprocessor.extract_symptom_keywords(symptoms_text)

            return {
                'disease': self.label_encoder.classes_[prediction_idx],
                'confidence': round(float(proba[prediction_idx]) * 100, 2),
                'top_predictions': top_predictions,
                'symptoms_detected': symptoms_detected[:8],
                'error': None
            }

        except Exception as e:
            return {
                'error': str(e),
                'disease': None,
                'confidence': 0,
                'top_predictions': [],
                'symptoms_detected': []
            }

    def get_all_diseases(self) -> list:
        return self.label_encoder.classes_.tolist()