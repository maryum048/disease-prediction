"""
Text Preprocessing & Feature Extraction Module
==============================================
Handles all NLP preprocessing for symptom text input.
"""

import re
import string
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK data
def download_nltk_data():
    resources = ['punkt', 'stopwords', 'wordnet', 'omw-1.4', 'punkt_tab']
    for r in resources:
        try:
            nltk.download(r, quiet=True)
        except:
            pass

download_nltk_data()

# Medical stopwords to KEEP (important symptoms)
MEDICAL_KEEP_WORDS = {
    'pain', 'fever', 'ache', 'swelling', 'rash', 'cough', 'nausea',
    'vomiting', 'fatigue', 'weakness', 'loss', 'high', 'severe',
    'chronic', 'acute', 'mild', 'heavy', 'frequent', 'sudden',
    'burning', 'itching', 'bleeding', 'difficulty', 'shortness'
}

# Symptom synonyms normalization
SYMPTOM_SYNONYMS = {
    'temp': 'temperature',
    'temps': 'temperature',
    'stomachache': 'abdominal pain',
    'tummy': 'abdominal',
    'puking': 'vomiting',
    'throwing up': 'vomiting',
    'runny nose': 'rhinorrhea',
    'can\'t breathe': 'shortness of breath',
    'hard to breathe': 'shortness of breath',
    'tired': 'fatigue',
    'exhausted': 'fatigue',
    'dizzy': 'dizziness',
    'head hurts': 'headache',
    'eyes hurt': 'eye pain',
    'chest hurts': 'chest pain',
}


class SymptomPreprocessor:
    """
    Complete NLP pipeline for symptom text preprocessing.
    Steps: Clean → Normalize → Tokenize → Remove Stopwords → Lemmatize
    """

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words('english')) - MEDICAL_KEEP_WORDS
        except:
            self.stop_words = set()

    def clean_text(self, text: str) -> str:
        """Step 1: Basic text cleaning"""
        text = text.lower().strip()
        # Remove special characters but keep spaces and letters
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        return text

    def normalize_synonyms(self, text: str) -> str:
        """Step 2: Replace common synonyms with standard medical terms"""
        for synonym, standard in SYMPTOM_SYNONYMS.items():
            text = text.replace(synonym, standard)
        return text

    def tokenize(self, text: str) -> list:
        """Step 3: Tokenize into words"""
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()
        return tokens

    def remove_stopwords(self, tokens: list) -> list:
        """Step 4: Remove stopwords but keep medical terms"""
        return [t for t in tokens if t not in self.stop_words or t in MEDICAL_KEEP_WORDS]

    def lemmatize(self, tokens: list) -> list:
        """Step 5: Lemmatize to root forms"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def preprocess(self, text: str) -> str:
        """
        Full pipeline: raw text → clean preprocessed string
        Returns processed text ready for vectorization
        """
        text = self.clean_text(text)
        text = self.normalize_synonyms(text)
        tokens = self.tokenize(text)
        tokens = [t for t in tokens if len(t) > 2]  # Remove very short tokens
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize(tokens)
        return ' '.join(tokens)

    def extract_symptom_keywords(self, text: str) -> list:
        """Extract and return list of identified symptom keywords"""
        processed = self.preprocess(text)
        return processed.split()


class FeatureExtractor:
    """
    TF-IDF based feature extraction for symptom text.
    Fitted on training data, transforms input for model prediction.
    """

    def __init__(self, max_features=5000, ngram_range=(1, 2)):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,      # Unigrams + bigrams
            analyzer='word',
            sublinear_tf=True,            # Apply log normalization
            min_df=1,
            max_df=0.95
        )
        self.preprocessor = SymptomPreprocessor()
        self.is_fitted = False

    def fit_transform(self, texts: list) -> np.ndarray:
        """Fit on training data and transform"""
        processed = [self.preprocessor.preprocess(t) for t in texts]
        features = self.vectorizer.fit_transform(processed)
        self.is_fitted = True
        return features

    def transform(self, texts: list) -> np.ndarray:
        """Transform new input using fitted vectorizer"""
        if not self.is_fitted:
            raise ValueError("FeatureExtractor not fitted yet. Call fit_transform first.")
        processed = [self.preprocessor.preprocess(t) for t in texts]
        return self.vectorizer.transform(processed)

    def get_feature_names(self) -> list:
        """Return feature names from vectorizer"""
        return self.vectorizer.get_feature_names_out().tolist()