# FINAL FIX — Duplicates smart tarike se hatao
import re, os, pickle
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, classification_report, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer

for r in ['punkt','stopwords','wordnet','omw-1.4','punkt_tab']:
    nltk.download(r, quiet=True)

# ── STEP 1: Load ───────────────────────────────────────────────
df = pd.read_csv('dataset (1).csv')
print(f"✅ Loaded: {df.shape[0]} rows")

# ── STEP 2: Fix Format ─────────────────────────────────────────
symptom_cols = [c for c in df.columns if c.startswith('Symptom_')]

def row_to_symptoms(row):
    parts = []
    for col in symptom_cols:
        val = row[col]
        if pd.notna(val) and str(val).strip() != '':
            parts.append(str(val).strip().replace('_',' '))
    return ' '.join(parts)

df['symptoms'] = df.apply(row_to_symptoms, axis=1)
df['disease']  = df['Disease'].str.strip()
df = df[df['symptoms'].str.strip() != ''].reset_index(drop=True)

# ── STEP 3: Smart Deduplication ────────────────────────────────
# Pehle sorted symptoms banao (order matter nahi karta)
df['symptoms_sorted'] = df['symptoms'].apply(
    lambda x: ' '.join(sorted(x.split()))
)

before = len(df)
# Har disease ke liye max 20 samples rakho
df = (df.groupby('disease')
       .apply(lambda g: g.drop_duplicates(subset='symptoms_sorted').head(20))
       .reset_index(drop=True))

print(f"✅ Smart dedup: {before} → {len(df)} samples")
print(f"   Diseases: {df['disease'].nunique()}")
print(f"   Per disease: {df.groupby('disease').size().describe()}")

# ── STEP 4: Preprocess ─────────────────────────────────────────
KEEP = {'pain','fever','ache','swelling','rash','cough','nausea',
        'vomiting','fatigue','weakness','loss','high','severe',
        'burning','itching','bleeding','difficulty','shortness'}

lemmatizer = WordNetLemmatizer()
try:
    stop_words = set(stopwords.words('english')) - KEEP
except:
    stop_words = set()

def preprocess(text):
    text = text.lower().strip()
    text = re.sub(r'[^a-zA-Z\s]',' ',text)
    text = re.sub(r'\s+',' ',text)
    try:    tokens = word_tokenize(text)
    except: tokens = text.split()
    tokens = [t for t in tokens if len(t) > 2]
    tokens = [t for t in tokens if t not in stop_words or t in KEEP]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

print("\n⚙️  Preprocessing...")
df['processed'] = df['symptoms'].apply(preprocess)
print(f"✅ Done!")

# ── STEP 5: BERT Embeddings ────────────────────────────────────
print(f"\n🤖 BERT encoding {len(df)} samples...")
bert = SentenceTransformer('all-MiniLM-L6-v2')
X = bert.encode(df['processed'].tolist(),
                show_progress_bar=True, batch_size=64)
print(f"✅ Shape: {X.shape}")

# ── STEP 6: Labels ─────────────────────────────────────────────
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['disease'])
print(f"✅ {len(label_encoder.classes_)} diseases")

# ── STEP 7: Split ──────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✅ Train: {len(X_train)} | Test: {len(X_test)}")

# ── STEP 8: Train ──────────────────────────────────────────────
print("\n🏋️  Training...")
clf = LogisticRegression(
    max_iter=1000,
    C=1.0,           # C=0.1 se accuracy gir gayi thi, wapis 1.0
    solver='lbfgs',
    multi_class='multinomial',
    random_state=42
)
clf.fit(X_train, y_train)
print("✅ Done!")

# ── STEP 9: Cross Validation ───────────────────────────────────
print("\n⚙️  5-Fold Cross Validation...")
cv_scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
print(f"✅ CV Each Fold : {[f'{s*100:.1f}%' for s in cv_scores]}")
print(f"   CV Mean      : {cv_scores.mean()*100:.2f}%")
print(f"   CV Std       : ±{cv_scores.std()*100:.2f}%")

# ── STEP 10: Evaluate ──────────────────────────────────────────
y_pred    = clf.predict(X_test)
acc       = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall    = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1        = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print("\n" + "="*50)
print(f"  ✅ Accuracy  : {acc*100:.2f}%")
print(f"  ✅ Precision : {precision*100:.2f}%")
print(f"  ✅ Recall    : {recall*100:.2f}%")
print(f"  ✅ F1 Score  : {f1:.4f}")
print(f"  ✅ CV Mean   : {cv_scores.mean()*100:.2f}%")
print("="*50)

present_labels = sorted(set(y_test))
present_names  = label_encoder.classes_[present_labels]
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred,
      labels=present_labels,
      target_names=present_names,
      zero_division=0))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=present_labels)
plt.figure(figsize=(18,14))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=present_names,
            yticklabels=present_names,
            cmap='Blues', linewidths=0.5)
plt.title('Confusion Matrix — Disease Prediction', fontsize=14)
plt.ylabel('Actual'); plt.xlabel('Predicted')
plt.xticks(rotation=45, ha='right', fontsize=7)
plt.yticks(fontsize=7)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# ── STEP 11: Save + Download ───────────────────────────────────
os.makedirs('saved_model', exist_ok=True)
with open('saved_model/classifier.pkl','wb') as f: pickle.dump(clf, f)
with open('saved_model/label_encoder.pkl','wb') as f: pickle.dump(label_encoder, f)
with open('saved_model/config.pkl','wb') as f:
    pickle.dump({
        'use_bert': True,
        'bert_model_name': 'all-MiniLM-L6-v2',
        'classes': label_encoder.classes_.tolist()
    }, f)

import shutil
from google.colab import files
shutil.make_archive('saved_model','zip','.','saved_model')
files.download('saved_model.zip')
