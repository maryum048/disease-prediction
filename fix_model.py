"""
fix_model.py
============
Sirf ek baar chalao — preprocessor.pkl ban jayegi!
Run: python fix_model.py
"""

import os
import sys
import pickle
import nltk

print("=" * 45)
print("  MediSense — preprocessor.pkl Fix")
print("=" * 45)

# NLTK downloads
for r in ['punkt', 'stopwords', 'wordnet', 'omw-1.4', 'punkt_tab']:
    nltk.download(r, quiet=True)

# ── Yeh important hai: model folder se import karo ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from model.preprocessing import SymptomPreprocessor  # ← correct import

# ── Object banao aur save karo ─────────────────────
save_path = os.path.join('model', 'saved_model', 'preprocessor.pkl')
print(f"\n📁 Saving to: {save_path}")

preprocessor_obj = SymptomPreprocessor()

with open(save_path, 'wb') as f:
    pickle.dump(preprocessor_obj, f)

print(f"✅ preprocessor.pkl ban gayi!\n")

# ── Final check ─────────────────────────────────────
folder = os.path.join('model', 'saved_model')
required = ['classifier.pkl', 'label_encoder.pkl', 'config.pkl', 'preprocessor.pkl']

print("📋 Final check:")
all_ok = True
for fname in required:
    fpath = os.path.join(folder, fname)
    if os.path.exists(fpath):
        print(f"   ✅ {fname}")
    else:
        print(f"   ❌ {fname} — MISSING!")
        all_ok = False

if all_ok:
    print("\n🎉 Sab theek hai! Ab chalao: python app.py")
else:
    print("\n⚠️  Kuch files missing hain!")