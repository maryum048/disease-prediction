"""
Flask Application — Disease Prediction Web App
===============================================
Run:
    python app.py
Then open: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify
import os
import sys

# Ensure proper import path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

app = Flask(__name__)

# ─── Lazy Load Predictor ─────────────────────────────────────────────
predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        try:
            from model.predictor import DiseasePredictor
            predictor = DiseasePredictor()
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            predictor = None
    return predictor


# ═════════════════════════════════════════════════════════════════════
# ROUTES
# ═════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if not data or 'symptoms' not in data:
            return jsonify({'error': 'No symptoms provided'}), 400

        symptoms_text = data['symptoms'].strip()

        # Validation
        if len(symptoms_text) < 5:
            return jsonify({
                'error': 'Please enter more detailed symptoms.'
            }), 400

        if len(symptoms_text) > 1000:
            return jsonify({
                'error': 'Input too long (max 1000 chars).'
            }), 400

        # Load model
        pred = get_predictor()
        if pred is None:
            return jsonify({
                'error': 'Model not loaded. Train first.'
            }), 503

        # Prediction — seedha string bhejo, list nahi ✅
        result = pred.predict(symptoms_text)

        if not isinstance(result, dict):
            return jsonify({'error': 'Invalid model response'}), 500

        return jsonify({
            'success': True,
            'disease': result.get('disease', 'Unknown'),
            'confidence': result.get('confidence', 0),
            'top_predictions': result.get('top_predictions', []),
            'symptoms_detected': result.get('symptoms_detected', [])
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/diseases', methods=['GET'])
def get_diseases():
    pred = get_predictor()
    if pred is None:
        return jsonify({'error': 'Model not loaded'}), 503

    try:
        diseases = pred.get_all_diseases()
        return jsonify({'diseases': diseases})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    pred = get_predictor()
    return jsonify({
        'status': 'healthy',
        'model_loaded': pred is not None
    })


# ═════════════════════════════════════════════════════════════════════
# RUN
# ═════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n🏥 Disease Prediction Web App")
    print("=" * 40)
    print("   Starting Flask server...")
    print("   Open: http://localhost:5000")
    print("=" * 40 + "\n")

    # Pre-load model
    get_predictor()

    app.run(debug=True, host='0.0.0.0', port=5000)