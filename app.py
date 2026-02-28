import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
import os

app = Flask(__name__)

# Load Model
model_path = os.path.join('model', 'cardio_model.pkl')
try:
    model = joblib.load(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Warning: Model not found or corrupt.")
    model = None

# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/visuals')
def visuals():
    # Visuals are now pre-generated static images for performance
    return render_template('visuals.html')

@app.route('/model')
def model_info():
    return render_template('model_info.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')
    
    # API Logic for POST (AJAX)
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.json
        features = [
            float(data.get('age')),
            int(data.get('gender')),
            float(data.get('height')),
            float(data.get('weight')),
            float(data.get('ap_hi')),
            float(data.get('ap_lo')),
            int(data.get('cholesterol')),
            int(data.get('gluc')),
            int(data.get('smoke')),
            int(data.get('alco')),
            int(data.get('active'))
        ]
        
        final_features = [np.array(features)]
        prediction = model.predict(final_features)
        
        # Calculate probability if possible
        try:
            proba = model.predict_proba(final_features)[0][1]
            proba = round(proba * 100, 1)
        except:
            proba = 0

        output = int(prediction[0])
        result_text = "Risk Detected" if output == 1 else "Normal"
        
        return jsonify({
            'prediction': output,
            'prediction_text': result_text,
            'probability': proba
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
