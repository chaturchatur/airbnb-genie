from flask import Flask, request, jsonify, render_template
import os
import json
from inference import load_model_and_artifacts, predict_price

app = Flask(__name__)

# global variables for model and artifacts
model = None
artifacts = None

# load model at startup
if os.environ.get('FLASK_ENV') != 'test':
    try:
        model, artifacts = load_model_and_artifacts()
        print("model loaded on startup")
    except Exception as e:
        print(f"error loading model on startup: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global model, artifacts
    
    if not model or not artifacts:
        # try loading again if failed previously
        model, artifacts = load_model_and_artifacts()
        if not model:
            return jsonify({'error': 'model not available'}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'no input data provided'}), 400
            
        price = predict_price(data, model, artifacts)
        
        return jsonify({
            'predicted_price': round(price, 2),
            'currency': 'USD'
        })
        
    except Exception as e:
        print(f"prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/config', methods=['GET'])
def get_config():
    """return pre-extracted config without loading model"""
    try:
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        return jsonify(config)
    except Exception as e:
        print(f"error loading config: {e}")
        return jsonify({'error': 'config not available'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

