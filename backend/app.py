from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from flask_cors import CORS
import numpy as np
import cv2
import os

app = Flask(__name__)

CORS(app)
IMG_SIZE = 150
MODEL_PATH = 'model/improved_pneumonia_model.h5'
model = None

@app.route('/')
def home():
    return 'PneumoScan backend is running!'

@app.route('/predict', methods=['POST'])
def predict():
    global model
    if model is None:
        model = load_model(MODEL_PATH)

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    prediction = model.predict(img)[0][0]
    label = 'NORMAL' if prediction > 0.35 else 'PNEUMONIA'
    confidence = prediction if label == 'NORMAL' else 1 - prediction

    return jsonify({
        'label': label,
        'confidence': round(float(confidence) * 100, 2)
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)