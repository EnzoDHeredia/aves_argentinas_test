from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
import os
from bird_classifier import BirdClassifier

MODEL_PATH = os.path.join('..', 'model', 'modelo_102_B1.pth')
THRESHOLD = 0.5
IMG_SIZE = 224

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

model = BirdClassifier(DEVICE, 102, MODEL_PATH)
app = Flask(__name__)
CORS(app)

from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    try:
        file = request.files['file']
        img = Image.open(file.stream).convert('RGB')
        tensor = transform(img).unsqueeze(0).to(DEVICE)
        idx, confidence = model.predict(tensor)
        if confidence < THRESHOLD:
            return jsonify({'class': None, 'confidence': confidence})
        return jsonify({'class': idx, 'confidence': confidence})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Vercel ejecuta como serverless, no uses app.run()
