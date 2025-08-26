# app.py
import os
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import base64
from io import BytesIO
from u2net import U2NET
from test import convert_image, save_output

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes (necessary for Flutter Web)

# Allowed file extensions
ALLOWED_EXTS = {'jpg', 'jpeg', 'png'}

def check_allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTS

# Load U2NET model
MODEL_PATH = os.path.join("weights", "u2net.pth")  
net = U2NET(3, 1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
net.to(device)
net.eval()

def img_mask(pillow_image):
    torch_image = convert_image(pillow_image).to(device)
    with torch.no_grad():
        d1, *_ = net(torch_image)
        mask = d1[:, 0, :, :]
    u2bg_img = save_output(pillow_image, mask)
    return u2bg_img

# API route
@app.route('/remove_bg', methods=['POST'])
def remove_bg():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not check_allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400

    img = Image.open(file.stream).convert("RGBA")
    bg_removed = img_mask(img)

    # Convert to base64
    buffered = BytesIO()
    bg_removed.save(buffered, format="PNG")
    encoded_bg_removed = base64.b64encode(buffered.getvalue()).decode()

    return jsonify({'bg_removed': encoded_bg_removed})

# Health check route
@app.route('/')
def home():
    return "U2NET Background Removal API is running!"

# Entry point
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 7860))
    app.run(host='0.0.0.0', port=port, debug=True)
