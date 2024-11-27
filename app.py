import os
import base64
import pickle
import face_recognition
from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
import numpy as np

app = Flask(__name__)

ENCODINGS_FILE = "encodings.pkl"
FACES_DIR = "faces"

os.makedirs(FACES_DIR, exist_ok=True)

def load_encodings():
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def save_encodings(encodings):
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(encodings, f)

def process_image(image_data):
    try:
        header, encoded = image_data.split(",", 1)
        image_data = base64.b64decode(encoded)
        image = Image.open(BytesIO(image_data))
        image_array = np.array(image)
        return image_array
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

user_encodings = load_encodings()

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    name = data.get('name')
    image_data = data.get('image')
    if not name or not image_data:
        return jsonify({'error': 'Name and image are required'}), 400

    image = process_image(image_data)
    if image is None:
        return jsonify({'error': 'Invalid image'}), 400

    face_encodings = face_recognition.face_encodings(image)
    if not face_encodings:
        return jsonify({'error': 'No face found'}), 400

    user_encodings[name] = face_encodings[0]
    save_encodings(user_encodings)
    
    image_path = os.path.join(FACES_DIR, f"{name}.jpg")
    Image.fromarray(image).save(image_path)

    return jsonify({'message': f'{name} registered'})

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    image_data = data.get('image')
    if not image_data:
        return jsonify({'error': 'Image is required'}), 400

    image = process_image(image_data)
    if image is None:
        return jsonify({'error': 'Invalid image'}), 400

    face_encodings = face_recognition.face_encodings(image)
    if not face_encodings:
        return jsonify({'error': 'No face found'}), 400

    for name, encoding in user_encodings.items():
        results = face_recognition.compare_faces([encoding], face_encodings[0], tolerance=0.6)
        if results[0]:
            return jsonify({'message': f'Welcome, {name}!'})
    return jsonify({'error': 'Face not recognized'}), 401

if __name__ == '__main__':
    app.run(debug=True)