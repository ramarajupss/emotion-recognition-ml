import logging
import watchtower
import boto3
import os
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# Configure logging to CloudWatch
REGION_NAME = 'us-east-1'  # Set your AWS region to us-east-1
LOG_GROUP_NAME = 'emotion-recognition-project'
LOG_STREAM_NAME = 'CloudFront_E1XE7A18FML6CK'  # Corrected stream name

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

try:
    boto3_session = boto3.Session(region_name=REGION_NAME)
    handler = watchtower.CloudWatchLogHandler(
        log_group=LOG_GROUP_NAME,
        stream_name=LOG_STREAM_NAME,
        boto3_session=boto3_session
    )
    logger.addHandler(handler)
    logger.info("CloudWatch logging configured successfully.")
except Exception as e:
    print(f"Error configuring CloudWatch logging: {e}")
    logging.basicConfig(level=logging.INFO)
    logger.error("Failed to configure CloudWatch logging.", exc_info=True)

# Load the pre-trained model
model = load_model('emotion_recognition_model.h5')

# List of emotions as per your model
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

@app.route('/')
def index():
    return render_template('index.html')  # This will render index.html from the templates folder

@app.route('/predict', methods=['POST','OPTIONS'])
def predict():
    logger.info("Received /predict request")
    if request.method == 'OPTIONS':
        logger.info("Received OPTIONS request (preflight)")
        return '', 204  # Just return empty for preflight requests
    if 'image' not in request.files:
        logger.warning("No image file provided in the request.")
        return jsonify({"error": "No image provided"}), 400
    file = request.files['image']
    if file.filename == '':
        logger.warning("No image selected.")
        return jsonify({"error": "No image selected"}), 400
    if file:
        try:
            img = Image.open(file)
            img = img.convert('L')
            img = img.resize((48, 48))
            img_array = np.array(img)
            img_array = img_array.astype('float32') / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            img_array = np.expand_dims(img_array, -1) # Add the channel dimension for grayscale

            prediction = model.predict(img_array)
            emotion_index = np.argmax(prediction)
            emotion = emotions[emotion_index]
            confidence = float(prediction[0][emotion_index] * 100)

            logger.info(f"Prediction: Emotion - {emotion}, Confidence - {confidence:.2f}%")
            return jsonify({"emotion": emotion, "confidence": confidence})

        except Exception as e:
            logger.error(f"Error processing image: {e}", exc_info=True)
            return jsonify({"error": f"Error processing image: {e}"}), 500

    return jsonify({"error": "Unknown error"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
