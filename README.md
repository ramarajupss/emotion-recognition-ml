# Emotion Recognition Web App (ML + AWS)

This project uses a Convolutional Neural Network to detect emotions from facial images.

## Features
- Image-based emotion recognition using FER2013 dataset
- Frontend with HTML + JS (S3 + CloudFront hosted)
- Backend using Flask (initially hosted on EC2, now moving to SageMaker)
- Model trained using TensorFlow/Keras

## Technologies
- Python, TensorFlow, Flask
- AWS: S3, CloudFront, EC2, SageMaker (planned), IAM

## Project Structure
- `/model`: Trained model file
- `/backend`: Flask app for serving predictions
- `/frontend`: Combined UI HTML page

## Future Enhancements
- SageMaker model hosting
- Real-time webcam input
- Improve accuracy with additional training

## Trained Emotions
- Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
