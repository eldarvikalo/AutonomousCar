import socketio
import eventlet
import numpy as np
from flask import Flask
import tensorflow as tf
import keras
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2
import os

# @keras.saving.register_keras_serializable()
# class MSE(MeanSquaredError):
#     pass



# Set up the Socket.IO server and Flask app
sio = socketio.Server()
app = Flask(__name__)

# Speed limit for throttle calculation
speed_limit = 10

# Image preprocessing function
def img_preprocess(img):
    img = img[60:135, :, :]  # Crop the image to focus on relevant area
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)  # Convert image to YUV color space
    img = cv2.GaussianBlur(img, (3, 3), 0)  # Apply Gaussian blur for noise reduction
    img = cv2.resize(img, (200, 66))  # Resize image to match the model input size
    img = img / 255.0  # Normalize the image
    return img

# Function to load the model
def load_trained_model(model_path):
    try:
        print(f"Loading model from {model_path}")
        model = keras.models.load_model(model_path)  # Load model using TensorFlow/Keras
        print(f"Model successfully loaded from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Telemetry event handler for receiving data
@sio.on('telemetry')
def telemetry(sid, data):
    try:
        speed = float(data['speed'])  # Extract speed from telemetry data
        image = Image.open(BytesIO(base64.b64decode(data['image'])))  # Decode image
        image = np.asarray(image)  # Convert to numpy array
        image = img_preprocess(image)  # Preprocess the image
        image = np.array([image])  # Add batch dimension for prediction

        # Make a prediction using the model
        steering_angle = float(model.predict(image))  # Predict the steering angle
        throttle = 1.0 - speed / speed_limit  # Calculate throttle based on speed

        print(f'Steering angle: {steering_angle}, Throttle: {throttle}, Speed: {speed}')
        send_control(steering_angle, throttle)  # Send control signals

    except Exception as e:
        print(f"Error processing telemetry data: {e}")

# Connection event handler
@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    send_control(0, 1)  # Send a stop signal initially

# Send control signals to the car
def send_control(steering_angle, throttle):
    """Send steering angle and throttle to the car."""
    sio.emit('steer', data={
        'steering_angle': str(steering_angle),  # Send the steering angle as a string
        'throttle': str(throttle)  # Send the throttle as a string
    })

# Main entry point
if __name__ == '__main__':
    model = load_trained_model('Model/model.h5')  # Load the model
    if model:  # Proceed only if the model is loaded successfully
        # Wrap Flask app with SocketIO
        app = socketio.WSGIApp(sio, app)
        eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
    else:
        print("Model not loaded. Exiting.")
