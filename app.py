import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import numpy as np
import av
import tensorflow as tf

# model = tf.keras.models.load_model('my_model.h5')
# model.load_weights('my_model_weights.weights.h5')

import requests
import tensorflow as tf

def download_file_from_github(url, output_path):
    response = requests.get(url)
    response.raise_for_status()  # Raise an error if the download failed
    with open(output_path, 'wb') as f:
        f.write(response.content)

# URLs to the model and weights files on GitHub
model_url = "https://raw.githubusercontent.com/NandaKishoreYadav/Facial-Emotion-Detection/main/my_model.h5"
weights_url = "https://raw.githubusercontent.com/NandaKishoreYadav/Facial-Emotion-Detection/main/my_model_weights.weights.h5"

# Paths where the files will be saved
model_path = "model.h5"
weights_path = "model_weights.h5"

# Download the model file
download_file_from_github(model_url, model_path)

# Download the weights file
download_file_from_github(weights_url, weights_path)

# Load the model
model = tf.keras.models.load_model(model_path)

# Load the weights into the model
model.load_weights(weights_path)

# Now the model is fully loaded with the architecture and weights




mapper = {
    0: "anger",
    1: "happy",
    2: "sad",
    3: "surprise",
    4: "neutral",
}

st.title('Facial Emotion Detection')

def detect_emotion(face_image):
    face_image = cv2.resize(face_image, (48, 48))
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    face_image = face_image.astype('float32') / 255.0
    face_image = np.expand_dims(face_image, axis=0)
    face_image = np.expand_dims(face_image, axis=-1)

    predictions = model.predict(face_image)
    emotion_index = np.argmax(predictions)
    return mapper.get(emotion_index, "unknown")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def transform(frame: av.VideoFrame):
    img = frame.to_ndarray(format="bgr24")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        emotion = detect_emotion(face)
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img, f"Emotion: {emotion}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="streamer",
    video_frame_callback=transform,
    sendback_audio=False
)
