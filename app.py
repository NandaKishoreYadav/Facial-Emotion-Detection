import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import numpy as np
import av
from tensorflow.keras.models import load_model

# Load the model and ensure it is compiled
try:
    model = load_model('my_model.h5')
    model.compile()  # Ensure the model is compiled
except Exception as e:
    st.error(f"Error loading model: {e}")

mapper = {
    0: "anger",
    1: "happy",
    2: "sad",
    3: "surprise",
    4: "neutral",
}

st.title('Facial Emotion Detection')

def detect_emotion(face_image):
    try:
        face_image = cv2.resize(face_image, (48, 48))
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        face_image = face_image.astype('float32') / 255.0
        face_image = np.expand_dims(face_image, axis=0)
        face_image = np.expand_dims(face_image, axis=-1)

        predictions = model.predict(face_image)
        emotion_index = np.argmax(predictions)
        return mapper.get(emotion_index, "unknown")
    except Exception as e:
        st.error(f"Error in emotion detection: {e}")
        return "error"

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def transform(frame: av.VideoFrame):
    try:
        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]
            emotion = detect_emotion(face)
            if emotion != "error":
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(img, f"Emotion: {emotion}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    except Exception as e:
        st.error(f"Error in video frame processing: {e}")
        return frame

webrtc_streamer(
    key="streamer",
    video_frame_callback=transform,
    sendback_audio=False
)
