import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load your trained model
model = load_model("sign_language_model.h5")
letters = ['A','B','C','D','E','F','G','H','I','K','L','M','N',
           'O','P','Q','R','S','T','U','V','W','X','Y']

st.title("Sign Language Recognition App")

# Webcam input
img_file_buffer = st.camera_input("Show your hand sign to the camera")

if img_file_buffer is not None:
    # Read the image
    image = Image.open(img_file_buffer)
    image = image.convert('L')              # Convert to grayscale
    image = image.resize((28,28))           # Resize to 28x28
    img_array = np.array(image)/255.0       # Normalize
    img_array = img_array.reshape(1,28,28,1)

    # Predict
    pred_prob = model.predict(img_array)
    pred_label = np.argmax(pred_prob)
    letter = letters[pred_label]

    st.write(f"Predicted Letter: {letter}")
    st.image(image, caption="Captured Image", use_column_width=True)
