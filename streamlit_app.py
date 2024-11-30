import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64

# Load TensorFlow model
model = tf.keras.models.load_model("cancer_model.h5")

def preprocess_image(image_data):
    """Preprocess the uploaded image for model prediction"""
    image = Image.open(io.BytesIO(image_data))
    image = image.resize((224, 224))  # Adjust to your model's input size
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

def predict(image_data):
    """Run the prediction and return results"""
    preprocessed = preprocess_image(image_data)
    predictions = model.predict(preprocessed)
    return predictions

# Streamlit interface
st.title("SlideSense Prediction API")
st.write("Use this API to send an image and get predictions.")

# Backend API simulation
if st.button("Start API"):
    st.write("The API is ready. Send POST requests to this app for predictions.")

# Interactive Demo for Testing
uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "bmp"])
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image")
    predictions = predict(uploaded_file.read())
    st.write(f"Predictions: {predictions}")
