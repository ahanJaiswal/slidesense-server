import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import json

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

def decode_base64_image(base64_str):
    """Decode the base64-encoded string into image bytes"""
    return base64.b64decode(base64_str)

# Streamlit interface
st.title("SlideSense Prediction API")
st.write("Use this API to send an image and get predictions.")

# Backend API simulation
if st.button("Start API"):
    st.write("The API is ready. Send POST requests to this app for predictions.")

# Accepting base64 input from the frontend
image_base64_input = st.text_area("Enter Base64 Image Data")

if image_base64_input:
    try:
        # Decode the base64 input to image data
        image_data = decode_base64_image(image_base64_input)

        # Get the prediction from the model
        predictions = predict(image_data)

        # Display the result
        st.write(f"Predictions: {predictions}")
    except Exception as e:
        st.write(f"Error: {str(e)}")

