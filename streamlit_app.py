from flask import Flask, request, jsonify
import threading
import streamlit as st
import requests
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# TensorFlow Model
model = tf.keras.models.load_model('cancer_model.h5')

def preprocess_image(image_data):
    image = Image.open(io.BytesIO(image_data))
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

def predict(image_data):
    preprocessed = preprocess_image(image_data)
    predictions = model.predict(preprocessed)
    return predictions.tolist()

@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.get_json()
    image_data = base64.b64decode(data['image'])
    predictions = predict(image_data)
    return jsonify(predictions)

def run_flask():
    app.run(host='0.0.0.0', port=5000)

def run_streamlit():
    st.title("SlideSense Prediction API")
    st.write("Use this API to send an image and get predictions.")

    image_base64_input = st.text_area("Enter Base64 Image Data")
    if image_base64_input:
        response = requests.post('http://localhost:5000/predict', json={'image': image_base64_input})
        predictions = response.json()
        st.write(f"Predictions: {predictions}")

if __name__ == '__main__':
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.start()

    run_streamlit()
