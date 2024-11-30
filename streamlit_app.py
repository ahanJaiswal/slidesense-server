from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)
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
    return predictions.tolist()

@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.get_json()
    if "image" not in data:
        return jsonify({"error": "No image provided"}), 400
    image_base64 = data["image"]
    image_data = base64.b64decode(image_base64)
    predictions = predict(image_data)
    return jsonify({"predictions": predictions})

if __name__ == '__main__':
    app.run(debug=True)
