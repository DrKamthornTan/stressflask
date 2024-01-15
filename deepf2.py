import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import cv2
import streamlit as st
import matplotlib.pyplot as plt
from deepface import DeepFace
from flask import Flask, request, jsonify

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tf.keras.models.load_model("keras_model.h5", compile=False)

# Load the labels
class_names = ["stressless", "stressful"]

app = Flask(__name__)

# Define a function to get the prediction from an uploaded image
def predict_image(image_file):
    # Load the image
    image = Image.open(image_file).convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, method=Image.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Predict using the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    result = {
        "class_name": class_name,
        "confidence_score": float(confidence_score),
    }

    return result

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify(error="No file found."), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify(error="No file selected."), 400

    result = predict_image(file)
    return jsonify(result), 200

if __name__ == "__main__":
    app.run()