"""
main.py

Usage:
    1. Train the model: python main.py train
    2. Run the server: python main.py

This script either trains a neural network model using the MNIST dataset loaded from CSV files,
or starts a Flask server that provides endpoints for digit prediction and introspection.
It uses TensorFlow and Keras for model definition and training, and Flask for serving HTTP requests.
"""

import sys
import os
import base64
import numpy as np
from io import BytesIO
from PIL import Image, ImageOps
import pandas as pd  # Import the pandas library for handling CSV files

# Suppress TensorFlow logging for information messages; set '0' to see all messages.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Flatten
from flask import Flask, request, jsonify, render_template

# --- Model Training ---

MODEL_FILENAME = 'mnist_model.h5'


def train_and_save_model():
    """
    Loads the MNIST dataset from CSV files, builds a neural network model,
    trains and evaluates it, and finally saves the trained model to disk.
    """
    print("Loading MNIST dataset from CSV files...")

    # --- Load data from CSV files ---
    train_file_path = 'dataset/mnist_train.csv'
    test_file_path = 'dataset/mnist_test.csv'

    if not os.path.exists(train_file_path) or not os.path.exists(test_file_path):
        print(f"ERROR: Dataset files not found.")
        print(f"Ensure that `{train_file_path}` and `{test_file_path}` exist.")
        sys.exit(1)

    train_df = pd.read_csv(train_file_path)
    test_df = pd.read_csv(test_file_path)

    # Separate labels (first column) from image pixel values
    y_train = train_df['label'].values
    x_train = train_df.drop('label', axis=1).values

    y_test = test_df['label'].values
    x_test = test_df.drop('label', axis=1).values

    # Reshape flat array (784 pixels) to 28x28 image format
    x_train = x_train.reshape(-1, 28, 28)
    x_test = x_test.reshape(-1, 28, 28)
    # ----------------------------------------------

    # Normalize images to have values between 0 and 1
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    print("Defining the model architecture...")
    model = Sequential([
        tf.keras.Input(shape=(28, 28), name='input_layer'),
        Flatten(),
        Dense(128, activation='relu', name='hidden_layer_1'),
        Dense(64, activation='relu', name='hidden_layer_2'),
        Dense(10, activation='softmax', name='output_layer')
    ])

    print("Compiling the model...")
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    print("Training the model...")
    model.fit(x_train, y_train, epochs=5, validation_split=0.1)

    print("Evaluating the model...")
    model.evaluate(x_test, y_test)

    print(f"Saving the model as `{MODEL_FILENAME}`...")
    model.save(MODEL_FILENAME)
    print("Model saved successfully!")


# --- Flask Server Logic ---

app = Flask(__name__)
model = None
activation_model = None


def load_keras_model():
    """
    Loads the trained Keras model from disk and creates an auxiliary model
    to extract activations from each layer (excluding the input layer).
    """
    global model, activation_model
    if not os.path.exists(MODEL_FILENAME):
        print(f"ERROR: Model file `{MODEL_FILENAME}` not found.")
        print("Please train the model first by running: python main.py train")
        sys.exit(1)

    print(f"Loading model `{MODEL_FILENAME}`...")
    model = load_model(MODEL_FILENAME)
    print("Model loaded.")

    # Create a model that returns the outputs (activations) of each layer
    layer_outputs = [layer.output for layer in model.layers]
    activation_model = Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers[1:]])
    print("Activation model created.")


def preprocess_image(img_data):
    """
    Processes the base64-encoded image received from the canvas into a format
    suitable for the model. This includes decoding, converting to grayscale,
    inverting colors, resizing, normalizing, and adjusting dimensions.

    Args:
        img_data (str): The base64 encoded image data with header.

    Returns:
        numpy.ndarray: Preprocessed image ready for model prediction.
    """
    # Decode the base64 string
    img_str = base64.b64decode(img_data.split(',')[1])
    img = Image.open(BytesIO(img_str))

    # Convert image to grayscale
    img = img.convert('L')

    # Invert colors: the MNIST model expects white digits on a black background
    img = ImageOps.invert(img)

    # Resize image to 28x28 pixels using high-quality resampling
    img = img.resize((28, 28), Image.Resampling.LANCZOS)

    # Convert to numpy array and normalize pixel values
    img_array = np.array(img) / 255.0

    # Expand dimensions to match model input shape (batch_size, height, width)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


@app.route('/')
def index():
    """
    Serves the main HTML page.

    Returns:
        Rendered HTML template for the index page.
    """
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives an image from the client, preprocesses it, performs a prediction
    using the loaded model, extracts layer activations, and returns the results
    in JSON format.

    Returns:
        A JSON response containing the predicted digit, activations from
        intermediate layers, and output layer weights.
    """
    if not model or not activation_model:
        return jsonify({'error': 'Model not loaded'}), 500

    data = request.get_json()
    if 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    # Preprocess the incoming image for prediction
    processed_image = preprocess_image(data['image'])

    # Get activations from all layers from the activation model
    activations = activation_model.predict(processed_image, verbose=0)

    # The last activation corresponds to the final prediction probabilities
    prediction_probabilities = activations[-1][0]
    predicted_digit = int(np.argmax(prediction_probabilities))

    # Convert activation arrays to lists for JSON serialization
    serializable_activations = {
        'hidden_layer_1': activations[0][0].tolist(),
        'hidden_layer_2': activations[1][0].tolist(),
        'output_layer': activations[2][0].tolist()
    }

    # Retrieve the weights of the output layer (connection from hidden_layer_2 to output_layer)
    output_layer_weights = model.layers[-1].get_weights()[0].tolist()

    return jsonify({
        'prediction': predicted_digit,
        'activations': serializable_activations,
        'weights': output_layer_weights
    })


def run_server():
    """
    Loads the Keras model and starts the Flask server.
    """
    load_keras_model()
    print("Starting Flask server at http://127.0.0.1:5000")
    app.run(debug=True, use_reloader=False)


if __name__ == '__main__':
    # Determine action based on command-line argument: train the model or run the server.
    if len(sys.argv) > 1 and sys.argv[1].lower() == 'train':
        train_and_save_model()
    else:
        run_server()