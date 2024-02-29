import numpy as np
import hyper as hp
import torch
import flask
import json
import io
import os
import utils
from flask import Flask, request, jsonify, render_template
from sandbox import _load_model, translate,greedy_decode
import time


# Initialize our Flask application and the PyTorch model.
app = Flask(__name__)

# Initial route for 1 API
@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def _predict():
    # Initialize the data dictionary
    data = {"success": False}
    # Ensure an text was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.json.get("text"):
            # Read the text in the request
            text = flask.request.json["text"]
            # Make the prediction
            translated_text = translate(transformer, text, source_text, target_vocab)
            data["response"] = translated_text
            # Indicate that the request was a success
            data["success"] = True
    return jsonify({"response": data["response"],
                    "success": data["success"]})

if __name__=="__main__":
    print("App is running")
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: ", DEVICE)
    # Load the models
    start_time = time.time()
    try:
        transformer, source_vocab, target_vocab, source_text, target_text = _load_model()
        print("Model loaded")
        transformer.to(DEVICE)
    except Exception as e:
        print("Error loading model: ", e)    
    end_time = time.time()
    print(f"Model loaded in {end_time - start_time} seconds")
    app.run(debug=True)
    
    