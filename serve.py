import numpy as np
import hyper as hp
import torch
import flask
import json
import io
import os
import utils
from flask import Flask, request, jsonify, render_template


# Initial model
global transformer
transformer = None

# Initialize our Flask application and the PyTorch model.
app = flask.Flask(__name__)

# Initial route for 1 API
@app.route('/')
def home():
    return render_template('templates/index.html')

@app.route("/predict", methods=["POST"])
def _predict():
    # Initialize the data dictionary
    data = {"success": False}
    # Ensure an text was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("text"):
            # Read the text in the request
            text = flask.request.files["text"].read()
            text = text.decode("utf-8")
            # Make the prediction
            data["response"] = ...
            # Indicate that the request was a success
            data["success"] = True
    return json.dumps(data, ensure_ascii=False)

if __name__=="__main__":
    print("App is running")
    #Load model
    transformer, source_vocab, target_vocab, source_text, target_text = utils._load_model()
    print("Model loaded")
    app.run(debug=True)