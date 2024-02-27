# I dont know what to do now ._.
# Maybe flask, idk, hu ac hu ce
# Where should i host? heroku?
# Maybe torchserve is cool
import os
from flask import Flask, render_template
from flask import request, jsonify
from dotenv import load_dotenv
from src.seq2seq.model import greedy_decode

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    load_dotenv()  # take environment variables from .env.
    app.run(debug=True)


@app.route('/translate', methods=['POST'])
def translate():
    # Extract the input text from the request
    input_text = request.form['source']
    
    # Translate the input text using your model
    # Replace `translate_text` with the actual function to translate text
    translated_text = greedy_decode(input_text)
    
    # Return the translated text
    return jsonify({'translation': translated_text})