# I dont know what to do now ._.
# Maybe flask, idk, hu ac hu ce
# Where should i host? heroku?
# Maybe torchserve is cool
import os
from flask import Flask, render_template
from dotenv import load_dotenv

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    load_dotenv()  # take environment variables from .env.
    app.run(debug=True)

