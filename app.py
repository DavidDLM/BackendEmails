from flask import Flask, jsonify
import random
import os
from keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocess import preprocess_new_text
import joblib

app = Flask(__name__)

dataPath = "./data"
categories = ["Crime", "Entertainment", "Politics", "Science"]
tfidfPath = './models/tfidf_vectorizer.pkl'

if os.path.exists(tfidfPath):
    tfidf = joblib.load(tfidfPath)
else:
    raise FileNotFoundError("TF-IDF vectorizer not found!")

model = None
def getModel():
    global model
    if model is None:
        model = load_model("./models/w2v_nn.keras")
    return model

@app.route('/random-classify', methods=['GET'])
def random_classify():
    try:
        email_folder = './assets'
        email_files = [f for f in os.listdir(email_folder) if f.endswith('.txt')]

        if not email_files:
            return jsonify({"error": "No email files found"}), 404

        random_file = random.choice(email_files)
        file_path = os.path.join(email_folder, random_file)

        with open(file_path, 'r') as file:
            email_text = file.read()

        vectorized_text = preprocess_new_text(email_text, tfidf)
        prediction = getModel().predict(vectorized_text)
        labels = ["Crime", "Entertainment", "Politics", "Science"]
        predicted_category = labels[prediction.argmax()]

        return jsonify({
            "fileName": random_file,
            "text": email_text,
            "category": predicted_category
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
