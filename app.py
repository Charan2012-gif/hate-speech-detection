import os
import re
import string
import pickle
import nltk
from flask import Flask, render_template, request, jsonify
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure a persistent path for NLTK data on Google Cloud
nltk_data_path = "/tmp/nltk_data"
os.makedirs(nltk_data_path, exist_ok=True)  # Create the directory if it doesn't exist
nltk.data.path.append(nltk_data_path)  # Tell NLTK to use this path

# Download required NLTK data
nltk.download('stopwords', download_dir=nltk_data_path)
nltk.download('punkt', download_dir=nltk_data_path)

# Initialize Flask app
app = Flask(__name__)

# Define paths for model and vectorizer
model_path = "savemodel.sav"
vectorizer_path = "vectorizer.pkl"

# Check if files exist before loading
if not os.path.exists(model_path):
    print(f"❌ ERROR: Model file '{model_path}' not found!")
    exit(1)  # Stop execution

if not os.path.exists(vectorizer_path):
    print(f"❌ ERROR: Vectorizer file '{vectorizer_path}' not found!")
    exit(1)  # Stop execution

try:
    model = pickle.load(open(model_path, 'rb'))
    vectorizer = pickle.load(open(vectorizer_path, 'rb'))
    print("✅ Model and Vectorizer loaded successfully!")
except Exception as e:
    print(f"❌ ERROR loading model/vectorizer: {e}")
    exit(1)  # Stop execution

# Define text preprocessing function
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@\w+|\#', '', text)  # Remove mentions and hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only letters and spaces
    words = word_tokenize(text)  # Tokenize
    words = [word for word in words if word not in stopwords.words('english')]  # Remove stopwords
    return " ".join(words)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        tweet = request.form.get('tweet', '')  # Use .get() to prevent KeyError

        if not tweet:
            return jsonify({"error": "No tweet provided"}), 400

        cleaned_tweet = clean_text(tweet)
        transformed_tweet = vectorizer.transform([cleaned_tweet])  # Convert to TF-IDF features
        prediction = model.predict(transformed_tweet)[0]  # Predict using model

        # Map numerical predictions to labels
        label_map = {0: "Hate Speech", 1: "Offensive", 2: "Neutral"}
        prediction_label = label_map.get(prediction, "Unknown")  # Default to 'Unknown' if not found
        
        return render_template('index.html', prediction=prediction_label, tweet=tweet)
    
    except Exception as e:
        print(f"❌ ERROR in prediction: {e}")
        return jsonify({"error": f"Internal Server Error: {e}"}), 500  # Return error response

if __name__ == "__main__":
    app.run(debug=True)
