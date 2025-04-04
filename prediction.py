import pandas as pd
import numpy as np
import re
import nltk
import string
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Download required resources
nltk.download('stopwords')
nltk.download('punkt')

# Define preprocessing function
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@\w+|\#', '', text)  # Remove mentions and hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    words = word_tokenize(text)  # Tokenization
    words = [word for word in words if word not in stopwords.words('english')]  # Remove stopwords
    return " ".join(words)

# Function to preprocess input text for model
def preprocess_input(text, vectorizer):
    text = clean_text(text)
    return vectorizer.transform([text])

if _name_ == "_main_":
    # Load dataset
    df = pd.read_csv("labeled_data.csv")
    
    # Drop unnecessary columns
    df = df[['tweet', 'class']]  # Selecting only tweet and class labels
    
    # Apply preprocessing
    df['clean_tweet'] = df['tweet'].apply(clean_text)
    
    # Vectorization
    vectorizer = TfidfVectorizer(max_features=5000)
    X_tfidf = vectorizer.fit_transform(df['clean_tweet'])
    y = df['class']
    
    # Save vectorizer for deployment
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    
    # Define base models
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    log_reg = LogisticRegression(max_iter=500)
    svm = SVC(kernel='linear', probability=True)
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    
    # Combine models using Voting Classifier
    ensemble_model = VotingClassifier(estimators=[
        ('rf', rf),
        ('log_reg', log_reg),
        ('svm', svm),
        ('xgb', xgb)
    ], voting='soft')  # Soft voting averages probabilities
    
    # Train the model
    ensemble_model.fit(X_tfidf, y)
    
    # Save the trained model
    with open("model.pkl", "wb") as f:
        pickle.dump(ensemble_model, f)
    
    print("Model training completed and saved!")
    
    # Load the saved model and vectorizer for prediction
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    
    # Example prediction
    input_text = "This is an example tweet for classification."
    processed_input = preprocess_input(input_text, vectorizer)
    prediction = model.predict(processed_input)
    
    print(f"Predicted Class: {prediction[0]}")