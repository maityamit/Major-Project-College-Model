import pandas as pd
import numpy as np
import pickle
import re
import string
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')

# ---------- 1. Load Dataset ----------
df = pd.read_csv('dataset.csv')  # Replace with your CSV path
df = df.dropna(subset=['text', 'label'])  # Drop missing entries

# ---------- 2. Preprocessing Functions ----------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    stopwrds = set(stopwords.words('english'))
    words = [w for w in words if w not in stopwrds]
    stemmer = PorterStemmer()
    words = [stemmer.stem(w) for w in words]
    return ' '.join(words)

df['text'] = df['text'].apply(preprocess_text)

# ---------- 3. Vectorization ----------
X = df['text']
y = df['label'].astype(int)

tfidf = TfidfVectorizer(max_features=2500, min_df=2)
X_tfidf = tfidf.fit_transform(X)

# ---------- 4. Train Model ----------
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42, stratify=y)

model = MultinomialNB()
model.fit(X_train, y_train)

# ---------- 5. Evaluate ----------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Trained — Accuracy: {accuracy:.3f}")

# ---------- 6. Save Model and Vectorizer ----------
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

with open('nb_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ TF-IDF vectorizer and Naive Bayes model saved to disk.")