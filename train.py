import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
from utils import preprocess_text

# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")[['v1', 'v2']]
df.columns = ['label', 'text']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Preprocess text
df['clean_text'] = df['text'].apply(preprocess_text)

# Split data before vectorizing to avoid leakage
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df['clean_text'], df['label'], test_size=0.2, random_state=42)

# Vectorize text with limited features and remove rare words
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000, min_df=5)
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

# Train Logistic Regression with regularization
model = LogisticRegression(max_iter=1000, C=1.0)
model.fit(X_train, y_train)

# Evaluate
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("Train Accuracy:", accuracy_score(y_train, y_train_pred))
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print("Classification Report (Test):\n", classification_report(y_test, y_test_pred))

# Save model and vectorizer
with open("spam_classifier.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
