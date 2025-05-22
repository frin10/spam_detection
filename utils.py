import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download these once
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """
    Preprocess input text with best NLP practices for spam detection:
    - Lowercasing
    - Remove URLs, emails, and HTML tags
    - Remove punctuation and digits
    - Tokenize and remove stopwords
    - Lemmatize tokens
    - Join tokens back to string
    """
    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove emails
    text = re.sub(r'\S+@\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove digits (optional; emails may have numbers)
    text = re.sub(r'\d+', '', text)

    # Tokenize by whitespace
    tokens = text.split()

    # Remove stopwords
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Lemmatize tokens
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    # Join back to string
    clean_text = ' '.join(lemmatized_tokens)

    return clean_text
