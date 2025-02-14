import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_text(text):
    """
    Clean text data by removing URLs, mentions, hashtags, and non-alphanumeric characters
    Args:
        text: string
    """
    
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower().strip()
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)

def load_data():
    """
    Load training and test data from CSV files
    Returns:
        train_df: pandas DataFrame containing training data
        test_df: pandas DataFrame containing test data
    """
    train_df = pd.read_csv('data/raw/train.csv')
    test_df = pd.read_csv('data/raw/test.csv')
    return train_df, test_df

def vectorize_text(train_text, val_text, test_text, max_features=5000, ngram_range=(1, 2)):
    """
    Vectorize text data using TF-IDF
    Args:
        train_text: pandas Series containing training text data
        val_text: pandas Series containing validation text data
        test_text: pandas Series containing test text data
        max_features: int, default=5000
        ngram_range: tuple, default=(1, 2)
    Returns:
        X_train_vec: sparse matrix for training data
        X_val_vec: sparse matrix for validation data
        X_test_vec: sparse matrix for test data
    """
    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X_train_vec = tfidf.fit_transform(train_text)
    X_val_vec = tfidf.transform(val_text)
    X_test_vec = tfidf.transform(test_text)
    return X_train_vec, X_val_vec, X_test_vec