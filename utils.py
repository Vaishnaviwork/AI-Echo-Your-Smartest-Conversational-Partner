import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- NLTK Data Check ---
# Run this once in your python environment to avoid LookupError:
# import nltk
# nltk.download(['punkt', 'stopwords', 'wordnet', 'omw-1.4'])

def clean_text(text):
    """
    Performs core NLP text cleaning: lowercasing, removing noise, tokenizing,
    removing stopwords, and lemmatization.
    """
    if not isinstance(text, str):
        return ""  # Handle NaN or non-string inputs
        
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Tokenize
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords (requires nltk.download('stopwords'))
    try:
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
    except LookupError:
        print("NLTK stopwords data not found. Please run nltk.download('stopwords').")
    
    # Lemmatization (requires nltk.download('wordnet'))
    try:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    except LookupError:
        print("NLTK wordnet data not found. Please run nltk.download('wordnet').")
    
    return ' '.join(tokens)

# The __init__.py file can remain empty:
# 📝 AI_Echo_Sentiment_Analysis/src/__init__.py
# (Leave this file empty)