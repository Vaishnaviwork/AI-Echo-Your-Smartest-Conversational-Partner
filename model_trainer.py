import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from joblib import dump
import os
import warnings
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

warnings.filterwarnings("ignore", category=FutureWarning)

# --- NLTK Setup ---
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    # CRITICAL FIX: We keep 'not', 'no', 'never' out of STOP_WORDS to allow negation handling
    STOP_WORDS = set(stopwords.words('english')) - {'not', 'no', 'never'} 
    LEMMATIZER = WordNetLemmatizer()
except LookupError:
    STOP_WORDS = set()
    LEMMATIZER = None
    print("Warning: NLTK data not fully loaded. Text cleaning will be basic.")

# --- Text Preprocessing Function (The Negation Fix) ---
def clean_text(text):
    """Applies cleaning, tokenization, stopword removal, negation handling, and lemmatization."""
    if not isinstance(text, str): 
        return ""
    
    text = text.lower()
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-z\s]', '', text) 
    tokens = nltk.word_tokenize(text)

    # --- NEW: NEGATION HANDLING (CRITICAL FOR ACCURACY) ---
    processed_tokens = []
    # Words that signal negation
    negation_words = {"not", "no", "never", "n't"} 
    is_negated = False
    
    for word in tokens:
        if word in negation_words:
            # Set the negation flag for the next word
            is_negated = True
            # Keep the negation word itself, as it is no longer a stop word
            processed_tokens.append(word) 
            continue

        if is_negated:
            # Prepend 'NOT_' to the next word and reset the flag
            # This creates features like 'NOT_good', which the model learns is negative
            processed_tokens.append('NOT_' + word)
            is_negated = False
        else:
            processed_tokens.append(word)
    
    tokens = processed_tokens
    # --- END NEGATION HANDLING ---

    # Stop Word Removal (Uses the adjusted STOP_WORDS set)
    tokens = [word for word in tokens if word not in STOP_WORDS]
    
    if LEMMATIZER:
        tokens = [LEMMATIZER.lemmatize(word) for word in tokens]
        
    return ' '.join(tokens)
    
# --- Original Functions ---

def to_binary_sentiment(rating):
    """Converts 1-5 star ratings to Positive (4, 5) or Negative (1, 2)."""
    if rating >= 4:
        return 'positive'
    elif rating <= 2:
        return 'negative'
    # Neutral (3) reviews are filtered out during model training
    else:
        return None 

def train_and_save_model():
    """Trains a balanced Logistic Regression model with optimized features."""
    
    # --- Path Configuration (using absolute path logic is safer, but keeping relative paths for now) ---
    # The relative path assumes the script is run from the 'src' directory
    PROCESSED_DATA_PATH = os.path.join('..', 'data', 'processed', 'cleaned_data.csv')
    ML_MODEL_PATH = os.path.join('..', 'models', 'best_ml_model.pkl')
    VECTORIZER_PATH = os.path.join('..', 'models', 'tfidf_vectorizer.pkl')
    
    print("\n--- Model Training ---")
    
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)
        # Assuming the rating column is named 'sentiment' in your cleaned_data.csv
        df.rename(columns={'sentiment': 'rating'}, inplace=True)
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        
    except (FileNotFoundError, pd.errors.EmptyDataError) as e:
        print(f"🚨 Error loading data: {e}. Please ensure you ran data_cleaner.py.")
        return

    # 1. Convert 5-class problem to Binary Classification
    print("Converting 5-class rating to Binary Sentiment...")
    df['sentiment'] = df['rating'].apply(to_binary_sentiment)
    df.dropna(subset=['sentiment'], inplace=True) 

    if df.empty:
        print("🛑 Error: DataFrame is empty after dropping Neutral (3) ratings. Cannot train model.")
        return

    # --- APPLY NEW CLEANING FUNCTION HERE ---
    df['cleaned_review'] = df['cleaned_review'].fillna('').apply(clean_text)
    
    X = df['cleaned_review']
    y = df['sentiment']
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 2. FEATURE OPTIMIZATION: TfidfVectorizer 
    print("Fitting TfidfVectorizer with N-grams and Min Document Frequency...")
    # TfidfVectorizer will now create features like 'NOT_good'
    vectorizer = TfidfVectorizer(max_features=20000, 
                                 ngram_range=(1, 2), 
                                 min_df=5) 
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # 3. MODEL FIX: Logistic Regression with Class Weighting and Regularization
    print("Training Balanced Logistic Regression Model...")
    
    model = LogisticRegression(
        max_iter=2000, 
        solver='lbfgs', 
        random_state=42, 
        C=1.0, 
        class_weight='balanced'
    )
    model.fit(X_train_vec, y_train)
    print("Training complete.")

    # Evaluate Model
    y_pred = model.predict(X_test_vec)
    
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Save Model and Vectorizer
    os.makedirs(os.path.dirname(ML_MODEL_PATH), exist_ok=True)
    dump(model, ML_MODEL_PATH)
    print(f"✅ Model saved to {ML_MODEL_PATH}")
    
    dump(vectorizer, VECTORIZER_PATH)
    print(f"✅ Vectorizer saved to {VECTORIZER_PATH}")

if __name__ == '__main__':
    train_and_save_model()