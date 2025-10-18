import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
import sys

# --- CONFIGURATION AND PATHS ---
RAW_DATA_PATH = r"C:\Users\vaish\OneDrive\Desktop\Learn Data Science\Data-Science-Projects\AI_Echo_Sentiment_Analysis\data\raw\chatgpt_reviews.csv"
PROCESSED_DATA_PATH = r"C:\Users\vaish\OneDrive\Desktop\Learn Data Science\Data-Science-Projects\AI_Echo_Sentiment_Analysis\data\processed\cleaned_data.csv"

# --- NLTK DOWNLOAD (Self-Correcting Step) ---
# NOTE: This assumes NLTK data (punkt, stopwords, etc.) is already downloaded.

# --- CORE CLEANING FUNCTION (Assumed to be defined in data_cleaner.py) ---
# Ensure you have 'nltk.download' run once successfully for a cleaner run.
def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    try:
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    except LookupError:
        pass 
    return ' '.join(tokens)


def process_data():
    """Loads raw data, fixes column names, cleans it, and saves the processed data."""
    
    print(f"\n--- Starting Data Processing ---")
    print(f"Loading raw data from {RAW_DATA_PATH}...")
    
    try:
        df = pd.read_csv(RAW_DATA_PATH)
    except FileNotFoundError:
        print(f"🚨 Error: Raw data file not found at {RAW_DATA_PATH}")
        return

    # 🚨 CRITICAL FIX: RENAME_MAP using the column names you provided!
    RENAME_MAP = {
        'Review': 'review',    # Map 'Review' (text) to 'review'
        'Ratings': 'sentiment' # Map 'Ratings' (score) to 'sentiment'
    }
    df.rename(columns=RENAME_MAP, inplace=True)
    
    # Check if the required columns exist after renaming
    if 'review' not in df.columns or 'sentiment' not in df.columns:
        print("🚨 CRITICAL ERROR: Renaming failed. Check the column map again.")
        print(f"Current columns: {df.columns.tolist()}")
        return

    # --- Data Cleaning Continues ---
    
    # Now df.dropna should work
    df.dropna(subset=['review', 'sentiment'], inplace=True) 
    df['review'].fillna('', inplace=True)
    
    if df.empty:
        print("🛑 WARNING: DataFrame is empty after dropping missing values. Nothing will be saved.")
        return
        
    print(f"Total reviews remaining: {len(df)}")
    print("Applying text cleaning and preprocessing...")
    df['cleaned_review'] = df['review'].apply(clean_text)
    
    cleaned_df = df[['cleaned_review', 'sentiment']]
    
    output_dir = os.path.dirname(PROCESSED_DATA_PATH)
    os.makedirs(output_dir, exist_ok=True)
    
    cleaned_df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"✅ Cleaned data successfully saved to {PROCESSED_DATA_PATH}")


if __name__ == '__main__':
    # Add NLTK data path search (optional but safe)
    try:
        import nltk
        NLTK_DATA_DIR = r'C:\Users\vaish\AppData\Roaming\nltk_data'
        if NLTK_DATA_DIR not in nltk.data.path:
            nltk.data.path.insert(0, NLTK_DATA_DIR)
    except ImportError:
        pass
    process_data()