import streamlit as st
import pandas as pd
import plotly.express as px
from joblib import load
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
import io

# Suppress warnings and set page configuration
warnings.filterwarnings("ignore", category=FutureWarning)
st.set_page_config(page_title="AI Echo: Your Smartest Conversational Partner Dashboard", layout="wide", initial_sidebar_state="collapsed")

# --- CONFIGURATION & ROBUST PATHS FIX ---
# Get the directory of the current script (src/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Navigate one level up to the project root
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

# Define absolute paths using the project root
DATA_PATH = os.path.join(ROOT_DIR, 'data', 'processed', 'cleaned_data.csv')
MODEL_PATH = os.path.join(ROOT_DIR, 'models', 'best_ml_model.pkl')
VECTORIZER_PATH = os.path.join(ROOT_DIR, 'models', 'tfidf_vectorizer.pkl')

# --- NLTK Setup ---
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    # NOTE: We keep 'not', 'no', 'never' out of STOP_WORDS to allow negation handling
    STOP_WORDS = set(stopwords.words('english')) - {'not', 'no', 'never'} 
    LEMMATIZER = WordNetLemmatizer()
except LookupError:
    STOP_WORDS = set()
    LEMMATIZER = None
    print("Warning: NLTK data not fully loaded. Text cleaning will be basic.")


# --- Data and Model Loading ---

@st.cache_data
def load_data():
    """Loads and preprocesses the cleaned data."""
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        st.error(f"Error: Data file not found. Checked path: {DATA_PATH}. Please ensure cleaned_data.csv exists.")
        st.stop()
    except pd.errors.EmptyDataError:
        st.error(f"Error: Data file is empty. Please check your data_cleaner.py script.")
        st.stop()
        
    # Standardize column names
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]

    # Handle 'sentiment' or 'rating' column naming
    if 'sentiment' in df.columns and 'rating' not in df.columns:
        df.rename(columns={'sentiment': 'rating'}, inplace=True)
        
    # Ensure rating column exists and is numeric
    if 'rating' not in df.columns:
        st.error("Error: The essential 'rating' column is missing from your cleaned data.")
        st.stop()
        
    # --- Feature Engineering for Analysis ---
    
    # 1. 3-Class Sentiment (Used for EDA)
    def to_3class_sentiment(rating):
        if rating in [4, 5]:
            return 'Positive'
        elif rating == 3:
            return 'Neutral'
        elif rating in [1, 2]:
            return 'Negative'
        return 'Unknown'
        
    df['sentiment_3class'] = df['rating'].apply(to_3class_sentiment)
    
    # 2. Review Length
    df['review_length'] = df['cleaned_review'].apply(lambda x: len(str(x).split()))
    
    return df

@st.cache_resource
def load_artifacts():
    """Loads the pre-trained model and vectorizer."""
    try:
        model = load(MODEL_PATH)
        vectorizer = load(VECTORIZER_PATH)
        return model, vectorizer
    except FileNotFoundError:
        st.error(f"Error: Model or Vectorizer file not found. Checked path: {MODEL_PATH}. Please ensure both files exist in the 'models' folder.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        st.stop()

# Load everything once
df = load_data()
model, vectorizer = load_artifacts()

# --- Text Preprocessing Function (Used for live prediction) ---
def clean_text(text):
    """Applies cleaning, tokenization, stopword removal, negation handling, and lemmatization."""
    if not isinstance(text, str): 
        return ""
    
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = nltk.word_tokenize(text)

    # --- NEGATION HANDLING ---
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
            processed_tokens.append('NOT_' + word)
            is_negated = False
        else:
            processed_tokens.append(word)
    
    tokens = processed_tokens
    # --- END NEGATION HANDLING ---

    # Stop Word Removal (The set above was adjusted to keep negation words)
    tokens = [word for word in tokens if word not in STOP_WORDS]
    
    if LEMMATIZER:
        tokens = [LEMMATIZER.lemmatize(word) for word in tokens]
        
    return ' '.join(tokens)


# --- Prediction Function ---
def predict_sentiment(review_text):
    """Cleans text, vectorizes, and predicts sentiment."""
    cleaned_text = clean_text(review_text)
    features = vectorizer.transform([cleaned_text])
    
    prediction = model.predict(features)[0] 
    
    probabilities = model.predict_proba(features)[0]
    
    # Get confidence for the predicted class
    confidence = probabilities[list(model.classes_).index(prediction)]
        
    return prediction, confidence

# --- Visualization Helpers ---

def plot_wordcloud(text_data, title, color_map='magma'):
    """Generates a word cloud from a list of text."""
    full_text = ' '.join(text_data.astype(str))
    
    if not full_text.strip():
        st.warning(f"No text data available for {title} Word Cloud.")
        return

    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap=color_map, 
                          max_words=100).generate(full_text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)


# --- DASHBOARD LAYOUT ---

st.title("⭐️ AI-Powered Sentiment Insights Dashboard 📈")
st.markdown("---")

tab1, tab2 = st.tabs(["🚀 Live Model Prediction", "📊 Comprehensive Insights (EDA)"])

with tab1:
    st.header("Live Sentiment Analyzer")
    st.subheader("Predict Sentiment from Your Text Reviews")

    # Input area for live prediction
    review_input = st.text_area("Enter your review text:", height=150, placeholder="e.g., The product is excellent, but the battery life is awful.")

    if st.button("Analyze Sentiment"):
        if review_input:
            prediction, confidence = predict_sentiment(review_input)
            
            # Display Results
            st.markdown("---")
            
            # Determine color and icon based on binary prediction
            if prediction == 'positive':
                icon = "✅"
                color = "green"
                message = f"**Positive**"
            elif prediction == 'negative':
                icon = "❌"
                color = "red"
                message = f"**Negative**"
            
            st.markdown(f"### {icon} Predicted Sentiment: <span style='color:{color}'>{message}</span>", unsafe_allow_html=True)
            st.markdown(f"Confidence: **{confidence:.2f}**")
                
        else:
            st.warning("Please enter some text to analyze.")

with tab2:
    st.header("Core Sentiment Insights")
    st.markdown("Exploring the relationship between Review Ratings, Sentiment Classification, and Text Length.")

    # -------------------------------------------------------------------------
    # ROW 1: Overall Distributions
    # -------------------------------------------------------------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("1. Review Rating Distribution (Q1)")
        st.caption("Distribution of 1 to 5-star ratings.")
        
        rating_counts = df['rating'].value_counts().sort_index()
        fig_rating = px.bar(rating_counts, 
                            x=rating_counts.index, 
                            y=rating_counts.values, 
                            labels={'x': 'Star Rating', 'y': 'Number of Reviews'},
                            color=rating_counts.index.astype(str),
                            color_discrete_sequence=px.colors.sequential.Plotly3)
        st.plotly_chart(fig_rating, use_container_width=True)
        st.markdown(f"**Insight:** The majority of reviews are **{df['rating'].mode()[0]}-star**, indicating overall positive bias.")

    with col2:
        st.subheader("2. Overall Sentiment Split (Q1 Key)")
        st.caption("Proportion of Positive, Neutral, and Negative reviews based on ratings.")
        
        sentiment_counts = df['sentiment_3class'].value_counts()
        fig_sentiment = px.pie(sentiment_counts, names=sentiment_counts.index, values=sentiment_counts.values, 
                               title='Overall 3-Class Sentiment Proportion',
                               color_discrete_map={'Positive':'#00cc96', 'Negative':'#ef553b', 'Neutral':'#FECB52'})
        st.plotly_chart(fig_sentiment, use_container_width=True)
        st.markdown(f"**Insight:** **{sentiment_counts.get('Positive', 0) / sentiment_counts.sum() * 100:.1f}%** of reviews are classified as Positive.")


    with col3:
        st.subheader("3. Rating vs. Sentiment Consistency (Q2 Key)")
        st.caption("Do 1-star reviews match Negative sentiment? (Count of reviews by rating and sentiment)")
        
        cross_tab = pd.crosstab(df['rating'], df['sentiment_3class'])
        fig_cross = px.bar(cross_tab, 
                           title='Sentiment Classification by Star Rating',
                           color_discrete_map={'Positive':'#00cc96', 'Negative':'#ef553b', 'Neutral':'#FECB52'})
        fig_cross.update_layout(barmode='stack')
        st.plotly_chart(fig_cross, use_container_width=True)
        st.markdown("**Insight:** The mapping shows high consistency: 1-star reviews are almost entirely **Negative**.")


    st.markdown("---")

    # -------------------------------------------------------------------------
    # ROW 2: Keywords and Text Length
    # -------------------------------------------------------------------------
    col4, col5 = st.columns(2)

    with col4:
        st.subheader("4. Sentiment vs. Review Length (Q8 Key)")
        st.caption("Distribution of review length (word count) for each rating.")
        
        fig_length = px.box(df, x='rating', y='review_length', 
                            title='Review Length Distribution by Rating', 
                            color='rating')
        st.plotly_chart(fig_length, use_container_width=True)
        
        avg_len_pos = df[df['sentiment_3class'] == 'Positive']['review_length'].mean()
        avg_len_neg = df[df['sentiment_3class'] == 'Negative']['review_length'].mean()
        
        st.markdown(f"**Insight:** Negative reviews are typically **{avg_len_neg:.1f}** words, while Positive are **{avg_len_pos:.1f}** words. This indicates people write **longer reviews when they are more expressive** (often when unhappy).")

    with col5:
        st.subheader("5. Most Common Keywords (Q3 Key)")
        st.caption("What users love (Positive) vs. what they complain about (Negative).")
        
        # Positive Keywords
        positive_text = df[df['sentiment_3class'] == 'Positive']['cleaned_review']
        st.markdown("**Positive Keywords**")
        plot_wordcloud(positive_text, 'Positive Keywords', color_map='Greens')

        # Negative Keywords
        negative_text = df[df['sentiment_3class'] == 'Negative']['cleaned_review']
        st.markdown("**Negative Feedback Themes**")
        plot_wordcloud(negative_text, 'Negative Keywords', color_map='Reds')

    st.markdown("---")
    
    # -------------------------------------------------------------------------
    # ROW 3: Model Performance Report
    # -------------------------------------------------------------------------
    st.subheader("6. Model Performance Summary")
    st.info("The trained Logistic Regression model is highly effective and balanced.")
    
    # Corrected Markdown block for better rendering
    st.markdown("""
    The final model metrics are:
    
    * Overall Accuracy: $\mathbf{90\%}$
    * Negative Recall (Minority Class): $\mathbf{0.72}$ 
    * Positive Recall (Majority Class): $\mathbf{0.92}$
    
    This balanced performance ensures that the model is reliable and avoids **overfitting to the majority class**, a critical success for this type of imbalanced text data.
    """)