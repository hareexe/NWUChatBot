import streamlit as st
import nltk
import os
import json
import random

from nltk.tokenize import PunktSentenceTokenizer, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


@st.cache_resource(show_spinner="Initializing NLTK resources...")
def initialize_nltk_data():
    """
    Sets up a reliable NLTK data path and downloads required resources.
    This function is run only once thanks to @st.cache_resource.
    It returns the initialized resources (lemmatizer, stopwords, and tokenizer).
    """
    base_dir = os.path.abspath(os.path.dirname(__file__))
    NLTK_DATA_DIR = os.path.join(base_dir, ".nltk_data")
    
    if NLTK_DATA_DIR not in nltk.data.path:
        nltk.data.path.insert(0, NLTK_DATA_DIR)
        
    if not os.path.exists(NLTK_DATA_DIR):
        os.makedirs(NLTK_DATA_DIR, exist_ok=True)

    required_resources = ['punkt', 'stopwords', 'wordnet']
    
    for resource_name in required_resources:
        try:
            nltk.data.find(resource_name)
        except LookupError:
            print(f"Downloading NLTK resource: {resource_name}...")
            nltk.download(resource_name, download_dir=NLTK_DATA_DIR, quiet=True)
        except Exception as e:
            print(f"Error during NLTK resource setup for {resource_name}: {e}")

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    try:
        punkt_data_path = nltk.data.find('tokenizers/punkt/english.pickle')
        tokenizer = PunktSentenceTokenizer(punkt_data_path)
    except LookupError:
        tokenizer = PunktSentenceTokenizer()

    regexp_word_tokenizer = RegexpTokenizer(r'\w+')

    return lemmatizer, stop_words, tokenizer, regexp_word_tokenizer

lemmatizer, stop_words, tokenizer, regexp_word_tokenizer = initialize_nltk_data()

@st.cache_resource(show_spinner="Loading chatbot data...")
def load_data():
    """Loads the intents data from the JSON file."""
    
    base_dir = os.path.abspath(os.path.dirname(__file__))
    filepath_simple = os.path.join(base_dir, "intents.json")
    
    try:
        with open(filepath_simple, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"FATAL ERROR: The required data file 'intents.json' was not found.")
        st.info(f"The app looked in: {filepath_simple}")
        st.info("Please ensure 'intents.json' is committed and pushed to the same directory as 'app.py'.")
        return {"intents": []}
    except json.JSONDecodeError:
        st.error(f"FATAL ERROR: The file '{filepath_simple}' is not valid JSON.")
        st.info("Please check the file for syntax errors (commas, braces, quotes).")
        return {"intents": []}


intents_data = load_data()

def preprocess(text):
    """Tokenizes, lowercases, removes stopwords, and lemmatizes text."""

    words = regexp_word_tokenizer.tokenize(text.lower())
    
    final_tokens = []
    for word in words:

        if word not in stop_words:
            final_tokens.append(lemmatizer.lemmatize(word))
    
    return set(final_tokens)

def get_response_by_keywords(user_input, intents_data):
    """Finds the best response based on keyword matching score."""
    if not intents_data.get('intents'):
        return "Chatbot data is currently unavailable. Please contact the developer."
        
    user_tokens = preprocess(user_input)

    best_intent = None
    best_score = 0

    if not user_tokens:
        return "Please try asking a more detailed question."

    for intent in intents_data['intents']:
        for pattern in intent['patterns']:
            pattern_tokens = preprocess(pattern)
            
            score = len(user_tokens.intersection(pattern_tokens))
            
            if score > best_score:
                best_score = score
                best_intent = intent

    if best_intent and best_score > 0:
        return random.choice(best_intent['responses'])
    else:
        return "I'm sorry, I don't have information on that specific topic for Northwestern University."


st.title("Northwestern University History Chatbot")
st.subheader("Ask questions about the founding, courses, and history.")

if 'history' not in st.session_state:
    st.session_state['history'] = [
        {"role": "assistant", "content": "Hello! I can answer questions about Northwestern University's history. Try asking, 'When was the university founded?'"}
    ]

for message in st.session_state['history']:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_prompt = st.chat_input("Ask a question...")

if user_prompt:
    st.session_state['history'].append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)

    with st.spinner('Checking data...'):
        bot_response = get_response_by_keywords(user_prompt, intents_data)
        
    st.session_state['history'].append({"role": "assistant", "content": bot_response})
    with st.chat_message("assistant"):
        st.write(bot_response)
