import streamlit as st
import nltk
import os
import json
import random

from nltk.tokenize import PunktSentenceTokenizer, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_resource(show_spinner="Initializing NLTK resources...")
def initialize_nltk_data():
    """
    Sets up a reliable NLTK data path and downloads required resources.
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
        with open(filepath_simple, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("FATAL ERROR: 'intents.json' not found.")
        st.info(f"Expected at: {filepath_simple}")
        return {"intents": []}
    except json.JSONDecodeError:
        st.error(f"FATAL ERROR: '{filepath_simple}' is not valid JSON.")
        return {"intents": []}


intents_data = load_data()


def preprocess(text):
    """Tokenizes, lowercases, removes stopwords, and lemmatizes text."""
    words = regexp_word_tokenizer.tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(tokens)


def get_response_by_keywords(user_input, intents_data):
    """
    Uses TF-IDF + cosine similarity for smarter pattern matching.
    """
    if not intents_data.get('intents'):
        return "Chatbot data is currently unavailable. Please contact the developer."

    user_processed = preprocess(user_input)

    if not user_processed.strip():
        return "Please try asking a more detailed question."


    all_patterns = []
    intent_tags = []
    for intent in intents_data['intents']:
        for pattern in intent['patterns']:
            all_patterns.append(preprocess(pattern))
            intent_tags.append(intent['tag'])

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(all_patterns + [user_processed])
    user_vector = vectors[-1]
    pattern_vectors = vectors[:-1]

    similarities = cosine_similarity(user_vector, pattern_vectors).flatten()

    if max(similarities) == 0:
        return "I'm sorry, I don't have information on that specific topic for Northwestern University."

    best_index = similarities.argmax()
    best_tag = intent_tags[best_index]

    best_intent = next((i for i in intents_data['intents'] if i['tag'] == best_tag), None)

    if best_intent:
        return random.choice(best_intent['responses'])
    else:
        return "I'm sorry, I don't have information on that specific topic for Northwestern University."


st.title("NWU History Chatbot")
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
