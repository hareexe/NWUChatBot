import streamlit as st
import nltk
import os
import json
import random
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, util

@st.cache_resource(show_spinner="Initializing NLTK resources...")
def initialize_nltk_data():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    NLTK_DATA_DIR = os.path.join(base_dir, ".nltk_data")

    if NLTK_DATA_DIR not in nltk.data.path:
        nltk.data.path.insert(0, NLTK_DATA_DIR)

    os.makedirs(NLTK_DATA_DIR, exist_ok=True)

    for r in ['punkt', 'stopwords', 'wordnet']:
        try:
            nltk.data.find(r)
        except LookupError:
            nltk.download(r, download_dir=NLTK_DATA_DIR, quiet=True)

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    regexp_word_tokenizer = RegexpTokenizer(r'\w+')

    return lemmatizer, stop_words, regexp_word_tokenizer


lemmatizer, stop_words, regexp_word_tokenizer = initialize_nltk_data()


@st.cache_resource(show_spinner="Loading chatbot data...")
def load_data():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    filepath = os.path.join(base_dir, "intents.json")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading intents.json: {e}")
        return {"intents": []}


intents_data = load_data()


@st.cache_resource(show_spinner="Loading AI model (this may take a few seconds)...")
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')


model = load_embedding_model()

def preprocess(text):
    words = regexp_word_tokenizer.tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(tokens)


def get_semantic_response(user_input, intents_data, context=None):
    if not intents_data.get('intents'):
        return "Chatbot data is unavailable. Please contact the developer."

    if context:
        user_input = context + " " + user_input  

    user_processed = preprocess(user_input)

    all_patterns, all_responses, all_tags = [], [], []
    for intent in intents_data['intents']:
        for pattern in intent['patterns']:
            all_patterns.append(preprocess(pattern))
            all_responses.append(intent['responses'])
            all_tags.append(intent['tag'])

    pattern_embeddings = model.encode(all_patterns, convert_to_tensor=True)
    user_embedding = model.encode([user_processed], convert_to_tensor=True)

    similarities = util.cos_sim(user_embedding, pattern_embeddings)[0]
    best_index = similarities.argmax().item()
    best_score = similarities[best_index].item()

    if best_score < 0.45:
        return "I'm not sure about that topic related to Northwestern University."

    best_tag = all_tags[best_index]
    best_response = random.choice(all_responses[best_index])

    st.session_state['last_intent'] = best_tag

    return best_response

st.title("NWU History Chatbot")
st.subheader("Ask questions about Northwestern University's history, founders, and more!")

if 'history' not in st.session_state:
    st.session_state['history'] = [
        {"role": "assistant", "content": "Hello! I can answer questions about Northwestern University. Try asking, 'When was NWU founded?'"}
    ]
if 'last_intent' not in st.session_state:
    st.session_state['last_intent'] = None

for msg in st.session_state['history']:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_prompt = st.chat_input("Ask something...")

if user_prompt:
    st.session_state['history'].append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)

    context = st.session_state.get('last_intent')
    with st.spinner("Thinking..."):
        bot_reply = get_semantic_response(user_prompt, intents_data, context)

    st.session_state['history'].append({"role": "assistant", "content": bot_reply})
    with st.chat_message("assistant"):
        st.write(bot_reply)
