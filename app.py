import streamlit as st
import nltk
import os
import json
import random
import torch
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


@st.cache_resource(show_spinner="Loading sentence transformer model...")
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')


model = load_embedding_model()

def preprocess(text):
    words = regexp_word_tokenizer.tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(tokens)

@st.cache_data
def get_all_patterns(intents_data):
 
    patterns = set()
    EXCLUDED_TAGS = {"end_chat", "thank_you", "greeting"}
    
    for intent in intents_data.get("intents", []):
        tag = intent.get("tag")
        if tag not in EXCLUDED_TAGS:
            patterns.update(intent.get("patterns", []))

    pattern_list = list(patterns)
    random.shuffle(pattern_list)
    
    return pattern_list[:8]  


@st.cache_resource(show_spinner="Encoding all chatbot patterns...")
def build_intent_embeddings(intents_data):
    all_texts = []
    meta_data = []  

    for intent in intents_data.get("intents", []):
        tag = intent.get("tag")
        responses = intent.get("responses", [])
        for pattern in intent.get("patterns", []):
            processed = preprocess(pattern)
            if processed.strip():
                all_texts.append(processed)
                meta_data.append((tag, responses, pattern))

    if not all_texts:
        return [], [], None

    embeddings = model.encode(all_texts, convert_to_tensor=True)
    return all_texts, embeddings, meta_data


all_pattern_texts, pattern_embeddings, pattern_meta = build_intent_embeddings(intents_data)


def get_semantic_response(user_input):
    if not intents_data.get("intents"):
        return "Chatbot data is unavailable. Please check intents.json."

    user_processed = preprocess(user_input)

    # Handle empty or non-informative input
    if not user_processed.strip():
        return "Could you please rephrase that question about Northwestern University?"

    user_embedding = model.encode([user_processed], convert_to_tensor=True)
    similarities = util.cos_sim(user_embedding, pattern_embeddings)[0]

    best_index = similarities.argmax().item()
    best_score = similarities[best_index].item()
    best_tag, responses, original_pattern = pattern_meta[best_index]

    CONFIDENCE_THRESHOLD = 0.60 

    if best_score < CONFIDENCE_THRESHOLD:
        st.session_state['last_intent'] = None
        return "Hmm, I’m not entirely sure about that topic. Could you rephrase your question about Northwestern University?"

    best_response = random.choice(responses)
    st.session_state['last_intent'] = best_tag
    return best_response


st.title("NWU History Chatbot")
st.subheader("Northwestern University's history chatmate")

if 'history' not in st.session_state:
    st.session_state['history'] = [
        {"role": "assistant", "content": "Hello! I can answer questions about Northwestern University."}
    ]

suggestions = get_all_patterns(intents_data)

st.markdown("Try asking questions like:")
suggestion_markdown = "\n".join([f"* {s}" for s in suggestions])
st.markdown(suggestion_markdown)

if 'last_intent' not in st.session_state:
    st.session_state['last_intent'] = None

# Display conversation history
for msg in st.session_state['history']:
    with st.chat_message(msg["role"], avatar=None): 
        st.write(msg["content"])

# ---  FOOTER   ---
st.markdown("""
---
<p style='font-size: 0.75rem; color: #808080;'>
Source Information: Northwestern University Portal, and the book: LEGACY, The people, events, ideas and amazing faith that built Northwestern University by Erlinda Magbual-Gloria.
</p>
""", unsafe_allow_html=True)

# Input handling
user_prompt = st.chat_input("Ask something...")

if user_prompt:
    st.session_state['history'].append({"role": "user", "content": user_prompt})
    with st.chat_message("user", avatar=None):
        st.write(user_prompt)

    with st.spinner("Thinking..."):
        bot_reply = get_semantic_response(user_prompt)

    st.session_state['history'].append({"role": "assistant", "content": bot_reply})
    with st.chat_message("assistant", avatar=None):
        st.write(bot_reply)
