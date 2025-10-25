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
    except FileNotFoundError:
        st.error(f"Error: intents.json not found at {filepath}. Please ensure it is present.")
        return {"intents": []}
    except Exception as e:
        st.error(f"Error loading intents.json: {e}")
        return {"intents": []}


intents_data = load_data()


@st.cache_resource(show_spinner="Loading model (this may take a few seconds)...")
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_embedding_model()

def preprocess(text):
    """Tokenize, lowercase, remove stop words, and lemmatize the input text."""
    words = regexp_word_tokenizer.tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(tokens)


def get_semantic_response(user_input, intents_data):
    """
    Finds the best matching intent (tag) using semantic similarity (cosine similarity) 
    based on averaged pattern embeddings for each tag.
    """
    if not intents_data.get('intents'):
        return "Chatbot data is unavailable. Please contact the developer or check intents.json."

    user_processed = preprocess(user_input)
    
    if len(user_processed.split()) < 1 and len(user_input.strip()) > 0:
        return "That's interesting! Could you elaborate on that, or ask a question about NWU?"

    tag_info = {}
    all_patterns = []
    
    for intent in intents_data['intents']:
        tag = intent['tag']
        tag_info[tag] = {
            'patterns': [],
            'responses': intent['responses']
        }
        for pattern in intent['patterns']:
            processed_pattern = preprocess(pattern)
            if processed_pattern:
                tag_info[tag]['patterns'].append(processed_pattern)
                all_patterns.append((processed_pattern, tag))

    if not all_patterns:
        return "No usable patterns found in the chatbot data."

    pattern_to_tag_map = {p: tag for p, tag in all_patterns}
    all_pattern_texts = [p for p, tag in all_patterns]
    
    pattern_embeddings = model.encode(all_pattern_texts, convert_to_tensor=True)

    intent_embeddings = []
    tag_list = []
    
    for tag in tag_info:
        tag_patterns_embeddings = [
            pattern_embeddings[i] for i, p in enumerate(all_pattern_texts) 
            if pattern_to_tag_map[p] == tag
        ]
        
        if tag_patterns_embeddings:
            avg_embedding = torch.stack(tag_patterns_embeddings).mean(dim=0)
            intent_embeddings.append(avg_embedding)
            tag_list.append(tag)
            
    if not intent_embeddings:
        return "Could not compute intent embeddings."

    intent_embeddings_tensor = torch.stack(intent_embeddings)

    user_embedding = model.encode([user_processed], convert_to_tensor=True)

    similarities = util.cos_sim(user_embedding, intent_embeddings_tensor)[0]
    best_index = similarities.argmax().item()
    best_score = similarities[best_index].item()

    if best_score < 0.66:
        st.session_state['last_intent'] = None
        return "I'm not sure about that topic related to Northwestern University. Can you rephrase your question?"

    best_tag = tag_list[best_index]
    best_response = random.choice(tag_info[best_tag]['responses'])

    st.session_state['last_intent'] = best_tag

    return best_response

st.title("NWU History Chatbot")
st.subheader("Northwestern University's history chatmate")

if 'history' not in st.session_state:
    st.session_state['history'] = [
        {"role": "assistant", "content": "Hello! I can answer questions about Northwestern University. Try asking, 'When was NWU founded?'"}
    ]
if 'last_intent' not in st.session_state:
    st.session_state['last_intent'] = None

for msg in st.session_state['history']:
    with st.chat_message(msg["role"], avatar=None):
        st.write(msg["content"])

user_prompt = st.chat_input("Ask something...")

if user_prompt:
    st.session_state['history'].append({"role": "user", "content": user_prompt})
    with st.chat_message("user", avatar=None):
        st.write(user_prompt)

    with st.spinner("Thinking..."):
        bot_reply = get_semantic_response(user_prompt, intents_data)

    st.session_state['history'].append({"role": "assistant", "content": bot_reply})
    with st.chat_message("assistant", avatar=None):
        st.write(bot_reply)

