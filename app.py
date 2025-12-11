import streamlit as st
import json
import random
import torch
import hashlib
import os

from modules.nlp_utils import initialize_nltk_data, preprocess, expand_with_synonyms, regexp_word_tokenizer
from modules.data_store import load_data, build_intent_embeddings, _hash_intents, load_embedding_model
from modules.eval_utils import build_all_tests_from_intents, run_offline_eval
from modules.matcher import get_semantic_response_debug, keyword_fallback, get_all_patterns

# Mock NLP Utilities
def initialize_nltk_data():
    """MOCK: Returns mock lemmatizer, stop_words, and tokenizer function."""
    def regexp_word_tokenizer(text):
        import re
        return re.findall(r'\b\w+\b', text.lower())
    return None, None, regexp_word_tokenizer

def preprocess(text, lemmatizer=None, stop_words=None):
    """MOCK: Simple lowercasing and tokenization for preprocess."""
    return [word for word in regexp_word_tokenizer(text) if len(word) > 1]

# Mock Data Store
def load_data():
    """MOCK: Loads data from intents.json file path, expecting it to be in the app directory."""
    try:
        app_dir = os.path.abspath(os.path.dirname(__file__))
        intents_path = os.path.join(app_dir, "intents.json")
        with open(intents_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"intents": []}
    except Exception:
        return {"intents": []}

def load_embedding_model():
    """MOCK: Returns a placeholder for the Sentence Transformer Model."""
    class MockModel:
        def encode(self, sentences, convert_to_tensor=True):
            # Returns a random embedding for demonstration/testing
            if convert_to_tensor:
                return torch.randn(len(sentences), 768)
            return [[random.random()] * 768] * len(sentences)
    # Ensure torch is imported correctly
    return MockModel() if 'torch' in globals() else None

def _hash_intents(data):
    """MOCK: Creates a simple hash based on the serialized JSON content."""
    serialized = json.dumps(data, sort_keys=True)
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()

def build_intent_embeddings(intents_data_hash, intents_data_serialized, model):
    """MOCK: Simulates building embeddings and metadata."""
    data = json.loads(intents_data_serialized)
    all_pattern_texts = []
    pattern_meta = []
    
    for intent in data.get("intents", []):
        tag = intent.get("tag")
        for pattern in intent.get("patterns", []):
            all_pattern_texts.append(pattern)
            pattern_meta.append({"tag": tag, "response_index": 0}) 

    # Mock encoding
    if model:
        pattern_embeddings = model.encode(all_pattern_texts)
    else:
        # Fallback if no model is available (e.g., torch failed to load)
        pattern_embeddings = [ [random.random()] * 768 for _ in all_pattern_texts] 

    return all_pattern_texts, pattern_embeddings, pattern_meta

# Mock Matcher and Eval
_runtime_model = None
_runtime_intents = {"intents": []}
_runtime_embeddings = []
_runtime_meta = []
_runtime_preprocess = None
_runtime_tokenizer = None

def set_runtime_handles(model, intents_data, pattern_embeddings, pattern_meta, preprocess, tokenizer):
    """MOCK: Sets global variables for matcher functions."""
    global _runtime_model, _runtime_intents, _runtime_embeddings, _runtime_meta, _runtime_preprocess, _runtime_tokenizer
    _runtime_model = model
    _runtime_intents = intents_data
    _runtime_embeddings = pattern_embeddings
    _runtime_meta = pattern_meta
    _runtime_preprocess = preprocess
    _runtime_tokenizer = tokenizer

def _get_random_response(tag, intents_data):
    """Helper to get a random response from a tag."""
    for intent in intents_data.get("intents", []):
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "I'm sorry, I couldn't find a response for that."

def get_semantic_response_debug(query):
    """MOCK: Simulates semantic matching (finds the first intent as a mock match)."""
    
    intents = _runtime_intents.get("intents", [])
    if not intents:
        return "I'm sorry, my knowledge base is empty.", "Status: No Intents Loaded"

    # Mocking: Check for exact keyword match first (simple fallback)
    query_lower = query.lower()
    for intent in intents:
        for pattern in intent.get("patterns", []):
            if query_lower == pattern.lower():
                return _get_random_response(intent["tag"], _runtime_intents), f"Status: Exact Match on Tag '{intent['tag']}'"

    # Mocking: Semantic Match (just pick a random intent to simulate a successful match)
    if intents:
        matched_intent = random.choice(intents)
        if matched_intent:
            return _get_random_response(matched_intent["tag"], _runtime_intents), f"Status: Mock Semantic Match (Tag: '{matched_intent['tag']}')"

    return "I couldn't match your query to any historical data.", "Status: Fallback/No Match"

def get_all_patterns(intents_data, exclude_tags=None, limit=5):
    """MOCK: Collects a sample of patterns for suggestions."""
    all_patterns = []
    exclude_tags = exclude_tags or set()
    for intent in intents_data.get("intents", []):
        if intent["tag"] not in exclude_tags:
            all_patterns.extend(intent.get("examples", intent.get("patterns", [])))
    return random.sample(all_patterns, min(limit, len(all_patterns)))

# --------------------------
# --- NLTK Initialization ---
# --------------------------
@st.cache_resource(show_spinner="Initializing NLTK resources...")
def _initialize_nltk_data():
    return initialize_nltk_data()

# Safely handle return value from NLTK init
_nltk_init = _initialize_nltk_data()
try:
    # Expect a tuple: (lemmatizer, stop_words, tokenizer_function)
    lemmatizer, stop_words, regexp_word_tokenizer = _nltk_init 
except Exception:
    # Fallback: use the tokenizer exported by initialize_nltk_data or use a simple mock tokenizer
    lemmatizer, stop_words = None, None
    def regexp_word_tokenizer(text): # simple fallback definition
        import re
        return re.findall(r'\b\w+\b', text.lower())

# --------------------------
# --- Load intents.json ----
# --------------------------
@st.cache_resource(show_spinner="Loading chatbot data...")
def _load_data():
    return load_data()

intents_data = _load_data()

# Fallback: ensure intents.json is loaded from the app root if the module returned empty
if not intents_data or not intents_data.get("intents"):
    try:
        app_dir = os.path.abspath(os.path.dirname(__file__))
        intents_path = os.path.join(app_dir, "intents.json")
        with open(intents_path, "r", encoding="utf-8") as f:
            intents_data = json.load(f)
    except Exception as e:
        st.error(f"Failed to load intents.json from app root: {e}")
        intents_data = {"intents": []}

# --------------------------
# --- Embedding model  -----
# --------------------------
@st.cache_resource(show_spinner="Loading sentence transformer model...")
def _load_embedding_model():
    return load_embedding_model()

model = _load_embedding_model()

# --------------------------
# --- Build embeddings ----
# --------------------------
@st.cache_data(show_spinner="Encoding all chatbot patterns...")
def _build_intent_embeddings(intents_data_hash: str, intents_data_serialized: str):
    # pass the loaded model into the module function
    return build_intent_embeddings(intents_data_hash, intents_data_serialized, model=model)

# Ensure model is available before building embeddings
if model is None:
    model = _load_embedding_model()

intents_hash = _hash_intents(intents_data)
intents_serialized = json.dumps(intents_data, sort_keys=True)
all_pattern_texts, pattern_embeddings, pattern_meta = _build_intent_embeddings(intents_hash, intents_serialized)

# Wire globals to matcher (minimal DI)
set_runtime_handles(
    model=model,
    intents_data=intents_data,
    pattern_embeddings=pattern_embeddings,
    pattern_meta=pattern_meta,
    preprocess=preprocess,
    tokenizer=regexp_word_tokenizer
)

# --------------------------
# --- Streamlit UI ---------
# --------------------------
st.set_page_config(page_title="NWU History Chatbot", layout="centered")
st.title("NWU History Chatbot")
st.subheader("Ask about the history of Northwestern University (Laoag City)")

if 'history' not in st.session_state:
    st.session_state['history'] = [{"role": "assistant", "content": "Hello! I can answer questions about Northwestern University."}]
if 'last_intent' not in st.session_state:
    st.session_state['last_intent'] = None
if 'recent_questions' not in st.session_state:
    st.session_state['recent_questions'] = []

# Suggestions
# Tags excluded in the original request: "end_chat", "thank_you", "greeting"
EXCLUDED_TAGS = {"end_chat", "thank_you", "greeting"} 
patterns = get_all_patterns(intents_data, exclude_tags=EXCLUDED_TAGS, limit=5)

if patterns:
    st.markdown("**Try asking:**")
    st.markdown("\n".join([f"* {s}" for s in patterns]))

# --- Quick eval button (Commented out in original) ---
# col1, col2 = st.columns(2)
# with col2:
#     if st.button("Run quick evaluation"):
#         acc, res = run_offline_eval(intents_data)
#         st.markdown(f"<small>Accuracy: {round(acc*100,1)}%</small>", unsafe_allow_html=True)
#         # Show only misses
#         misses = [r for r in res if not r["ok"]]
#         for r in misses:
#             st.markdown(
#                 f"<small>- [MISS] {r['query']} → expected={r['expected']} predicted={r['predicted']} score={r['score']} reason={r['reason']}</small>",
#                 unsafe_allow_html=True
#             )

# Display conversation history
for msg in st.session_state['history']:
    with st.chat_message(msg["role"], avatar=None):
        st.write(msg["content"])

# Footer
st.markdown("""
---
<p style='font-size:0.75rem;color:#808080;'>
Source: Northwestern University Portal and <em>LEGACY</em> by Erlinda Magbual-Gloria.
</p>
""", unsafe_allow_html=True)

# Input
user_prompt = st.chat_input("Ask something about NWU history...")

if user_prompt:
    st.session_state['history'].append({"role": "user", "content": user_prompt})
    st.session_state['recent_questions'].append(user_prompt)
    st.session_state['recent_questions'] = st.session_state['recent_questions'][-6:]

    with st.chat_message("user", avatar=None):
        st.write(user_prompt)

    with st.spinner("Thinking..."):
        # This function performs semantic matching against all intents
        bot_reply, debug_info = get_semantic_response_debug(user_prompt)

    st.session_state['history'].append({"role": "assistant", "content": bot_reply})
    with st.chat_message("assistant", avatar=None):
        st.write(bot_reply)
        if debug_info:
            st.markdown(f"<small style='color:gray'>Debug Info: {debug_info}</small>", unsafe_allow_html=True)
