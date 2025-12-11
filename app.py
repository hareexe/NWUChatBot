import streamlit as st
import json
import random
import torch
import hashlib
import os

# --- Imports from Modules ---
from modules.nlp_utils import initialize_nltk_data, preprocess, expand_with_synonyms, regexp_word_tokenizer
from modules.data_store import load_data, build_intent_embeddings, _hash_intents, load_embedding_model
from modules.eval_utils import build_all_tests_from_intents, run_offline_eval
from modules.matcher import get_semantic_response_debug, keyword_fallback, get_all_patterns, set_runtime_handles 


# --- NLTK Initialization ---
@st.cache_resource(show_spinner="Initializing NLTK resources...")
def _initialize_nltk_data():
    return initialize_nltk_data()

# Safely handle return value from NLTK init
_nltk_init = _initialize_nltk_data()
try:
    lemmatizer, stop_words, regexp_word_tokenizer = _nltk_init  # expect a tuple
except Exception:
    # Fallback: use the tokenizer exported by modules.nlp_utils; lemmatizer/stop_words optional
    lemmatizer, stop_words = None, None
    from modules.nlp_utils import regexp_word_tokenizer as _exported_tokenizer
    regexp_word_tokenizer = _exported_tokenizer

# --- Load intents.json ----
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

# --- Embedding model  -----
# Note: _load_embedding_model() just calls the imported function and applies @st.cache_resource
@st.cache_resource(show_spinner="Loading sentence transformer model...")
def _load_embedding_model():
    return load_embedding_model()

model = _load_embedding_model()

# --- Build embeddings ----
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
from modules.matcher import set_runtime_handles
set_runtime_handles(
    model=model,
    intents_data=intents_data,
    pattern_embeddings=pattern_embeddings,
    pattern_meta=pattern_meta,
    preprocess=preprocess,
    tokenizer=regexp_word_tokenizer
)

# --- Streamlit UI ---------
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
EXCLUDED_TAGS = ["greeting", "end_chat"] 

try:
    # Use the version of get_all_patterns with the exclusion list
    suggestions = get_all_patterns(intents_data, limit=5, exclude_tags=EXCLUDED_TAGS)
except TypeError:
    # Fallback for compatibility if get_all_patterns doesn't yet support exclude_tags
    st.warning("`get_all_patterns` is missing the `exclude_tags` parameter. Please update `modules/matcher.py` for correct filtering.")
    suggestions = get_all_patterns(intents_data, limit=5)

if suggestions:
    st.markdown("**Try asking:**")
    st.markdown("\n".join([f"* {s}" for s in suggestions]))
    

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
    # --- Start of user message handling ---
    st.session_state['history'].append({"role": "user", "content": user_prompt})
    st.session_state['recent_questions'].append(user_prompt)
    st.session_state['recent_questions'] = st.session_state['recent_questions'][-6:]

    # Display user message in the current chat area
    with st.chat_message("user", avatar=None):
        st.write(user_prompt)

    # --- Start of model processing ---
    with st.spinner("Thinking..."):
        bot_reply, debug_info = get_semantic_response_debug(user_prompt)

    # Display bot response
    st.session_state['history'].append({"role": "assistant", "content": bot_reply})
    with st.chat_message("assistant", avatar=None):
        st.write(bot_reply)
