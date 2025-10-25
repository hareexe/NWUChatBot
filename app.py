import streamlit as st
import nltk
import os
import json
import random
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, util

# --- NLTK and Resource Initialization ---

# Use st.cache_resource for heavy initialization tasks
@st.cache_resource(show_spinner="Initializing NLTK resources...")
def initialize_nltk_data():
    # Setup NLTK data path (important for deployment environments)
    base_dir = os.path.abspath(os.path.dirname(__file__))
    NLTK_DATA_DIR = os.path.join(base_dir, ".nltk_data")

    if NLTK_DATA_DIR not in nltk.data.path:
        nltk.data.path.insert(0, NLTK_DATA_DIR)

    os.makedirs(NLTK_DATA_DIR, exist_ok=True)

    # Download required NLTK data if missing
    for r in ['punkt', 'stopwords', 'wordnet']:
        try:
            nltk.data.find(r)
        except LookupError:
            # Download quietly to avoid excessive console output
            nltk.download(r, download_dir=NLTK_DATA_DIR, quiet=True)

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    # Tokenizer to only capture words (alphanumeric)
    regexp_word_tokenizer = RegexpTokenizer(r'\w+')

    return lemmatizer, stop_words, regexp_word_tokenizer

# Initialize global NLTK tools
lemmatizer, stop_words, regexp_word_tokenizer = initialize_nltk_data()


@st.cache_resource(show_spinner="Loading chatbot data...")
def load_data():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    # Assuming intents.json is in the same directory
    filepath = os.path.join(base_dir, "intents.json")

    try:
        # NOTE: In a real environment, you need to ensure intents.json is available
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Error: intents.json not found at {filepath}. Please ensure it is present.")
        return {"intents": []}
    except Exception as e:
        st.error(f"Error loading intents.json: {e}")
        return {"intents": []}


# Load the intents data
intents_data = load_data()


@st.cache_resource(show_spinner="Loading AI model (this may take a few seconds)...")
def load_embedding_model():
    # 'all-MiniLM-L6-v2' is a good, fast general-purpose model
    return SentenceTransformer('all-MiniLM-L6-v2')


# Load the Sentence Transformer model
model = load_embedding_model()


# --- Core Logic Functions ---

def preprocess(text):
    """Tokenize, lowercase, remove stop words, and lemmatize the input text."""
    words = regexp_word_tokenizer.tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(tokens)


def get_semantic_response(user_input, intents_data):
    """
    Finds the best matching intent pattern using semantic similarity (cosine similarity).
    Context injection logic has been removed to prevent repetition for unrelated inputs.
    """
    if not intents_data.get('intents'):
        return "Chatbot data is unavailable. Please contact the developer or check intents.json."

    # Process user input without context
    user_processed = preprocess(user_input)
    
    # If the user input is too short after preprocessing, it's probably irrelevant
    if len(user_processed.split()) < 1 and len(user_input.strip()) > 0:
        return "That's interesting! Could you elaborate on that, or ask a question about NWU?"

    all_patterns, all_responses, all_tags = [], [], []
    for intent in intents_data['intents']:
        for pattern in intent['patterns']:
            # Only use patterns that result in non-empty preprocessed strings
            processed_pattern = preprocess(pattern)
            if processed_pattern:
                all_patterns.append(processed_pattern)
                all_responses.append(intent['responses'])
                all_tags.append(intent['tag'])
            
    # Handle case where no valid patterns were loaded (e.g., all patterns are stop words)
    if not all_patterns:
        return "No usable patterns found in the chatbot data."

    # Calculate embeddings
    pattern_embeddings = model.encode(all_patterns, convert_to_tensor=True)
    user_embedding = model.encode([user_processed], convert_to_tensor=True)

    # Calculate similarity
    similarities = util.cos_sim(user_embedding, pattern_embeddings)[0]
    best_index = similarities.argmax().item()
    best_score = similarities[best_index].item()

    # MODIFIED: Increased threshold from 0.45 to 0.55 to demand a better semantic match
    # This reduces the chance of vague inputs matching a random intent.
    if best_score < 0.55:
        # Reset last_intent if the match is too poor
        st.session_state['last_intent'] = None
        return "I'm not sure about that topic related to Northwestern University. Can you rephrase your question?"

    best_tag = all_tags[best_index]
    best_response = random.choice(all_responses[best_index])

    # Store the successfully matched intent for potential future context (though not currently used)
    st.session_state['last_intent'] = best_tag

    return best_response


# --- Streamlit UI ---

# Inject custom CSS for chat bubbles
st.markdown("""
<style>
/* Hide the default role header (User/Assistant) and avatars */
[data-testid^="stChatMessage"] > div:first-child {
    display: none;
}

/* Base styling for all message content boxes (the bubble itself) */
[data-testid="stChatMessageContent"] {
    max-width: 70%;
    padding: 12px 16px;
    border-radius: 20px;
    font-size: 16px;
    margin-top: 5px;
    margin-bottom: 5px; 
    line-height: 1.5;
    text-align: left; 
    box-shadow: 0 1px 1px rgba(0,0,0,0.1); 
}

/* Styles for User - Right/Blue bubble */
[data-testid^="stChatMessage"] [role="user"] ~ [data-testid="stChatMessageContent"] {
    background-color: #0056b3; /* Darker Blue for better contrast on dark themes */
    color: white;
    border-top-right-radius: 5px; /* Pointy corner */
    margin-left: auto; 
    margin-right: 0;
}

/* Styles for Assistant (Bot) - Left/Grey bubble */
[data-testid^="stChatMessage"] [role="assistant"] ~ [data-testid="stChatMessageContent"] {
    background-color: #dcdcdc; /* Slightly darker grey for visibility on light themes */
    color: #333; /* Dark text for contrast */
    border-top-left-radius: 5px; /* Pointy corner */
    margin-right: auto;
    margin-left: 0; 
}
</style>
""", unsafe_allow_html=True)


st.title("NWU History Chatbot")
st.subheader("Ask questions about Northwestern University's history, founders, and more!")

# Initialize session state variables
if 'history' not in st.session_state:
    st.session_state['history'] = [
        {"role": "assistant", "content": "Hello! I can answer questions about Northwestern University. Try asking, 'When was NWU founded?'"}
    ]
if 'last_intent' not in st.session_state:
    st.session_state['last_intent'] = None


# Display chat history
for msg in st.session_state['history']:
    # Set avatar=None to explicitly hide the icon/avatar
    with st.chat_message(msg["role"], avatar=None):
        st.write(msg["content"])

# Get user input
user_prompt = st.chat_input("Ask something...")

if user_prompt:
    st.session_state['history'].append({"role": "user", "content": user_prompt})
    # Set avatar=None to explicitly hide the icon/avatar
    with st.chat_message("user", avatar=None):
        st.write(user_prompt)

    # REMOVED context retrieval as we are no longer using it for simple intent matching
    with st.spinner("Thinking..."):
        # MODIFIED: Removed the context argument from the function call
        bot_reply = get_semantic_response(user_prompt, intents_data)

    st.session_state['history'].append({"role": "assistant", "content": bot_reply})
    # Set avatar=None to explicitly hide the icon/avatar
    with st.chat_message("assistant", avatar=None):
        st.write(bot_reply)
