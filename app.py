import streamlit as st
import nltk
import os
import json
import random
# Using RegexpTokenizer instead of the problematic word_tokenize
from nltk.tokenize import PunktSentenceTokenizer, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- 1. NLTK Setup and Caching (Crucial for Deployment) ---

@st.cache_resource(show_spinner="Initializing NLTK resources...")
def initialize_nltk_data():
    """
    Sets up a reliable NLTK data path and downloads required resources.
    This function is run only once thanks to @st.cache_resource.
    It returns the initialized resources (lemmatizer, stopwords, and tokenizer).
    """
    # FIX: Use os.path.abspath for the most reliable path resolution
    base_dir = os.path.abspath(os.path.dirname(__file__))
    NLTK_DATA_DIR = os.path.join(base_dir, ".nltk_data")
    
    # Ensure this path is available to NLTK
    if NLTK_DATA_DIR not in nltk.data.path:
        nltk.data.path.insert(0, NLTK_DATA_DIR)

    # Explicitly create the directory if it doesn't exist
    if not os.path.exists(NLTK_DATA_DIR):
        os.makedirs(NLTK_DATA_DIR, exist_ok=True)

    # --- Robust Download Logic ---
    required_resources = ['punkt', 'stopwords', 'wordnet']
    
    for resource_name in required_resources:
        try:
            # Try to find the resource
            nltk.data.find(resource_name)
        except LookupError:
            # If not found, download it directly to the specific path
            print(f"Downloading NLTK resource: {resource_name}...")
            # Use download_dir to ensure it goes where we told NLTK to look
            nltk.download(resource_name, download_dir=NLTK_DATA_DIR, quiet=True)
        except Exception as e:
            print(f"Error during NLTK resource setup for {resource_name}: {e}")
            
    # Initialize expensive resources once
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # FIX: Explicitly load the Punkt Tokenizer data to avoid the LookupError
    # We pass the loaded data object to the initializer
    try:
        # Load the raw Punkt data for the English language
        punkt_data_path = nltk.data.find('tokenizers/punkt/english.pickle')
        tokenizer = PunktSentenceTokenizer(punkt_data_path)
    except LookupError:
        # Fallback if the above path still fails (shouldn't happen with the fixes above)
        tokenizer = PunktSentenceTokenizer()

    # Initialize a RegexpTokenizer that splits on non-word characters
    regexp_word_tokenizer = RegexpTokenizer(r'\w+')

    return lemmatizer, stop_words, tokenizer, regexp_word_tokenizer

# Initialize NLTK data and get resources
lemmatizer, stop_words, tokenizer, regexp_word_tokenizer = initialize_nltk_data()

# --- 2. Data Loading ---

@st.cache_resource(show_spinner="Loading chatbot data...")
def load_data(): # Removed default filepath, using dynamic path below
    """Loads the intents data from the JSON file."""
    
    # FIX: Use os.path.abspath for reliable path to intents.json
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


# --- 3. Core Chatbot Logic ---

def preprocess(text):
    """Tokenizes, lowercases, removes stopwords, and lemmatizes text."""
    # We no longer use tokenizer.tokenize() because we only need word tokens, not sentence tokens.
    # The direct word tokenization call should be the most resilient one.
    
    # FIX: Use the resilient RegexpTokenizer to tokenize words directly
    words = regexp_word_tokenizer.tokenize(text.lower())
    
    # Apply filtering and lemmatization
    final_tokens = []
    for word in words:
        # word.isalpha() check is implicitly handled by RegexpTokenizer(r'\w+'),
        # but we keep the stop_words and lemmatization.
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


# --- 4. Streamlit UI and Session Management ---

st.title("Northwestern University History Chatbot")
st.subheader("Ask questions about the founding, courses, and history.")

# Initialize chat history
if 'history' not in st.session_state:
    st.session_state['history'] = [
        {"role": "assistant", "content": "Hello! I can answer questions about Northwestern University's history. Try asking, 'When was the university founded?'"}
    ]

# Display conversation history
for message in st.session_state['history']:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Handle user input
user_prompt = st.chat_input("Ask a question...")

if user_prompt:
    # 1. Add user message to history and display it
    st.session_state['history'].append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)

    # 2. Get bot response
    with st.spinner('Checking data...'):
        bot_response = get_response_by_keywords(user_prompt, intents_data)
        
    # 3. Add bot response to history and display it
    st.session_state['history'].append({"role": "assistant", "content": bot_response})
    with st.chat_message("assistant"):
        st.write(bot_response)
