# NWU History Chatbot

A conversational AI chatbot built with Streamlit that answers questions about the history of Northwestern University (Laoag City). It uses intent matching, semantic embeddings, and NLP preprocessing to provide accurate and context-aware responses.

---

## Table of Contents

- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Main Components](#main-components)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)
- [Credits](#credits)

---

## Project Structure

```
NWUChatBot/
│
├── app.py                  # Main Streamlit app
├── intents.json            # Chatbot intents, patterns, and responses
├── requirements.txt        # Python dependencies
├── README.md               # This documentation
│
├── modules/                # Custom Python modules
│   ├── data_store.py       # Data loading, embedding, and model utilities
│   ├── detectors.py        # (Optional) Intent detectors
│   ├── eval_utils.py       # Evaluation and testing utilities
│   ├── matcher.py          # Intent matching and response logic
│   └── nlp_utils.py        # NLP preprocessing utilities
│
└── .nltk_data/             # Local NLTK data (stopwords, tokenizers, etc)
```

---

## How It Works

### 1. NLTK Initialization

- Loads NLTK resources (lemmatizer, stopwords, tokenizer) using `initialize_nltk_data` from `modules/nlp_utils.py`.
- Ensures all NLP preprocessing (tokenization, stopword removal, lemmatization) works offline via `.nltk_data/`.

### 2. Loading Intents

- Loads chatbot intents from `intents.json` using `load_data` from `modules/data_store.py`.
- Fallback: Loads directly from file if the module fails.

### 3. Embedding Model

- Loads a sentence transformer model for semantic similarity via `load_embedding_model`.
- Caches the model for performance.

### 4. Intent Embeddings

- Encodes all patterns from intents into embeddings using `build_intent_embeddings`.
- Uses a hash of the intents for cache invalidation and caching.

### 5. Matcher Setup

- Passes model, embeddings, and NLP utilities to `set_runtime_handles` in `modules/matcher.py` for use in matching.

### 6. Streamlit UI

- Displays conversation history and suggestions.
- Handles user input and displays bot responses.
- Optionally, can run offline evaluation via `run_offline_eval`.

---

## Main Components

### Intents

- Defined in `intents.json`.
- Each intent has:
  - `tag`: Unique identifier.
  - `patterns`: Example user queries.
  - `responses`: Possible bot replies.

### NLP Utilities (`modules/nlp_utils.py`)

- `preprocess`: Cleans and tokenizes user input.
- `expand_with_synonyms`: (Optional) Expands queries with synonyms.
- `regexp_word_tokenizer`: Tokenizes text using regex.

### Data Store (`modules/data_store.py`)

- `load_data`: Loads intents.
- `build_intent_embeddings`: Encodes patterns.
- `load_embedding_model`: Loads transformer model.
- `_hash_intents`: Hashes intents for caching.

### Matcher (`modules/matcher.py`)

- `get_semantic_response_debug`: Finds best-matching intent and generates a response (with debug info).
- `keyword_fallback`: Fallback for unmatched queries.
- `get_all_patterns`: Returns sample patterns for suggestions.
- `set_runtime_handles`: Dependency injection for matcher logic.

### Evaluation (`modules/eval_utils.py`)

- `run_offline_eval`: Tests chatbot accuracy on all patterns.

---

## Setup & Installation

1. **Clone the repository:**
    ```sh
    git clone <your-repo-url>
    cd NWUChatBot
    ```

2. **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

3. **Ensure NLTK data is present:**
    - The `.nltk_data/` directory should be in the project root (already included).
    - If missing, download required corpora:
      ```sh
      python -m nltk.downloader stopwords punkt wordnet -d .nltk_data
      ```

---

## Usage

1. **Start the chatbot:**
    ```sh
    streamlit run app.py
    ```

2. **Interact via the web UI:**
    - Ask questions about NWU history.
    - Try suggested sample questions.
    - View conversation history and bot responses.

---

## Customization

- **Add/Edit Intents:**  
  Modify `intents.json` to add new questions and answers. Each intent should have a unique `tag`, a list of `patterns` (user queries), and a list of `responses` (bot replies).

- **NLP/Matching Logic:**  
  Tweak code in `modules/nlp_utils.py` and `modules/matcher.py` for advanced behavior, such as custom preprocessing, synonym expansion, or new matching strategies.

- **Evaluation:**  
  Uncomment the evaluation section in `app.py` to run quick accuracy tests on all patterns.

---

## Troubleshooting

- **Missing NLTK Data:**  
  If you see errors about missing NLTK data, ensure `.nltk_data/` is in the project root and contains required corpora/tokenizers.

- **Model Loading Issues:**  
  If the model fails to load, check your internet connection (for first-time downloads) or ensure the model is cached.

- **Streamlit Issues:**  
  Make sure you are running a compatible version of Streamlit as specified in `requirements.txt`.

---

## Credits

- **Data sources:** Northwestern University Portal, *LEGACY* by Erlinda Magbual-Gloria.
- **Libraries:** [Streamlit](https://streamlit.io/), [NLTK](https://www.nltk.org/), [Sentence Transformers](https://www.sbert.net/).
- **Author:** [Your Name or Team]

---