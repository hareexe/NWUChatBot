import os
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

lemmatizer = None
stop_words = None
regexp_word_tokenizer = RegexpTokenizer(r'\w+')

# --------------------------
# --- NLTK Initialization ---
# --------------------------
def initialize_nltk_data():
    global lemmatizer, stop_words, regexp_word_tokenizer

    base_dir = os.path.abspath(os.path.dirname(__file__))
    NLTK_DATA_DIR = os.path.join(base_dir, ".nltk_data")

    if NLTK_DATA_DIR not in nltk.data.path:
        nltk.data.path.insert(0, NLTK_DATA_DIR)

    os.makedirs(NLTK_DATA_DIR, exist_ok=True)

    for r in ['punkt', 'stopwords', 'wordnet', 'omw-1.4']:
        try:
            nltk.data.find(r)
        except LookupError:
            nltk.download(r, download_dir=NLTK_DATA_DIR, quiet=True)

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    regexp_word_tokenizer = RegexpTokenizer(r'\w+')

initialize_nltk_data()

# --------------------------
# --- Preprocessing -------
# --------------------------
def preprocess(text: str) -> str:
    if not text:
        return ""
    text = text.lower().strip()
    tokens = regexp_word_tokenizer.tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return " ".join(tokens)

# --------------------------
# --- Synonym expansion ---
# --------------------------
def expand_with_synonyms(text: str, max_synonyms_per_token: int = 2) -> str:
    tokens = regexp_word_tokenizer.tokenize(text.lower())
    expanded_tokens = list(tokens)

    for token in tokens:
        try:
            synsets = wordnet.synsets(token)
        except Exception:
            synsets = []
        added = 0
        for s in synsets:
            for lemma in s.lemmas():
                name = lemma.name().replace('_', ' ').lower()
                if name != token and name.isalpha():
                    expanded_tokens.append(name)
                    added += 1
                    if added >= max_synonyms_per_token:
                        break
            if added >= max_synonyms_per_token:
                break

    seen = set()
    final = [t for t in expanded_tokens if not (t in seen or seen.add(t))]
    return " ".join(final)