import json
import random
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

REQUIRED_NLTK_DATA = {
    'punkt': 'tokenizers/punkt',
    'stopwords': 'corpora/stopwords',
    'wordnet': 'corpora/wordnet'
}

for resource_name, resource_path in REQUIRED_NLTK_DATA.items():
    try:
        nltk.data.find(resource_path)
    except LookupError:
        print(f"Downloading NLTK resource: {resource_name}...")
        nltk.download(resource_name)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def load_data(filepath):
    """Loads the intents data from the JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"ERROR: The data file '{filepath}' was not found.")
        print("Please ensure 'intents.json' is in the correct directory ('ProjectChatBot').")
        return {"intents": []}
    except json.JSONDecodeError:
        print(f"ERROR: The file '{filepath}' is not valid JSON.")
        return {"intents": []}

INTENTS_FILEPATH = "ProjectChatBot/intents.json"

intents_data = load_data(INTENTS_FILEPATH)

def preprocess_nltk(text):
    """Tokenizes, lowercases, removes stopwords, and lemmatizes text using NLTK."""
    tokens = word_tokenize(text.lower())
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word.isalpha() and word not in stop_words
    ]
    return set(tokens)


def get_response_by_keywords(user_input, intents_data):
    """Finds the best response based on keyword matching score using NLTK preprocessing."""
    if not intents_data.get('intents'):
        return "Chatbot data is unavailable."
  
    user_keywords = preprocess_nltk(user_input)

    best_intent = None
    best_score = 0

    if not user_keywords:
        return "Please say something."

    for intent in intents_data['intents']:
        for pattern in intent['patterns']:

            pattern_keywords = preprocess_nltk(pattern)

            score = len(user_keywords.intersection(pattern_keywords))

            if score > best_score:
                best_score = score
                best_intent = intent

    if best_intent and best_score > 0:
        return random.choice(best_intent['responses'])
    else:
        return "I'm sorry, I don't have information on that topic."

def chat_interface():
    print("NWU History Chatbot")
    print("Ask questions about Northwestern University's history'). Type 'quit' to exit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break

        if intents_data.get('intents'):
            response = get_response_by_keywords(user_input, intents_data)
            print(f"Bot: {response}")
        else:
            print("Bot: Cannot run due to data loading error.")
            break

if __name__ == "__main__":
    chat_interface()
