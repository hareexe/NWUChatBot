import os
import json
import torch
import hashlib
from sentence_transformers import SentenceTransformer, util
from .nlp_utils import preprocess, expand_with_synonyms

# --------------------------
# --- Load intents.json ----
# --------------------------
def load_data():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    filepath = os.path.join(base_dir, "intents.json")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        return {"intents": []}

def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# --------------------------
# --- Hash helper ----
# --------------------------
def _hash_intents(data):
    s = json.dumps(data, sort_keys=True).encode("utf-8")
    return hashlib.sha256(s).hexdigest()

# --------------------------
# --- Embedding model  -----
# --------------------------
def load_embedding_model():
    return SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

# --------------------------
# --- Build embeddings ----
# --------------------------
def build_intent_embeddings(intents_data_hash: str, intents_data_serialized: str, model=None):
    intents_data_local = json.loads(intents_data_serialized)
    all_texts = []
    meta_data = []

    for intent in intents_data_local.get("intents", []):
        tag = intent.get("tag")
        responses = intent.get("responses", [])
        examples = intent.get("examples", []) or intent.get("patterns", [])
        keywords = intent.get("keywords", [])
        context = intent.get("context", "")
        description = intent.get("intent_description", "")

        for ex in examples:
            combined_text = f"{ex} {' '.join(keywords)} {context} {description}"
            processed = preprocess(expand_with_synonyms(combined_text, max_synonyms_per_token=1))
            if processed.strip():
                all_texts.append(processed)
                meta_data.append({
                    "tag": tag,
                    "responses": responses,
                    "original_example": ex,
                    "keywords": keywords,
                    "context": context,
                    "description": description
                })

    if not all_texts:
        return [], None, []

    with torch.no_grad():
        embeddings = model.encode(all_texts, convert_to_tensor=True)
    return all_texts, embeddings, meta_data

intents_data = load_data()
intents_hash = _hash_intents(intents_data)
intents_serialized = json.dumps(intents_data, sort_keys=True)
model = load_embedding_model()
all_pattern_texts, pattern_embeddings, pattern_meta = build_intent_embeddings(intents_hash, intents_serialized, model)
