from modules.matcher import get_semantic_response_debug
from modules.matcher import build_all_tests_from_intents 
import random


# Use the definition that exists in the files:
def build_all_tests_from_intents(intents_data):
    # This function body is identical to the version found in matcher.py
    tests = []
    # EXCLUDED TAGS REMOVED as per request.
    
    for intent in intents_data.get("intents", []):
        tag = intent.get("tag")
        examples = intent.get("examples") or intent.get("patterns", [])
        for ex in examples:
            if isinstance(ex, str) and ex.strip():
                tests.append({"q": ex.strip(), "tag": tag})
    return tests

def run_offline_eval(intents_data):

    from modules.matcher import run_offline_eval as matcher_run_offline_eval
    return matcher_run_offline_eval()
