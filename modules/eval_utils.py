from modules.matcher import get_semantic_response_debug 
import random

def build_all_tests_from_intents(intents_data):
    tests = []
    excluded_tags = {"end_chat", "thank_you", "greeting"}  
    for intent in intents_data.get("intents", []):
        tag = intent.get("tag")
        if tag in excluded_tags:
            continue
        examples = intent.get("examples") or intent.get("patterns", [])
        for ex in examples:
            if isinstance(ex, str) and ex.strip():
                tests.append({"q": ex.strip(), "tag": tag})
    return tests

def run_offline_eval(intents_data):
    # Deterministic sampling for consistent eval
    random.seed(42)
    # Use ALL examples from intents.json
    tests = build_all_tests_from_intents(intents_data)

    results = []
    correct = 0
    total = len(tests)

    for t in tests:
        # call matcher in evaluation mode to ignore session context and avoid state mutations
        reply, dbg = get_semantic_response_debug(t["q"], eval_mode=True)
        best_tag = (dbg or {}).get("best_tag")
        ok = best_tag == t["tag"]
        correct += 1 if ok else 0
        results.append({
            "query": t["q"],
            "expected": t["tag"],
            "predicted": best_tag,
            "ok": ok,
            "reason": (dbg or {}).get("reason"),
            "score": (dbg or {}).get("best_score")
        })

    accuracy = correct / total if total else 0.0
    return accuracy, results
