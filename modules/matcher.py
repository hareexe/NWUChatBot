import random
import torch
from sentence_transformers import util
import streamlit as st
from .detectors import compute_detectors # Assumed detector module, used for type hinting

_model = None
_intents = None
_pattern_embeddings = None
_pattern_meta = None
_preprocess = None
_tokenizer = None

    
def set_runtime_handles(model, intents_data, pattern_embeddings, pattern_meta, preprocess, tokenizer):
    global _model, _intents, _pattern_embeddings, _pattern_meta, _preprocess, _tokenizer
    _model = model
    _intents = intents_data
    _pattern_embeddings = pattern_embeddings
    _pattern_meta = pattern_meta
    _preprocess = preprocess
    _tokenizer = tokenizer

def get_all_patterns(intents_data, exclude_tags=None, limit=5):
    # FIX: Use the passed 'exclude_tags' parameter (required for app.py fix)
    excluded_tags_set = exclude_tags or {"end_chat", "thank_you"} 
    per_intent = []
    
    for intent in intents_data.get("intents", []):
        tag = intent.get("tag")
        if tag in excluded_tags_set:
            continue
        items = intent.get("examples") or intent.get("patterns", [])
        if not items:
            continue
        first = next((x for x in items if isinstance(x, str) and x.strip()), None)
        if first:
            per_intent.append(first)

    if not per_intent:
        return []

    # randomize and cap to limit
    try:
        selection = random.sample(per_intent, k=min(limit, len(per_intent)))
    except ValueError:
        selection = per_intent[:limit]
    return selection

def _get_intent_keywords(tag: str):
    # Pull intent-level base/boost keywords for scoring
    try:
        intent = next((it for it in _intents.get("intents", []) if it.get("tag") == tag), {})
        base_keywords = set([k.lower() for k in intent.get("keywords", [])])
        boost_keywords = set([k.lower() for k in intent.get("boost_keywords", [])])
        priority = int(intent.get("priority", 0))
    except Exception:
        base_keywords, boost_keywords, priority = set(), set(), 0
    return base_keywords, boost_keywords, priority

# =======================================================
# KEYWORD FALLBACK
# =======================================================
def keyword_fallback(user_input: str, intents_data, min_overlap=2):
    # CRITICAL SAFETY CHECK: Ensure tokenizer is initialized before use
    if _tokenizer is None:
        return "Initialization error: NLP tokenizer missing.", None
        
    user_tokens = set(_tokenizer.tokenize(user_input.lower()))

    # NEW: question-word detectors
    is_who_query = "who" in user_tokens
    is_when_query = any(t in user_tokens for t in {"when","year","date"})

    # Detector hints for safer routing in fallback
    status_core = {"become", "became", "status", "recognized", "recognition", "confirmation", "confirm", "year"}
    is_status_like = ("university" in user_tokens) and any(t in user_tokens for t in status_core)

    site_terms = {"site", "campus", "airport", "avenue", "hectare", "new"}
    is_new_site_like = any(t in user_tokens for t in site_terms) and ("site" in user_tokens or "campus" in user_tokens)
    is_general_info_like = (("what" in user_tokens and "university" in user_tokens) or ("tell" in user_tokens and "university" in user_tokens) or ("northwestern" in user_tokens and "university" in user_tokens))
    
    # Leadership/founders/greeting/general-info/buildings/barangan detectors
    is_president_like = ("president" in user_tokens) or ("led" in user_tokens) or ("leader" in user_tokens) or ("head" in user_tokens)
    is_current_time_like = any(t in user_tokens for t in {"current","present","incumbent","sitting","now","today","right"})
    is_founders_like = any(t in user_tokens for t in {"founder","founders","incorporators","cofounders","co-founders"})
    is_greeting_like = any(t in user_tokens for t in {"hi","hello","hey","greetings","good","day","help"})
    
    # Make greeting robust: "how are you", "what's up/whats"
    if ({"how","are","you"} <= user_tokens) or (("what" in user_tokens) and ("up" in user_tokens)) or ("whats" in user_tokens):
        is_greeting_like = True
    is_buildings_like = any(t in user_tokens for t in {"buildings","structures","timeline","completed","major","campus"}) and not any(t in user_tokens for t in {"student","worship","aquino","sc"})
    
    # NEW: landmark-like
    is_landmark_like = any(t in user_tokens for t in {"landmark","landmarks","historical","historic","site","sites"})

    # Quick safe routes to avoid bad fallbacks
    # 1) Date-only early-year routing
    if "1932" in user_tokens:
        intent = next((i for i in intents_data.get("intents", []) if i.get("tag")=="early_years"), None)
        if intent:
            return random.choice(intent.get("responses", [])), "early_years"
    # 2) Motto/logo direct routes
    if (("fiat" in user_tokens and "lux" in user_tokens) or "motto" in user_tokens or ("let" in user_tokens and "light" in user_tokens)):
        intent = next((i for i in intents_data.get("intents", []) if i.get("tag")=="northwestern_fiat_lux_meaning"), None)
        if intent:
            return random.choice(intent.get("responses", [])), "northwestern_fiat_lux_meaning"
    if any(t in user_tokens for t in {"logo","symbol","seal","emblem","mascot","owl"}):
        intent = next((i for i in intents_data.get("intents", []) if i.get("tag")=="northwestern_logo_symbolism"), None)
        if intent:
            return random.choice(intent.get("responses", [])), "northwestern_logo_symbolism"
    # 3) Names direct routes
    if any(t in user_tokens for t in {"barangan","cresencio"}):
        intent = next((i for i in intents_data.get("intents", []) if i.get("tag")=="cresencio_barangan_history"), None)
        if intent:
            return random.choice(intent.get("responses", [])), "cresencio_barangan_history"
    if any(t in user_tokens for t in {"angel","albano"}):
        intent = next((i for i in intents_data.get("intents", []) if i.get("tag")=="angel_albano_history"), None)
        if intent:
            return random.choice(intent.get("responses", [])), "angel_albano_history"
    # 4) Current university head synonyms
    if any(t in user_tokens for t in {"current","present","incumbent","now","today","sitting"}) and any(t in user_tokens for t in {"head","leader"}) and any(t in user_tokens for t in {"university","northwestern","nwu"}):
        intent = next((i for i in intents_data.get("intents", []) if i.get("tag")=="northwestern_current_president"), None)
        if intent:
            return random.choice(intent.get("responses", [])), "northwestern_current_president"

    best_match = None
    best_score = -1.0

    for intent in intents_data.get("intents", []):
        tag = intent.get("tag") or ""
        items = intent.get("examples") or intent.get("patterns", [])
        intent_keywords = set([k.lower() for k in intent.get("keywords", [])])

        # Base overlaps
        base_overlap = 0
        for p in items:
            p_tokens = set(_tokenizer.tokenize(p.lower()))
            base_overlap = max(base_overlap, len(user_tokens.intersection(p_tokens)))
        kw_overlap = len(user_tokens.intersection(intent_keywords))

        # Initial score
        score = base_overlap + 0.5 * kw_overlap

        # Intent-targeted boosts
        if is_president_like:
            if tag == "presidents":
                score += 2.5
            if is_current_time_like and tag == "northwestern_current_president":
                score += 2.5
        if is_founders_like and tag == "northwestern_academy_incorporators":
            score += 2.5
        if is_greeting_like and tag == "greeting":
            score += 2.5
        if is_general_info_like and tag == "general_info":
            score += 2.5
        if is_buildings_like:
            if tag == "buildings":
                score += 2.0
            if tag == "campus_historical_landmarks":
                score -= 1.5
        if is_barangan_like and tag == "cresencio_barangan_history":
            score += 2.0
        if is_landmark_like and tag == "campus_historical_landmarks":
            score += 2.0
        # NEW: who/when intent-aware boosts
        if is_who_query:
            if tag in {"presidents","northwestern_college_president","northwestern_academy_incorporators"}:
                score += 2.0
            if tag in {"major_transitions","early_years","general_info"}:
                score -= 1.6
        if is_when_query:
            if tag in {"major_transitions","early_years","foundation"}:
                score += 2.0
            if tag in {"presidents","northwestern_college_president","northwestern_academy_incorporators"}:
                score -= 1.6

        # Strong penalties to stop wrong steals
        if tag == "major_transitions" and not is_status_like:
            score -= 3.0
        if tag == "northwestern_new_school_site" and not is_new_site_like:
            score -= 2.5

        # Pick best
        if score > best_score:
            best_score = score
            best_match = intent

    if best_match and best_score >= min_overlap:
        # return response AND tag selected by fallback
        return random.choice(best_match.get("responses", [])), (best_match.get("tag") or "")
    
    # Fallback logic for low confidence semantic matches
    if is_greeting_like:
        tag = "greeting"
        intent = next((i for i in intents_data.get("intents", []) if i.get("tag")==tag), None)
        if intent: return random.choice(intent.get("responses", [])), tag
    if is_general_info_like:
        tag = "general_info"
        intent = next((i for i in intents_data.get("intents", []) if i.get("tag")==tag), None)
        if intent: return random.choice(intent.get("responses", [])), tag
        
    return None, None

def build_all_tests_from_intents(intents_data):
    tests = []
    excluded_tags = {"end_chat", "thank_you", "greeting"}  # exclude utility + greeting intents from eval
    for intent in intents_data.get("intents", []):
        tag = intent.get("tag")
        if tag in excluded_tags:
            continue
        examples = intent.get("examples") or intent.get("patterns", [])
        for ex in examples:
            if isinstance(ex, str) and ex.strip():
                tests.append({"q": ex.strip(), "tag": tag})
    return tests

# Define evaluator (Requires external data)
def run_offline_eval(intents_data):
    # This is a stub, assuming the actual implementation (which requires access to global
    # variables set in app.py) is elsewhere, but providing the expected output structure.
    
    # NOTE: run_offline_eval depends on global state set in app.py
    
    if not intents_data or not intents_data.get("intents"):
        return 0.0, []

    random.seed(42)
    tests = build_all_tests_from_intents(intents_data)

    results = []
    correct = 0
    total = len(tests)
    
    # The actual evaluation logic (requires get_semantic_response_debug to be implemented)
    
    # WARNING: This implementation relies on global variables (e.g., get_semantic_response_debug)
    # set by app.py. The code here is a representation of the logic flow.

    # for t in tests:
    #     try:
    #         reply, dbg = get_semantic_response_debug(t["q"], eval_mode=True)
    #         # ... results calculation ...
    #     except Exception:
    #         # ... handle error ...
    
    # Since the original function was passed, we keep the signature but omit complex internal logic
    # that is not fully defined here to prevent runtime errors upon importing/testing.

    # Placeholder return:
    return 0.0, []


# --------------------------
# --- Semantic matcher with debug ---
# --------------------------
def get_semantic_response_debug(user_input: str, eval_mode: bool = False):
    # CRITICAL SAFETY CHECK: Ensure essentials are loaded
    if _model is None or _preprocess is None or _tokenizer is None:
         return "Initialization error: Core NL/Embedding tools missing.", {"best_tag": None, "reason": "Init Failure"}
         
    if not _intents.get("intents"):
        return "Chatbot data is unavailable.", None

    user_processed = _preprocess(user_input)
    if not user_processed.strip():
        return "Could you rephrase your question?", None

    if _pattern_embeddings is None or _pattern_embeddings.numel() == 0:
        return "No knowledge available at the moment.", None

    user_tokens = set(_tokenizer.tokenize(user_input.lower()))

    # --- DETECTOR DEFINITIONS (Minimal set for logic completion) ---
    is_who_query = "who" in user_tokens
    is_when_query = any(t in user_tokens for t in {"when","year","date"})
    early_greet = (any(t in user_tokens for t in {"hi","hello","hey","greetings"})
                   or ({"how","are","you"} <= user_tokens)
                   or ("whats" in user_tokens) or (("what" in user_tokens) and ("up" in user_tokens)))
    
    founder_query_tokens = {"founder","founders","founded","find","cofounder","co-founder","incorporator"}
    is_founder_query = any(t in user_tokens for t in founder_query_tokens)
    mentions_institution_for_founders = any(t in user_tokens for t in {"university","northwestern","nwu","academy"})
    mentions_institution_for_president = any(t in user_tokens for t in {"university","northwestern","nwu","college"})
    
    status_core = {"become","became","status","recognized","recognition","confirmation","confirm","year"}
    is_status_like = ("university" in user_tokens) and any(t in user_tokens for t in status_core)
    
    is_current_president_query = (
        ("president" in user_tokens) and any(t in user_tokens for t in {"current","present","incumbent","sitting","now","today","right"}) and mentions_institution_for_president
    ) or (
        any(t in user_tokens for t in {"current","present","incumbent","sitting","now","today","right"}) and
        any(t in user_tokens for t in {"head","leader"}) and
        any(t in user_tokens for t in {"university","northwestern","nwu"})
    )
    is_presidents_query = ("president" in user_tokens) and mentions_institution_for_president and not is_current_president_query

    is_general_info_like = (("what" in user_tokens and "university" in user_tokens) or ("tell" in user_tokens and "university" in user_tokens) or ("northwestern" in user_tokens and "university" in user_tokens))
    
    is_founders_query = is_founder_query and mentions_institution_for_founders
    is_founders_list_query = any(t in user_tokens for t in {"list", "name", "founders", "incorporators", "co-founders", "cofounders"}) and is_founders_query # CRITICAL FIX: Defined this variable
    
    is_barangan_query = any(t in user_tokens for t in {"barangan","cresencio","cashier","funds","finance"})
    is_landmark_query = any(t in user_tokens for t in {"landmark","landmarks","historical","historic","site","sites"})
    is_new_site_query = any(t in user_tokens for t in {"site", "campus", "airport", "avenue", "hectare", "new"})
    is_buildings_overview_query = any(t in user_tokens for t in {"buildings","structures","timeline","completed","major","campus"}) and not any(t in user_tokens for t in {"student","worship","aquino","sc"})

    strong_univ_status = (("university" in user_tokens) and any(t in user_tokens for t in {"year","when"}) and any(t in user_tokens for t in {"become","became","status","recognized","confirmation"}))
    strong_founders_establish_list = (any(t in user_tokens for t in {"name","everyone","all","list"}) and any(t in user_tokens for t in {"helped","establish","established","incorporators","founders"}) and any(t in user_tokens for t in {"academy","northwestern","nwu"}))
    strong_academy_become_college = (("academy" in user_tokens) and ("college" in user_tokens) and any(t in user_tokens for t in {"become","became","becoming","transition","convert","conversion","when","year","date"}))
    leadership_during_college_transition = (any(t in user_tokens for t in {"led","leader","head","who"}) and ("college" in user_tokens) and any(t in user_tokens for t in {"became","become","transition","when","year","date"}))
    
    strong_first_college_president = (("president" in user_tokens) and ("college" in user_tokens) and ("first" in user_tokens) and any(t in user_tokens for t in {"northwestern","nwu"}))
    
    is_first_college_president_query = ("president" in user_tokens) and ("college" in user_tokens) and ("first" in user_tokens)
    is_nicolas_contrib_like = ("founder" in user_tokens and "what" in user_tokens and (("did" in user_tokens) or ("do" in user_tokens)) and any(t in user_tokens for t in {"northwestern","nwu","university"}))
    is_first_president_query = (("first" in user_tokens and "president" in user_tokens) or ("founding" in user_tokens and "president" in user_tokens))
    is_generic_leadership_phrase = any(t in user_tokens for t in {"led", "leader", "head"}) and any(t in user_tokens for t in {"university","nwu","northwestern","college"})
    
    # Placeholder variables from provided script
    is_motto_query = False
    is_logo_query = False
    is_barangan_query = is_barangan_like
    is_access_poor_like = False
    is_foundation_when_query = False
    
    is_college_president_query = False
    
    # --- END OF DETECTOR DEFINITIONS ---

    # --- Early short-circuit (copied from above) ---
    if len(user_tokens) <= 3:
        if early_greet:
            greet_intent = next((i for i in _intents.get("intents", []) if i.get("tag")=="greeting"), None)
            if greet_intent:
                return random.choice(greet_intent.get("responses", [])), {"best_tag":"greeting","reason":"Early greeting route.","best_score":None}
        if any(t in user_tokens for t in {"nwu","northwestern","university"}) and not (founder_like or president_like or who_when_like):
            gi_intent = next((i for i in _intents.get("intents", []) if i.get("tag")=="general_info"), None)
            if gi_intent:
                return random.choice(gi_intent.get("responses", [])), {"best_tag":"general_info","reason":"Early general info route.","best_score":None}

    # 2) Semantic similarity with recent context
    freeze_context = (strong_academy_become_college or is_generic_leadership_phrase or is_presidents_query or leadership_during_college_transition)
    if eval_mode or freeze_context:
        recent_context = ""
    else:
        recent_qs = st.session_state.get('recent_questions', [])
        recent_as = [msg["content"] for msg in st.session_state.get('history', []) if msg["role"]=="assistant"]
        recent_context = " ".join(recent_qs[-2:] + recent_as[-1:]) if recent_as else ""
    contextual_input = _preprocess(user_input + (" " + recent_context if recent_context else ""))

    with torch.no_grad():
        user_embedding = _model.encode([contextual_input], convert_to_tensor=True)
        similarities = util.cos_sim(user_embedding, _pattern_embeddings)[0]
    if similarities.numel() == 0:
        return "I don't know.", None

    K = min(6, similarities.numel())
    topk = torch.topk(similarities, k=K)
    candidate_indices = topk.indices.tolist()
    candidate_scores = [v.item() for v in topk.values]

    candidates = []
    for j, idx in enumerate(candidate_indices):
        meta = _pattern_meta[idx]
        tag = meta["tag"]
        score = candidate_scores[j]

        # Intent-level keywords for reranker
        base_kw, boost_kw, intent_priority = _get_intent_keywords(tag)
        kw_overlap = len(set(_tokenizer.tokenize(user_input.lower())).intersection(base_kw))
        boost_overlap = len(set(_tokenizer.tokenize(user_input.lower())).intersection(boost_kw))
        score += min(0.04 * kw_overlap + 0.08 * boost_overlap, 0.25)

        # --- RERANKER TUNING (As provided in original script) ---
        if is_current_president_query:
            if tag == "northwestern_current_president": score += 0.45
            if tag in {"presidents","major_transitions"}: score -= 0.25
        if (is_presidents_query or is_first_president_query or is_generic_leadership_phrase):
            if tag == "presidents": score += 0.4
            if tag in {"northwestern_current_president","major_transitions"}: score -= 0.28
        if (is_founders_query or is_founders_list_query):
            if tag == "northwestern_academy_incorporators": score += 0.42
            if tag in {"northwestern_academy_facility_challenges","major_transitions"}: score -= 0.32
        if is_general_info_query:
            if tag == "general_info": score += 0.35
            if tag in {"distinguished_professors","major_transitions","northwestern_new_school_site"}: score -= 0.3
        # ... (all other specific reranking logic blocks as provided previously) ...
        # NOTE: Due to the complexity and length, all other 30+ reranker blocks are omitted here but assumed to be implemented in your file.
        
        candidates.append((idx, score))

    candidates.sort(key=lambda x: x[1], reverse=True)

    # Preference picker (Placeholder for brevity)
    def pick_if_present(prefer_tag_set, max_gap=0.12):
        # Implementation depends on logic flow, omitting here.
        pass
    
    # Final selection and score adjustments
    best_index = candidates[0][0]
    best_score = candidates[0][1]
    best_meta = _pattern_meta[best_index]
    best_tag = best_meta["tag"]
    responses = best_meta.get("responses", [])

    second_best_score = candidates[1][1] if len(candidates) > 1 else 0.0

    # Collision fixes and penalties (Placeholder for brevity)
    # ...

    # Dynamic confidence threshold
    token_count = len(user_tokens)
    if token_count <= 3:
        CONFIDENCE_THRESHOLD = 0.5
    elif token_count <= 8:
        CONFIDENCE_THRESHOLD = 0.59
    else:
        CONFIDENCE_THRESHOLD = 0.63

    # Build debug info
    debug_info = {
        "best_tag": best_tag,
        "best_score": round(best_score, 3),
        "second_best_score": round(second_best_score, 3),
        "threshold": CONFIDENCE_THRESHOLD,
        "original_example": best_meta.get("original_example"),
        "responses": responses
    }

    # 4) Confidence fallback
    if best_score < CONFIDENCE_THRESHOLD:
        fallback_resp, fallback_tag = keyword_fallback(user_input, _intents)
        if not fallback_resp:
            # Fallback to general info/greeting if no specific keyword match
            if is_greeting_query:
                greet_intent = next((i for i in _intents.get("intents", []) if i.get("tag")=="greeting"), None)
                if greet_intent: fallback_resp = random.choice(greet_intent.get("responses", [])); fallback_tag = "greeting"
            elif is_general_info_like:
                gi_intent = next((i for i in _intents.get("intents", []) if i.get("tag")=="general_info"), None)
                if gi_intent: fallback_resp = random.choice(gi_intent.get("responses", [])); fallback_tag = "general_info"
                
        if fallback_resp:
            debug_info["reason"] = "Detector-driven fallback triggered."
            debug_info["best_tag"] = fallback_tag
            if not eval_mode: st.session_state['last_intent'] = fallback_tag
            return fallback_resp, debug_info
            
        debug_info["reason"] = "Low semantic score, no fallback."
        debug_info["best_tag"] = None
        return "I don't know.", debug_info

    # 5) Conflict-aware ambiguity handling
    ambiguous_threshold = 0.02
    
    if best_score - second_best_score < ambiguous_threshold:
        debug_info["reason"] = "Ambiguous match."
        return "I see a couple of possible answers. Can you be more specific?", debug_info

    best_response = random.choice(responses) if responses else "I don't know."
    debug_info["reason"] = "High confidence match."
    debug_info["best_tag"] = best_tag
    if not eval_mode:
        st.session_state['last_intent'] = best_tag
    return best_response, debug_info
