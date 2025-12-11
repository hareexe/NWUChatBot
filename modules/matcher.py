import random
import torch
from sentence_transformers import util
import streamlit as st
from .detectors import compute_detectors

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

def get_all_patterns(intents_data, limit=5, exclude_tags=None):
    """
    Retrieves a sample of patterns from intents_data, excluding specified tags.
    (Updated to accept exclude_tags to resolve TypeError in app.py)
    """
    if exclude_tags is None:
        exclude_tags = set()
    elif isinstance(exclude_tags, list):
        exclude_tags = set(exclude_tags)
        
    per_intent = []
    for intent in intents_data.get("intents", []):
        tag = intent.get("tag")
        
        # Apply the exclusion logic
        if tag in exclude_tags:
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

def keyword_fallback(user_input: str, intents_data, min_overlap=2):
    user_tokens = set(_tokenizer.tokenize(user_input.lower()))

    # NEW: question-word detectors
    is_who_query = "who" in user_tokens
    is_when_query = any(t in user_tokens for t in {"when","year","date"})
    is_what_query = "what" in user_tokens

    # Detector hints for safer routing in fallback
    status_core = {"become", "became", "status", "recognized", "recognition", "confirmation", "confirm", "year"}
    is_status_like = ("university" in user_tokens) and any(t in user_tokens for t in status_core)

    site_terms = {"site", "campus", "airport", "avenue", "hectare", "new"}
    is_new_site_like = any(t in user_tokens for t in site_terms) and ("site" in user_tokens or "campus" in user_tokens)
    is_general_info_like = (("what" in user_tokens and "university" in user_tokens) or ("tell" in user_tokens and "university" in user_tokens) or ("northwestern" in user_tokens and "university" in user_tokens))
    # Leadership/founders/greeting/general-info/buildings/barangan detectors
    # Remove 'founding' to avoid presidents stealing founders queries
    is_president_like = ("president" in user_tokens) or ("led" in user_tokens) or ("leader" in user_tokens) or ("head" in user_tokens)
    is_current_time_like = any(t in user_tokens for t in {"current","present","incumbent","sitting","now","today","right"})
    is_founders_like = any(t in user_tokens for t in {"founder","founders","incorporators","cofounders","co-founders"})
    is_greeting_like = any(t in user_tokens for t in {"hi","hello","hey","greetings","good","day","help"})
    # Make greeting robust: "how are you", "what's up/whats"
    if ({"how","are","you"} <= user_tokens) or (("what" in user_tokens) and ("up" in user_tokens)) or ("whats" in user_tokens):
        is_greeting_like = True
    is_buildings_like = any(t in user_tokens for t in {"buildings","structures","timeline","completed","major","campus"}) and not any(t in user_tokens for t in {"student","worship","aquino","sc"})
    is_barangan_like = any(t in user_tokens for t in {"barangan","cresencio","cashier","funds","finance"})
    # NEW: landmark-like
    is_landmark_like = any(t in user_tokens for t in {"landmark","landmarks","historical","historic","site","sites"})
    # NEW: all-presidents-like
    is_all_presidents_like = any(t in user_tokens for t in {"all", "list", "past", "first", "college"}) and is_president_like
    # NEW: nicolas contribution like (Added "do" and "did" to who/what contribution queries)
    is_nicolas_contrib_like = ("nicolas" in user_tokens) and any(t in user_tokens for t in {"contribution", "do", "did", "impact", "role"})
    # NEW: nicolas teacher/faculty like
    is_nicolas_teacher_like = ("nicolas" in user_tokens) and any(t in user_tokens for t in {"teacher", "known", "faculty", "instructor"})
    # NEW: academy early sacrifices like
    is_academy_sacrifices_like = any(t in user_tokens for t in {"sacrifices", "goal", "vision"}) and ("academy" in user_tokens or "founders" in user_tokens)
    # DELETED: is_engineering_program_like = "engineering" in user_tokens and any(t in user_tokens for t in {"program", "courses", "dean", "flagship"})
    # NEW: generic president query (for "Who is the president?" to push to current)
    is_generic_president_query = is_president_like and is_who_query and not is_current_time_like and not is_all_presidents_like and not any(t in user_tokens for t in {"first", "past", "list", "all", "of", "college"})
    # NEW: nurturing years like
    is_nurturing_years_like = (any(t in user_tokens for t in {"early","beginnings","like","where","held","challenges","face"}) and ("northwestern" in user_tokens or "nwu" in user_tokens)) and not is_academy_sacrifices_like
    # NEW: common wealth era like (added constitution/expand)
    is_commonwealth_planning_like = any(t in user_tokens for t in {"planning","expand","programs","courses","why", "commonwealth", "1935", "constitution", "surge", "affect"})
    # NEW: transition process like
    is_transition_process_like = any(t in user_tokens for t in {"convert","conversion","transition","process","steps","petition","apply","application"}) and any(t in user_tokens for t in {"college", "academy", "how"})
    # NEW: student activists like
    is_activist_details_query = any(t in user_tokens for t in {"velasco", "pascual", "became", "support", "notable", "leaders", "marcos"})
    # NEW: generic course query
    is_generic_course_query = any(t in user_tokens for t in {"courses", "programs", "degree", "associate"}) and ("college" in user_tokens or "northwestern" in user_tokens)


    # --- Hard Routes for Ambiguous Short Queries ---
    # 1) Date-only early-year routing
    if "1932" in user_tokens:
        # Route to foundation for "what happened in 1932?"
        if is_what_query and len(user_tokens) <= 5:
            intent = next((i for i in intents_data.get("intents", []) if i.get("tag")=="foundation"), None)
            if intent:
                return random.choice(intent.get("responses", [])), "foundation"
        # Otherwise, route to early_years (e.g. "tell me about 1932")
        intent = next((i for i in intents_data.get("intents", []) if i.get("tag")=="early_years" or i.get("tag")=="foundation"), None)
        if intent:
            return random.choice(intent.get("responses", [])), "early_years"
            
    # 1.5) FOUNDATION/EARLY YEARS direct routes (FIX for MISSED founding/date queries)
    if (("founded" in user_tokens or "foundation" in user_tokens or "early" in user_tokens) and len(user_tokens) <= 5):
        intent = next((i for i in intents_data.get("intents", []) if i.get("tag")=="foundation"), None)
        if not intent:
            intent = next((i for i in intents_data.get("intents", []) if i.get("tag")=="early_years"), None)

        if intent:
             return random.choice(intent.get("responses", [])), "foundation_or_early_years_hard_route"
             
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
    # 4) Current university head synonyms OR Generic President Query
    if (any(t in user_tokens for t in {"head","leader","president"}) and any(t in user_tokens for t in {"university","northwestern","nwu"})) and (is_current_time_like or is_generic_president_query):
        intent = next((i for i in intents_data.get("intents", []) if i.get("tag")=="northwestern_current_president"), None)
        if intent:
            return random.choice(intent.get("responses", [])), "northwestern_current_president"
    # 5) First/Past/All president synonyms (routes to all-list, including "first college president")
    if is_all_presidents_like or (any(t in user_tokens for t in {"first", "college"}) and is_president_like and not is_current_time_like):
        intent = next((i for i in intents_data.get("intents", []) if i.get("tag")=="complete_northwestern_presidents_list"), None)
        if intent:
            return random.choice(intent.get("responses", [])), "complete_northwestern_presidents_list"
    # 6) Founders of NWU short route (Add explicit hard route for short 'incorporators' query)
    if ("incorporators" in user_tokens) and ("who" in user_tokens) and len(user_tokens) <= 5:
        intent = next((i for i in intents_data.get("intents", []) if i.get("tag")=="northwestern_academy_incorporators"), None)
        if intent:
            return random.choice(intent.get("responses", [])), "northwestern_academy_incorporators"
            
    if any(t in user_tokens for t in {"founders", "incorporators"}) and any(t in user_tokens for t in ["nwu", "northwestern", "academy"]) and len(user_tokens) <= 5:
        intent = next((i for i in intents_data.get("intents", []) if i.get("tag")=="northwestern_academy_incorporators"), None)
        if intent:
            return random.choice(intent.get("responses", [])), "northwestern_academy_incorporators"
    # NEW: Route short queries like "Presidents of Northwestern" to the list
    if is_president_like and not any(t in user_tokens for t in {"current","incumbent","sitting","now","today","right"}) and len(user_tokens) <= 6:
        intent = next((i for i in intents_data.get("intents", []) if i.get("tag")=="complete_northwestern_presidents_list"), None)
        if intent:
            return random.choice(intent.get("responses", [])), "complete_northwestern_presidents_list"
    # NEW: Nicolas contribution/teacher route for short queries
    if is_nicolas_contrib_like:
        intent = next((i for i in intents_data.get("intents", []) if i.get("tag")=="nicolas_contribution"), None)
        if intent:
            return random.choice(intent.get("responses", [])), "nicolas_contribution"
    if is_nicolas_teacher_like:
        intent = next((i for i in intents_data.get("intents", []) if i.get("tag")=="northwestern_faculty_mentors"), None)
        if intent:
            return random.choice(intent.get("responses", [])), "northwestern_faculty_mentors"
    # NEW: Transition Process route for "how did become a college"
    if is_transition_process_like:
        intent = next((i for i in intents_data.get("intents", []) if i.get("tag")=="transition_process"), None)
        if intent:
            return random.choice(intent.get("responses", [])), "transition_process"


    best_match = None
    best_score = -1.0

    for intent in intents_data.get("intents", []):
        tag = intent.get("tag") or ""
        items = intent.get("examples") or intent.get("patterns", [])
        intent_keywords = set([k.lower() for k in intent.get("keywords", [])])
        intent_boost_keywords = set([k.lower() for k in intent.get("boost_keywords", [])])

        # Base overlaps
        base_overlap = 0
        for p in items:
            p_tokens = set(_tokenizer.tokenize(p.lower()))
            base_overlap = max(base_overlap, len(user_tokens.intersection(p_tokens)))
        kw_overlap = len(user_tokens.intersection(intent_keywords))
        boost_kw_overlap = len(user_tokens.intersection(intent_boost_keywords))

        # Initial score
        score = base_overlap + 0.5 * kw_overlap + 1.5 * boost_kw_overlap

        # Intent-targeted boosts
        if is_president_like:
            if tag == "complete_northwestern_presidents_list":
                score += 2.5
            if tag == "northwestern_current_president":
                score += 2.5
            # NEW: strong boost for all-list, including 'first president' queries
            if is_all_presidents_like and tag == "complete_northwestern_presidents_list":
                score += 3.5
            # NEW: strong boost for generic president query
            if is_generic_president_query and tag == "northwestern_current_president":
                score += 3.5

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
            
        # NEW: nicolas contribution routing (Fixed dangling elif to separate if blocks)
        if is_nicolas_contrib_like and tag == "nicolas_contribution":
            score += 2.5
        if is_nicolas_contrib_like and tag == "nicolas_contribution":
             score += 8
        if is_nicolas_contrib_like and tag in {"northwestern_academy_incorporators", "northwestern_faculty_mentors"}: 
            score = max(0, score - 4)
            
        # NEW: nicolas teacher routing (Fixed dangling elif to separate if blocks)
        if is_nicolas_teacher_like and tag == "northwestern_faculty_mentors":
            score += 2.5
        if is_nicolas_teacher_like and tag == "northwestern_faculty_mentors":
            score += 8
        if is_nicolas_teacher_like and tag in {"nicolas_contribution", "northwestern_academy_incorporators"}: 
            score = max(0, score - 4)

        # NEW: Maximo Caday routing (Fixed dangling elif to separate if blocks)
        if is_caday_relationship_like:
            if tag == "maximo_caday_relationship_with_founders": 
                score += 8
            elif tag == "nicolas_contribution": 
                score = max(0, score - 6)

        # NEW: Sacrifices routing (Fixed dangling elif to separate if blocks)
        if is_academy_sacrifices_like and tag == "northwestern_academy_early_sacrifices":
            score += 8
        elif is_academy_sacrifices_like and tag in {"early_years", "nurturing_years"}: 
            score = max(0, score - 4)

        # NEW: Commonwealth Era routing (Fixed dangling elif to separate if blocks)
        if is_commonwealth_planning_like and tag == "northwestern_academy_commonwealth_era":
            score += 8
        elif is_commonwealth_planning_like and tag in {"early_years", "northwestern_college_courses", "Accreditation"}: 
            score = max(0, score - 6)

        # NEW: Transition Process routing (Fixed dangling elif to separate if blocks)
        if is_transition_process_like and tag == "transition_process":
            score += 8
        elif is_transition_process_like and tag in {"early_years", "northwestern_college_courses", "Accreditation"}: 
            score = max(0, score - 6)
            
        # NEW: Nurturing Years routing (Fixed dangling elif to separate if blocks)
        if is_nurturing_years_like and tag == "nurturing_years":
            score += 8
        elif is_nurturing_years_like and tag == "early_years": 
            score = max(0, score - 6)

        # NEW: Student Activism routing (Fixed dangling elif to separate if blocks)
        if is_activist_details_query and tag == "northwestern_student_activists":
            score += 8
        elif is_activist_details_query and tag == "northwestern_martial_law": 
            score = max(0, score - 4)

        if is_landmark_like and tag == "campus_historical_landmarks":
            score += 2.0
        
        # NEW: generic course query
        if is_generic_course_query and tag == "northwestern_college_courses":
            score += 2.5
            
        # NEW: who/when/what intent-aware boosts - INCREASED PENALTIES/BOOSTS
        if is_who_query:
            if tag in {"complete_northwestern_presidents_list","northwestern_current_president","northwestern_academy_incorporators", "cresencio_barangan_history", "northwestern_student_activists", "northwestern_faculty_mentors", "northwestern_classroom_icons"}:
                score += 3.0
            if tag in {"major_transitions","early_years","general_info", "foundation", "northwestern_college_courses"}:
                score -= 2.5
        if is_when_query:
            if tag in {"major_transitions","early_years","foundation", "transition_process", "northwestern_academy_commonwealth_era"}:
                score += 3.0
            if tag in {"complete_northwestern_presidents_list","northwestern_current_president","northwestern_academy_incorporators"}:
                score -= 2.5
        if is_what_query:
            if tag in {"general_info", "northwestern_fiat_lux_meaning", "northwestern_logo_symbolism", "northwestern_college_courses", "nicolas_contribution", "northwestern_academy_early_sacrifices"}:
                score += 2.0
            if tag in {"northwestern_current_president", "northwestern_academy_incorporators", "foundation"}:
                score -= 1.5


        # Strong penalties to stop wrong steals
        if tag == "major_transitions" and not is_status_like:
            score -= 3.0
        if tag == "northwestern_new_school_site" and not is_new_site_like:
            score -= 2.5
        # NEW: barangan penalty on founders
        if is_barangan_like and tag == "northwestern_academy_incorporators":
            score -= 1.5
        # NEW: nicolas contribution penalty on founders
        if is_nicolas_contrib_like and tag == "northwestern_academy_incorporators":
            score -= 1.5
        # NEW: nicolas teacher penalty on founders
        if is_nicolas_teacher_like and tag == "northwestern_academy_incorporators":
            score -= 1.5
        # NEW: early sacrifices penalty on generic early_years
        if is_academy_sacrifices_like and tag == "early_years":
            score -= 1.5
        # NEW: Nurturing years penalty on generic early_years/foundation
        if is_nurturing_years_like and tag in {"early_years", "foundation"}:
             score -= 1.0
        # NEW: General info steals (Presidents of Northwestern/NWU award)
        if (tag == "general_info" or tag == "2004_Award" or tag == "Accreditation" or tag == "Deregulated_Status") and any(t in user_tokens for t in {"presidents", "award", "ranking", "courses", "degree"}):
             score -= 1.5 # Broader penalty
        # NEW: Strong penalty for current time queries stealing from list
        if is_current_time_like and tag == "complete_northwestern_presidents_list":
            score -= 4.0
        # NEW: Strong penalty for all-list stealing from current
        if is_all_presidents_like and not is_current_time_like and tag == "northwestern_current_president":
             score -= 4.0
        # NEW: Strong penalty for martial law stealing 1932 queries
        if tag == "northwestern_martial_law" and any(t in user_tokens for t in ["1932", "foundation", "founded", "original name", "early"]):
            score -= 4.5 
        # NEW: Strong penalty for 1932/foundation stealing martial law queries
        if tag in {"early_years", "foundation"} and any(t in user_tokens for t in ["martial", "law", "1970s", "activism"]):
            score -= 4.5
        # NEW: Penalty for maximo caday stealing nicolas contribution
        if tag == "nicolas_contribution" and any(t in user_tokens for t in ["maximo", "caday", "relationship"]):
            score -= 2.0
        # NEW: Penalty for college courses stealing commonwealth planning/transition process
        if tag == "northwestern_college_courses" and (is_commonwealth_planning_like or is_transition_process_like):
             score -= 2.5
        # DELETED: if is_engineering_program_like and tag == "northwestern_college_courses": score -= 2.5
        # NEW: Penalty for Accreditation/Status stealing Course queries
        if is_generic_course_query and tag in {"Accreditation", "Deregulated_Status", "major_transitions"}:
             score -= 2.5
        # NEW: Penalty for Accreditation/Status stealing Commonwealth Era
        if is_commonwealth_planning_like and tag in {"Accreditation", "Deregulated_Status"}:
             score -= 2.5
        # NEW: Penalty for faculty_mentors stealing general "great teachers"
        if tag == "northwestern_faculty_mentors" and tag not in user_input.lower():
             score -= 1.5
        if tag == "northwestern_faculty_mentors" and any(t in user_tokens for t in {"student", "leaders", "marcos", "protest"}):
             score -= 3.0
        # NEW: Massive penalty for Commonwealth stealing Incorporators
        if (is_founders_like or "incorporator" in user_tokens) and tag == "northwestern_academy_commonwealth_era":
             score -= 5.0
        # NEW: Massive penalty for Incorporators stealing Commonwealth
        if is_commonwealth_planning_like and tag == "northwestern_academy_incorporators":
             score -= 5.0
        # NEW: Penalty for Presidents List stealing Founders (Example 2 Fix)
        if (is_founders_like or "incorporator" in user_tokens) and tag == "complete_northwestern_presidents_list":
             score -= 5.0


        # Pick best
        if score > best_score:
            best_score = score
            best_match = intent

    if best_match and best_score >= min_overlap:
        # return response AND tag selected by fallback
        return random.choice(best_match.get("responses", [])), (best_match.get("tag") or "")
    return None, None

def build_all_tests_from_intents(intents_data):
    tests = []
    # EXCLUDED TAGS REMOVED as per request.
    
    for intent in intents_data.get("intents", []):
        tag = intent.get("tag")
        examples = intent.get("examples") or intent.get("patterns", [])
        for ex in examples:
            if isinstance(ex, str) and ex.strip():
                tests.append({"q": ex.strip(), "tag": tag})
    return tests

# Define evaluator BEFORE UI code
def run_offline_eval():
    # Deterministic sampling for consistent eval
    random.seed(42)
    # Use ALL examples from intents.json
    tests = build_all_tests_from_intents(_intents) # Use global _intents

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

# --------------------------
# --- Semantic matcher with debug ---
# --------------------------
def get_semantic_response_debug(user_input: str, eval_mode: bool = False):
    if not _intents.get("intents"):
        return "Chatbot data is unavailable.", None

    user_processed = _preprocess(user_input)
    if not user_processed.strip():
        return "Could you rephrase your question?", None

    if _pattern_embeddings is None or len(_pattern_meta) == 0:
        return "No knowledge available at the moment.", None

    user_tokens = set(_tokenizer.tokenize(user_input.lower()))

    # NEW: question-word detectors (must be defined before later use)
    is_who_query = "who" in user_tokens
    is_when_query = any(t in user_tokens for t in {"when","year","date"})
    is_what_query = "what" in user_tokens

    # --- Early short-circuit for greetings/general info/awards ---
    # Replace permissive greeting with robust combos
    early_greet = (any(t in user_tokens for t in {"hi","hello","hey","greetings"})
                    or ({"how","are","you"} <= user_tokens)
                    or ("whats" in user_tokens) or (("what" in user_tokens) and ("up" in user_tokens)))
    if len(user_tokens) <= 3:
        if early_greet:
            greet_intent = next((i for i in _intents.get("intents", []) if i.get("tag")=="greeting"), None)
            if greet_intent:
                return random.choice(greet_intent.get("responses", [])), {"best_tag":"greeting","reason":"Early greeting route.","best_score":None}
        
        # REMOVED: Early general info/award route to stop misclassification of short high-value noun queries
        pass # Intentional removal of the old `is_simple_general_or_award` block

    # Basic detectors used downstream
    mentions_university = "university" in user_tokens
    mentions_president = "president" in user_tokens
    # Broaden institution mention for founders/presidents
    mentions_institution_for_founders = any(t in user_tokens for t in {"university","northwestern","nwu","academy"})
    mentions_institution_for_president = any(t in user_tokens for t in {"university","northwestern","nwu","college"})
    founder_query_tokens = {"founder","founders","founded","find","cofounder","co-founder","incorporator"}
    is_founder_query_base = any(t in user_tokens for t in founder_query_tokens)
    status_core = {"become","became","status","recognized","recognition","confirmation","confirm","year"}
    is_university_status_query = ("university" in user_tokens) and any(t in user_tokens for t in status_core)

    # Site/status/building/general detectors (also used by fallback later)
    is_status_like = ("university" in user_tokens) and any(t in user_tokens for t in status_core)
    site_terms = {"site","campus","airport","avenue","hectare","new"}
    is_new_site_like = any(t in user_tokens for t in site_terms) and ("site" in user_tokens or "campus" in user_tokens)
    is_president_like = ("president" in user_tokens) or ("led" in user_tokens) or ("leader" in user_tokens) or ("head" in user_tokens)
    is_current_time_like = any(t in user_tokens for t in {"current","present","incumbent","sitting","now","today","right"})
    is_founders_like = any(t in user_tokens for t in {"founder","founders","incorporators","cofounders","co-founders"})
    is_greeting_like = any(t in user_tokens for t in {"hi","hello","hey","greetings","good","day","help"})
    # Make greeting robust: "how are you", "what's up/whats"
    if ({"how","are","you"} <= user_tokens) or (("what" in user_tokens) and ("up" in user_tokens)) or ("whats" in user_tokens):
        is_greeting_like = True
    is_general_info_like = (("what" in user_tokens and "university" in user_tokens) or ("tell" in user_tokens and "university" in user_tokens) or ("northwestern" in user_tokens and "university" in user_tokens))
    is_buildings_like = any(t in user_tokens for t in {"buildings","structures","timeline","completed","major","campus"}) and not any(t in user_tokens for t in {"student","worship","aquino","sc"})
    is_barangan_like = any(t in user_tokens for t in {"barangan","cresencio","cashier","funds","finance"})
    # NEW: landmark-like
    is_landmark_like = any(t in user_tokens for t in {"landmark","landmarks","historical","historic","site","sites"})

    # NEW: detector that targets college leadership specifically (used to route to complete list)
    college_leadership_focus = (
        ("college" in user_tokens) and
        any(t in user_tokens for t in {"who","led","leader","head", "first"})
    )

    # NEW: add missing detectors used later
    is_albano_query = any(t in user_tokens for t in {"angel","albano"})
    process_terms = {"convert","conversion","transition","process","steps","petition","apply","application","recognition","recognized","approval","approved","sec","decs","ched"}
    is_transition_process_like = any(t in user_tokens for t in process_terms) and any(t in user_tokens for t in {"college","university","northwestern","nwu", "academy", "how"})

    # NEW: add missing detectors used below
    # Buildings
    is_landmark_query = is_landmark_like
    
    # Motto, logo
    is_motto_query = ("fiat" in user_tokens and "lux" in user_tokens) or ("motto" in user_tokens) or ("let" in user_tokens and "light" in user_tokens)
    is_logo_query = any(t in user_tokens for t in {"logo", "symbol", "seal", "emblem", "mascot", "owl"})
    buildings_overview_tokens = {"buildings", "structures", "timeline", "completed", "major", "campus"}
    is_buildings_overview_query = any(t in user_tokens for t in buildings_overview_tokens)
    is_new_site_query = (("site" in user_tokens or "campus" in user_tokens) and any(t in user_tokens for t in {"airport", "avenue", "hectare", "new"}))

    # Greeting/general/status aliases + barangan alias
    is_greeting_query = early_greet or is_greeting_like
    is_general_info_query = is_general_info_like
    is_status_query = is_university_status_query
    
    # NEW Early History Flags for MISS 1, 2, 3, 4
    is_foundation_phrase = any(t in user_tokens for t in {"founded", "foundation"})
    is_beginnings_query = any(t in user_tokens for t in {"beginnings", "about"}) and ("northwestern" in user_tokens or "nwu" in user_tokens) and len(user_tokens) < 6
    
    # NEW: barangan alias to avoid NameError in later routing
    is_barangan_query = is_barangan_like

    # NEW: strong detector for "what year did NWU become a university" (punctuation-insensitive)
    strong_univ_year_status = (
        ("university" in user_tokens) and
        ("year" in user_tokens) and
        any(t in user_tokens for t in {"become","became","recognized","recognition","status","confirmation","confirm"})
    )
    # NEW: support "when did ... become a university" phrasing (punctuation-insensitive)
    strong_univ_when_status = (
        ("university" in user_tokens) and
        ("when" in user_tokens) and
        any(t in user_tokens for t in {"become","became","recognized","recognition","status","confirmation","confirm"})
    )
    # NEW: unified detector for both variants
    strong_univ_status = strong_univ_year_status or strong_univ_when_status

    # NEW: strong detector for founders establish list phrasing
    strong_founders_establish_list = (
        any(t in user_tokens for t in {"name","everyone","all","list"}) and
        any(t in user_tokens for t in {"helped","establish","established","incorporators","founders"}) and
        any(t in user_tokens for t in {"academy","northwestern","nwu"})
    )

    # NEW: strong detector for “When did Northwestern Academy become a college”
    strong_academy_become_college = (
        ("academy" in user_tokens) and
        ("college" in user_tokens) and
        any(t in user_tokens for t in {"become","became","becoming","transition","convert","conversion","when","year","date"})
    )

    # NEW: detector for “Who led NWU when it became a college”
    leadership_during_college_transition = (
        any(t in user_tokens for t in {"led","leader","head","who", "first"}) and
        ("college" in user_tokens) and
        any(t in user_tokens for t in {"became","become","transition","when","year","date"})
    )

    # Founders/presidents composed (extend current-president to head/leader)
    is_current_president_query = (
        ("president" in user_tokens) and any(t in user_tokens for t in {"current","present","incumbent","sitting","now","today","right"}) and mentions_institution_for_president
    ) or (
        any(t in user_tokens for t in {"current","present","incumbent","sitting","now","today","right"}) and
        any(t in user_tokens for t in {"head","leader"}) and
        any(t in user_tokens for t in {"university","northwestern","nwu"})
    )
    # UPDATED: is_presidents_query now targets complete list
    is_presidents_query = (("president" in user_tokens) or ("presidents" in user_tokens)) and mentions_institution_for_president and not is_current_president_query
    is_all_presidents_like = any(t in user_tokens for t in {"all", "list", "past", "first", "college"}) and is_president_like
    is_founders_query = is_founder_query_base and mentions_institution_for_founders
    founders_list_terms = {"list", "name", "founders", "incorporators", "co-founders", "cofounders", "ten"}
    is_founders_list_query = any(t in user_tokens for t in founders_list_terms) and is_founders_query

    # Add missing first-president and leadership phrase detectors (now route to complete list)
    is_first_president_query = (("first" in user_tokens and "president" in user_tokens) or ("founding" in user_tokens and "president" in user_tokens))
    generic_leader_terms = {"led", "leader", "head"}
    is_generic_leadership_phrase = any(t in user_tokens for t in generic_leader_terms) and any(t in user_tokens for t in {"university","nwu","northwestern","college"})
    # NEW: generic president query (for "Who is the president?" to push to current)
    is_generic_president_query = is_president_like and is_who_query and not is_current_time_like and not is_all_presidents_like and not any(t in user_tokens for t in {"first", "past", "list", "all", "of", "college"})


    # Stronger founders/founded detector and conflict sets
    is_founded_nwu_like = (
        any(t in user_tokens for t in {"founded","founder","founders","incorporators"}) and
        not any(t in user_tokens for t in {"college","establishment"})
    )
    CONFLICT_INTENT_SETS = [
        {"northwestern_academy_incorporators","foundation"},
        {"complete_northwestern_presidents_list", "northwestern_current_president"},
        {"northwestern_classroom_icons","northwestern_faculty_mentors"},
        {"northwestern_student_activists","northwestern_martial_law", "student_activism"},
        {"early_years","major_transitions"},
        {"northwestern_academy_early_sacrifices", "nurturing_years", "early_years"},
        {"nicolas_contribution", "northwestern_faculty_mentors", "northwestern_academy_incorporators"},
        {"transition_process", "northwestern_college_courses", "northwestern_academy_commonwealth_era"},
        {"northwestern_college_courses", "Accreditation", "Deregulated_Status", "major_transitions"}, # For course/status confusion
        {"general_info", "northwestern_academy_incorporators", "northwestern_college_courses", "northwestern_academy_open_admission"} # For general info confusion
    ]

    # Missing disambiguators used later (define them here)
    is_foundation_when_query = is_foundation_phrase and is_when_query
    is_founders_who_query = is_founder_query_base and is_who_query
    
    # Nicolas-specific detectors
    nicolas_dedication_keywords = {"mr", "title", "called", "referred", "earned", "dedication", "unmatched", "'mr.", "teacher", "instructor", "faculty", "known"}
    is_nicolas_teacher_like = any(t in user_tokens for t in {"teacher", "instructor", "faculty", "known"}) and ("nicolas" in user_tokens)
    is_nicolas_who_in_college = (("nicolas" in user_tokens) and is_who_query and ("college" in user_tokens))
    is_nicolas_contrib_like = any(t in user_tokens for t in {"nicolas","founder"}) and any(t in user_tokens for t in {"contribution", "contributions", "do", "did", "help", "impact", "expansion", "role"})
    is_nicolas_what_did_do = is_what_query and any(t in user_tokens for t in ["did", "do"]) and is_nicolas_contrib_like # NEW DETECTOR
    is_caday_relationship_like = any(t in user_tokens for t in ["maximo", "caday", "relationship", "colleagues", "get along"])

    # NEW: academy early sacrifices like
    is_academy_sacrifices_like = any(t in user_tokens for t in {"sacrifices", "goal", "vision"}) and ("academy" in user_tokens or is_founders_like)

    # Academy phase/program detectors
    is_nurturing_years_like = (any(t in user_tokens for t in {"early","beginnings","like","where","held","challenges","face"}) and (mentions_institution_for_founders)) and not is_academy_sacrifices_like
    is_operating_like = any(t in user_tokens for t in {"operating","operate","start","started","begin","began","location","located","held"}) and not any(t in user_tokens for t in {"sacrifices", "goal", "vision"})
    is_commonwealth_planning_like = any(t in user_tokens for t in {"planning","expand","expansion","programs","courses","why", "commonwealth", "1935", "constitution", "surge", "affect"})
    is_engineering_program_like = "engineering" in user_tokens and any(t in user_tokens for t in {"program", "courses", "dean", "flagship", "engineering"})
    is_generic_course_query = any(t in user_tokens for t in {"courses", "programs", "degree", "associate"}) and (mentions_institution_for_president) and not is_engineering_program_like
    is_transition_process_like = any(t in user_tokens for t in process_terms) and any(t in user_tokens for t in {"college","university","northwestern","nwu", "academy", "how"})

    # Access policy disambiguator
    is_access_poor_like = any(t in user_tokens for t in {"poor","needy","scholarship","scholarships","help","assist","students"})

    # Nurturing years/date hint
    has_1932 = "1932" in user_tokens

    # Alias to avoid NameError later
    is_barangan_query = is_barangan_like

    # NEW: strong detector for activism details
    is_activist_details_query = any(t in user_tokens for t in {"velasco", "pascual", "became", "support", "notable", "leaders", "marcos"})

    # 1) Hard keyword override scoring (soft preference)
    forced_index = None
    max_overlap = 0
    preferred_forced_tag = None
    for i, meta in enumerate(_pattern_meta):
        try:
            orig_intent = next((it for it in _intents.get("intents", []) if it.get("tag") == meta["tag"]), {})
            boost_keywords = set([k.lower() for k in orig_intent.get("boost_keywords", [])])
            base_keywords = set([k.lower() for k in orig_intent.get("keywords", [])])
            priority = int(orig_intent.get("priority", 0))
        except Exception:
            boost_keywords, base_keywords = set(), set()
            priority = 0

        overlap_boost = len(user_tokens.intersection(boost_keywords))
        overlap_base = len(user_tokens.intersection(base_keywords))

        has_specific_detector = False
        tag = meta.get("tag")
        
        if tag in {"northwestern_fiat_lux_meaning"} and is_motto_query: has_specific_detector = True
        if tag in {"northwestern_current_president"} and is_current_president_query: has_specific_detector = True
        if tag in {"complete_northwestern_presidents_list"} and is_all_presidents_like: has_specific_detector = True
        if tag in {"northwestern_academy_incorporators"} and (is_founders_query or is_founders_list_query): has_specific_detector = True
        if tag in {"major_transitions"} and is_status_query: has_specific_detector = True
        effective_priority = priority if (overlap_base + overlap_boost > 0 or has_specific_detector) else 0

        score = overlap_boost * 2 + overlap_base + effective_priority

        # Tight routing rules
        if is_current_president_query:
            if tag == "northwestern_current_president": score += 9
            elif tag in {"complete_northwestern_presidents_list","major_transitions"}: score = max(0, score - 6)
        elif is_presidents_query or is_first_president_query or is_generic_leadership_phrase:
            if tag == "complete_northwestern_presidents_list": score += 9
            elif tag in {"northwestern_current_president","major_transitions"}: score = max(0, score - 6)

        # NEW: Route generic "president" queries to current president
        if is_generic_president_query:
            if tag == "northwestern_current_president": score += 8
            if tag == "complete_northwestern_presidents_list": score -= 5
            
        if is_founders_query or is_founders_list_query:
            if tag == "northwestern_academy_incorporators": score += 9
            elif tag in {"major_transitions","northwestern_academy_open_admission","northwestern_academy_early_sacrifices"}: score = max(0, score - 6)

        if is_status_query:
            if tag == "major_transitions": score += 8
        else:
            if tag == "major_transitions" and (is_presidents_query or is_current_president_query or is_first_president_query or is_generic_leadership_phrase or is_founders_query or is_founders_list_query or is_logo_query or is_greeting_query or is_general_info_query):
                score = max(0, score - 7)
        
        # Removed individual building logic, now covered by campus_historical_landmarks

        if is_motto_query:
            if tag == "northwestern_fiat_lux_meaning": score += 7
            elif tag in {"angel_albano_history","nicolas_contribution","northwestern_academy_incorporators"}: score = max(0, score - 5)

        if is_landmark_query:
            if tag == "campus_historical_landmarks" and not (is_motto_query or is_current_president_query or is_presidents_query or is_founders_query or is_founders_list_query or is_status_query):
                score += 4
            if tag == "northwestern_engineering_success_and_impact": score = max(0, score - 6)
        else:
            if tag == "campus_historical_landmarks": score = max(0, score - 4)

        if ("first" in user_tokens and "president" in user_tokens) or any(t in user_tokens for t in generic_leader_terms):
            if tag == "complete_northwestern_presidents_list": score += 7
            elif tag in {"major_transitions","northwestern_current_president"}: score = max(0, score - 5)

        if tag == "major_transitions" and not is_status_query:
            score = max(0, score - 6)

        if is_general_info_query:
            if tag == "general_info": score += 8
            elif tag in {"northwestern_new_school_site","major_transitions", "northwestern_academy_incorporators", "northwestern_college_courses"}: score = max(0, score - 6)

        if is_greeting_query:
            if tag == "greeting": score += 9
            elif tag in {"northwestern_new_school_site","major_transitions"}: score = max(0, score - 6)

        if is_buildings_overview_query:
            if tag == "buildings": score += 7
            elif tag == "campus_historical_landmarks": score = max(0, score - 5)

        if is_barangan_query:
            if tag == "cresencio_barangan_history": score += 8
            elif tag in {"angel_albano_history","northwestern_academy_incorporators"}: score = max(0, score - 6)
            
        # FIX START: Replaced dangling elifs with proper if/elif blocks
        # NEW: Nicolas contribution routing
        if is_nicolas_contrib_like and tag == "nicolas_contribution":
            score += 8
        elif is_nicolas_contrib_like and tag in {"northwestern_academy_incorporators", "northwestern_faculty_mentors"}:
            score = max(0, score - 4)

        # NEW: Nicolas teacher routing
        if is_nicolas_teacher_like and tag == "northwestern_faculty_mentors":
            score += 8
        elif is_nicolas_teacher_like and tag in {"nicolas_contribution", "northwestern_academy_incorporators"}:
            score = max(0, score - 4)

        # NEW: Maximo Caday routing
        if is_caday_relationship_like:
            if tag == "maximo_caday_relationship_with_founders": 
                score += 8
            elif tag == "nicolas_contribution": 
                score = max(0, score - 6)

        # NEW: Sacrifices routing
        if is_academy_sacrifices_like and tag == "northwestern_academy_early_sacrifices":
            score += 8
        elif is_academy_sacrifices_like and tag in {"early_years", "nurturing_years"}: 
            score = max(0, score - 4)

        # NEW: Commonwealth Era routing
        if is_commonwealth_planning_like and tag == "northwestern_academy_commonwealth_era":
            score += 8
        elif is_commonwealth_planning_like and tag in {"early_years", "northwestern_college_courses", "Accreditation"}: 
            score = max(0, score - 6)

        # NEW: Transition Process routing
        if is_transition_process_like and tag == "transition_process":
            score += 8
        elif is_transition_process_like and tag in {"early_years", "northwestern_college_courses", "Accreditation"}: 
            score = max(0, score - 6)
            
        # NEW: Nurturing Years routing
        if is_nurturing_years_like and tag == "nurturing_years":
            score += 8
        elif is_nurturing_years_like and tag == "early_years": 
            score = max(0, score - 6)

        # NEW: Student Activism routing
        if is_activist_details_query and tag == "northwestern_student_activists":
            score += 8
        elif is_activist_details_query and tag == "northwestern_martial_law": 
            score = max(0, score - 4)
            
        if any(t in user_tokens for t in {"student", "leaders", "marcos", "protest"}):
            if tag == "student_activism": score += 4
            elif tag == "northwestern_faculty_mentors": score = max(0, score - 4)
        
        # NEW: Generic Courses routing
        if is_generic_course_query and tag == "northwestern_college_courses":
            score += 8
            # Penalty logic for generic courses stealing specific engineering is removed as the specific engineering intent is removed.
            elif tag in {"Accreditation", "Deregulated_Status"}: score = max(0, score - 6)
        # FIX END
            
        # NEW: Massive penalty for Commonwealth stealing Incorporators
        if (is_founders_like or "incorporator" in user_tokens) and tag == "northwestern_academy_commonwealth_era":
             score -= 5.0
        # NEW: Massive penalty for Incorporators stealing Commonwealth
        if is_commonwealth_planning_like and tag == "northwestern_academy_incorporators":
             score -= 5.0
        # NEW: Penalty for Presidents List stealing Founders
        if (is_founders_like or "incorporator" in user_tokens) and tag == "complete_northwestern_presidents_list":
             score -= 5.0


        # Pick best
        if score > best_score:
            best_score = score
            best_match = intent

    if best_match and best_score >= min_overlap:
        # return response AND tag selected by fallback
        return random.choice(best_match.get("responses", [])), (best_match.get("tag") or "")
    return None, None

def build_all_tests_from_intents(intents_data):
    tests = []
    # EXCLUDED TAGS REMOVED as per request.
    
    for intent in intents_data.get("intents", []):
        tag = intent.get("tag")
        examples = intent.get("examples") or intent.get("patterns", [])
        for ex in examples:
            if isinstance(ex, str) and ex.strip():
                tests.append({"q": ex.strip(), "tag": tag})
    return tests

# Define evaluator BEFORE UI code
def run_offline_eval():
    # Deterministic sampling for consistent eval
    random.seed(42)
    # Use ALL examples from intents.json
    tests = build_all_tests_from_intents(_intents) # Use global _intents

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

# --------------------------
# --- Semantic matcher with debug ---
# --------------------------
def get_semantic_response_debug(user_input: str, eval_mode: bool = False):
    if not _intents.get("intents"):
        return "Chatbot data is unavailable.", None

    user_processed = _preprocess(user_input)
    if not user_processed.strip():
        return "Could you rephrase your question?", None

    if _pattern_embeddings is None or len(_pattern_meta) == 0:
        return "No knowledge available at the moment.", None

    user_tokens = set(_tokenizer.tokenize(user_input.lower()))

    # NEW: question-word detectors (must be defined before later use)
    is_who_query = "who" in user_tokens
    is_when_query = any(t in user_tokens for t in {"when","year","date"})
    is_what_query = "what" in user_tokens

    # --- Early short-circuit for greetings/general info/awards ---
    # Replace permissive greeting with robust combos
    early_greet = (any(t in user_tokens for t in {"hi","hello","hey","greetings"})
                    or ({"how","are","you"} <= user_tokens)
                    or ("whats" in user_tokens) or (("what" in user_tokens) and ("up" in user_tokens)))
    if len(user_tokens) <= 3:
        if early_greet:
            greet_intent = next((i for i in _intents.get("intents", []) if i.get("tag")=="greeting"), None)
            if greet_intent:
                return random.choice(greet_intent.get("responses", [])), {"best_tag":"greeting","reason":"Early greeting route.","best_score":None}
        
        # REMOVED: Early general info/award route to stop misclassification of short high-value noun queries
        pass # Intentional removal of the old `is_simple_general_or_award` block

    # Basic detectors used downstream
    mentions_university = "university" in user_tokens
    mentions_president = "president" in user_tokens
    # Broaden institution mention for founders/presidents
    mentions_institution_for_founders = any(t in user_tokens for t in {"university","northwestern","nwu","academy"})
    mentions_institution_for_president = any(t in user_tokens for t in {"university","northwestern","nwu","college"})
    founder_query_tokens = {"founder","founders","founded","find","cofounder","co-founder","incorporator"}
    is_founder_query_base = any(t in user_tokens for t in founder_query_tokens)
    status_core = {"become","became","status","recognized","recognition","confirmation","confirm","year"}
    is_university_status_query = ("university" in user_tokens) and any(t in user_tokens for t in status_core)

    # Site/status/building/general detectors (also used by fallback later)
    is_status_like = ("university" in user_tokens) and any(t in user_tokens for t in status_core)
    site_terms = {"site","campus","airport","avenue","hectare","new"}
    is_new_site_like = any(t in user_tokens for t in site_terms) and ("site" in user_tokens or "campus" in user_tokens)
    is_president_like = ("president" in user_tokens) or ("led" in user_tokens) or ("leader" in user_tokens) or ("head" in user_tokens)
    is_current_time_like = any(t in user_tokens for t in {"current","present","incumbent","sitting","now","today","right"})
    is_founders_like = any(t in user_tokens for t in {"founder","founders","incorporators","cofounders","co-founders"})
    is_greeting_like = any(t in user_tokens for t in {"hi","hello","hey","greetings","good","day","help"})
    # Make greeting robust: "how are you", "what's up/whats"
    if ({"how","are","you"} <= user_tokens) or (("what" in user_tokens) and ("up" in user_tokens)) or ("whats" in user_tokens):
        is_greeting_like = True
    is_general_info_like = (("what" in user_tokens and "university" in user_tokens) or ("tell" in user_tokens and "university" in user_tokens) or ("northwestern" in user_tokens and "university" in user_tokens))
    is_buildings_like = any(t in user_tokens for t in {"buildings","structures","timeline","completed","major","campus"}) and not any(t in user_tokens for t in {"student","worship","aquino","sc"})
    is_barangan_like = any(t in user_tokens for t in {"barangan","cresencio","cashier","funds","finance"})
    # NEW: landmark-like
    is_landmark_like = any(t in user_tokens for t in {"landmark","landmarks","historical","historic","site","sites"})

    # NEW: detector that targets college leadership specifically (used to route to complete list)
    college_leadership_focus = (
        ("college" in user_tokens) and
        any(t in user_tokens for t in {"who","led","leader","head", "first"})
    )

    # NEW: add missing detectors used later
    is_albano_query = any(t in user_tokens for t in {"angel","albano"})
    process_terms = {"convert","conversion","transition","process","steps","petition","apply","application","recognition","recognized","approval","approved","sec","decs","ched"}
    is_transition_process_like = any(t in user_tokens for t in process_terms) and any(t in user_tokens for t in {"college","university","northwestern","nwu", "academy", "how"})

    # NEW: add missing detectors used below
    # Buildings
    is_landmark_query = is_landmark_like
    
    # Motto, logo
    is_motto_query = ("fiat" in user_tokens and "lux" in user_tokens) or ("motto" in user_tokens) or ("let" in user_tokens and "light" in user_tokens)
    is_logo_query = any(t in user_tokens for t in {"logo", "symbol", "seal", "emblem", "mascot", "owl"})
    buildings_overview_tokens = {"buildings", "structures", "timeline", "completed", "major", "campus"}
    is_buildings_overview_query = any(t in user_tokens for t in buildings_overview_tokens)
    is_new_site_query = (("site" in user_tokens or "campus" in user_tokens) and any(t in user_tokens for t in {"airport", "avenue", "hectare", "new"}))

    # Greeting/general/status aliases + barangan alias
    is_greeting_query = early_greet or is_greeting_like
    is_general_info_query = is_general_info_like
    is_status_query = is_university_status_query
    
    # NEW Early History Flags for MISS 1, 2, 3, 4
    is_foundation_phrase = any(t in user_tokens for t in {"founded", "foundation"})
    is_beginnings_query = any(t in user_tokens for t in {"beginnings", "about"}) and ("northwestern" in user_tokens or "nwu" in user_tokens) and len(user_tokens) < 6
    
    # NEW: barangan alias to avoid NameError in later routing
    is_barangan_query = is_barangan_like

    # NEW: strong detector for "what year did NWU become a university" (punctuation-insensitive)
    strong_univ_year_status = (
        ("university" in user_tokens) and
        ("year" in user_tokens) and
        any(t in user_tokens for t in {"become","became","recognized","recognition","status","confirmation","confirm"})
    )
    # NEW: support "when did ... become a university" phrasing (punctuation-insensitive)
    strong_univ_when_status = (
        ("university" in user_tokens) and
        ("when" in user_tokens) and
        any(t in user_tokens for t in {"become","became","recognized","recognition","status","confirmation","confirm"})
    )
    # NEW: unified detector for both variants
    strong_univ_status = strong_univ_year_status or strong_univ_when_status

    # NEW: strong detector for founders establish list phrasing
    strong_founders_establish_list = (
        any(t in user_tokens for t in {"name","everyone","all","list"}) and
        any(t in user_tokens for t in {"helped","establish","established","incorporators","founders"}) and
        any(t in user_tokens for t in {"academy","northwestern","nwu"})
    )

    # NEW: strong detector for “When did Northwestern Academy become a college”
    strong_academy_become_college = (
        ("academy" in user_tokens) and
        ("college" in user_tokens) and
        any(t in user_tokens for t in {"become","became","becoming","transition","convert","conversion","when","year","date"})
    )

    # NEW: detector for “Who led NWU when it became a college”
    leadership_during_college_transition = (
        any(t in user_tokens for t in {"led","leader","head","who", "first"}) and
        ("college" in user_tokens) and
        any(t in user_tokens for t in {"became","become","transition","when","year","date"})
    )

    # Founders/presidents composed (extend current-president to head/leader)
    is_current_president_query = (
        ("president" in user_tokens) and any(t in user_tokens for t in {"current","present","incumbent","sitting","now","today","right"}) and mentions_institution_for_president
    ) or (
        any(t in user_tokens for t in {"current","present","incumbent","sitting","now","today","right"}) and
        any(t in user_tokens for t in {"head","leader"}) and
        any(t in user_tokens for t in {"university","northwestern","nwu"})
    )
    # UPDATED: is_presidents_query now targets complete list
    is_presidents_query = (("president" in user_tokens) or ("presidents" in user_tokens)) and mentions_institution_for_president and not is_current_president_query
    is_all_presidents_like = any(t in user_tokens for t in {"all", "list", "past", "first", "college"}) and is_president_like
    is_founders_query = is_founder_query_base and mentions_institution_for_founders
    founders_list_terms = {"list", "name", "founders", "incorporators", "co-founders", "cofounders", "ten"}
    is_founders_list_query = any(t in user_tokens for t in founders_list_terms) and is_founders_query

    # Add missing first-president and leadership phrase detectors (now route to complete list)
    is_first_president_query = (("first" in user_tokens and "president" in user_tokens) or ("founding" in user_tokens and "president" in user_tokens))
    generic_leader_terms = {"led", "leader", "head"}
    is_generic_leadership_phrase = any(t in user_tokens for t in generic_leader_terms) and any(t in user_tokens for t in {"university","nwu","northwestern","college"})
    # NEW: generic president query (for "Who is the president?" to push to current)
    is_generic_president_query = is_president_like and is_who_query and not is_current_time_like and not is_all_presidents_like and not any(t in user_tokens for t in {"first", "past", "list", "all", "of", "college"})


    # Stronger founders/founded detector and conflict sets
    is_founded_nwu_like = (
        any(t in user_tokens for t in {"founded","founder","founders","incorporators"}) and
        not any(t in user_tokens for t in {"college","establishment"})
    )
    CONFLICT_INTENT_SETS = [
        {"northwestern_academy_incorporators","foundation"},
        {"complete_northwestern_presidents_list", "northwestern_current_president"},
        {"northwestern_classroom_icons","northwestern_faculty_mentors"},
        {"northwestern_student_activists","northwestern_martial_law", "student_activism"},
        {"early_years","major_transitions"},
        {"northwestern_academy_early_sacrifices", "nurturing_years", "early_years"},
        {"nicolas_contribution", "northwestern_faculty_mentors", "northwestern_academy_incorporators"},
        {"transition_process", "northwestern_college_courses", "northwestern_academy_commonwealth_era"},
        {"northwestern_college_courses", "Accreditation", "Deregulated_Status", "major_transitions"}, # For course/status confusion
        {"general_info", "northwestern_academy_incorporators", "northwestern_college_courses", "northwestern_academy_open_admission"} # For general info confusion
    ]

    # Missing disambiguators used later (define them here)
    is_foundation_when_query = is_foundation_phrase and is_when_query
    is_founders_who_query = is_founder_query_base and is_who_query
    
    # Nicolas-specific detectors
    nicolas_dedication_keywords = {"mr", "title", "called", "referred", "earned", "dedication", "unmatched", "'mr.", "teacher", "instructor", "faculty", "known"}
    is_nicolas_teacher_like = any(t in user_tokens for t in {"teacher", "instructor", "faculty", "known"}) and ("nicolas" in user_tokens)
    is_nicolas_who_in_college = (("nicolas" in user_tokens) and is_who_query and ("college" in user_tokens))
    is_nicolas_contrib_like = any(t in user_tokens for t in {"nicolas","founder"}) and any(t in user_tokens for t in {"contribution", "contributions", "do", "did", "help", "impact", "expansion", "role"})
    is_nicolas_what_did_do = is_what_query and any(t in user_tokens for t in ["did", "do"]) and is_nicolas_contrib_like # NEW DETECTOR
    is_caday_relationship_like = any(t in user_tokens for t in ["maximo", "caday", "relationship", "colleagues", "get along"])

    # NEW: academy early sacrifices like
    is_academy_sacrifices_like = any(t in user_tokens for t in {"sacrifices", "goal", "vision"}) and ("academy" in user_tokens or is_founders_like)

    # Academy phase/program detectors
    is_nurturing_years_like = (any(t in user_tokens for t in {"early","beginnings","like","where","held","challenges","face"}) and (mentions_institution_for_founders)) and not is_academy_sacrifices_like
    is_operating_like = any(t in user_tokens for t in {"operating","operate","start","started","begin","began","location","located","held"}) and not any(t in user_tokens for t in {"sacrifices", "goal", "vision"})
    is_commonwealth_planning_like = any(t in user_tokens for t in {"planning","expand","expansion","programs","courses","why", "commonwealth", "1935", "constitution", "surge", "affect"})
    is_engineering_program_like = "engineering" in user_tokens and any(t in user_tokens for t in {"program", "courses", "dean", "flagship", "engineering"})
    is_generic_course_query = any(t in user_tokens for t in {"courses", "programs", "degree", "associate"}) and (mentions_institution_for_president) and not is_engineering_program_like
    is_transition_process_like = any(t in user_tokens for t in process_terms) and any(t in user_tokens for t in {"college","university","northwestern","nwu", "academy", "how"})

    # Access policy disambiguator
    is_access_poor_like = any(t in user_tokens for t in {"poor","needy","scholarship","scholarships","help","assist","students"})

    # Nurturing years/date hint
    has_1932 = "1932" in user_tokens

    # Alias to avoid NameError later
    is_barangan_query = is_barangan_like

    # NEW: strong detector for activism details
    is_activist_details_query = any(t in user_tokens for t in {"velasco", "pascual", "became", "support", "notable", "leaders", "marcos"})

    # 1) Hard keyword override scoring (soft preference)
    forced_index = None
    max_overlap = 0
    preferred_forced_tag = None
    for i, meta in enumerate(_pattern_meta):
        try:
            orig_intent = next((it for it in _intents.get("intents", []) if it.get("tag") == meta["tag"]), {})
            boost_keywords = set([k.lower() for k in orig_intent.get("boost_keywords", [])])
            base_keywords = set([k.lower() for k in orig_intent.get("keywords", [])])
            priority = int(orig_intent.get("priority", 0))
        except Exception:
            boost_keywords, base_keywords = set(), set()
            priority = 0

        overlap_boost = len(user_tokens.intersection(boost_keywords))
        overlap_base = len(user_tokens.intersection(base_keywords))

        has_specific_detector = False
        tag = meta.get("tag")
        
        if tag in {"northwestern_fiat_lux_meaning"} and is_motto_query: has_specific_detector = True
        if tag in {"northwestern_current_president"} and is_current_president_query: has_specific_detector = True
        if tag in {"complete_northwestern_presidents_list"} and is_all_presidents_like: has_specific_detector = True
        if tag in {"northwestern_academy_incorporators"} and (is_founders_query or is_founders_list_query): has_specific_detector = True
        if tag in {"major_transitions"} and is_status_query: has_specific_detector = True
        effective_priority = priority if (overlap_base + overlap_boost > 0 or has_specific_detector) else 0

        score = overlap_boost * 2 + overlap_base + effective_priority

        # Tight routing rules
        if is_current_president_query:
            if tag == "northwestern_current_president": score += 9
            elif tag in {"complete_northwestern_presidents_list","major_transitions"}: score = max(0, score - 6)
        elif is_presidents_query or is_first_president_query or is_generic_leadership_phrase:
            if tag == "complete_northwestern_presidents_list": score += 9
            elif tag in {"northwestern_current_president","major_transitions"}: score = max(0, score - 6)

        # NEW: Route generic "president" queries to current president
        if is_generic_president_query:
            if tag == "northwestern_current_president": score += 8
            if tag == "complete_northwestern_presidents_list": score -= 5
            
        if is_founders_query or is_founders_list_query:
            if tag == "northwestern_academy_incorporators": score += 9
            elif tag in {"major_transitions","northwestern_academy_open_admission","northwestern_academy_early_sacrifices"}: score = max(0, score - 6)

        if is_status_query:
            if tag == "major_transitions": score += 8
        else:
            if tag == "major_transitions" and (is_presidents_query or is_current_president_query or is_first_president_query or is_generic_leadership_phrase or is_founders_query or is_founders_list_query or is_logo_query or is_greeting_query or is_general_info_query):
                score = max(0, score - 7)
        
        # Removed individual building logic, now covered by campus_historical_landmarks

        if is_motto_query:
            if tag == "northwestern_fiat_lux_meaning": score += 7
            elif tag in {"angel_albano_history","nicolas_contribution","northwestern_academy_incorporators"}: score = max(0, score - 5)

        if is_landmark_query:
            if tag == "campus_historical_landmarks" and not (is_motto_query or is_current_president_query or is_presidents_query or is_founders_query or is_founders_list_query or is_status_query):
                score += 4
            if tag == "northwestern_engineering_success_and_impact": score = max(0, score - 6)
        else:
            if tag == "campus_historical_landmarks": score = max(0, score - 4)

        if ("first" in user_tokens and "president" in user_tokens) or any(t in user_tokens for t in generic_leader_terms):
            if tag == "complete_northwestern_presidents_list": score += 7
            elif tag in {"major_transitions","northwestern_current_president"}: score = max(0, score - 5)

        if tag == "major_transitions" and not is_status_query:
            score = max(0, score - 6)

        if is_general_info_query:
            if tag == "general_info": score += 8
            elif tag in {"northwestern_new_school_site","major_transitions", "northwestern_academy_incorporators", "northwestern_college_courses"}: score = max(0, score - 6)

        if is_greeting_query:
            if tag == "greeting": score += 9
            elif tag in {"northwestern_new_school_site","major_transitions"}: score = max(0, score - 6)

        if is_buildings_overview_query:
            if tag == "buildings": score += 7
            elif tag == "campus_historical_landmarks": score = max(0, score - 5)

        if is_barangan_query:
            if tag == "cresencio_barangan_history": score += 8
            elif tag in {"angel_albano_history","northwestern_academy_incorporators"}: score = max(0, score - 6)
            
        # FIX START: Converted dangling elif blocks to proper if/elif
        # NEW: Nicolas contribution routing
        if is_nicolas_contrib_like:
            if tag == "nicolas_contribution":
                score += 8
            elif tag in {"northwestern_academy_incorporators", "northwestern_faculty_mentors"}:
                 score = max(0, score - 4)

        # NEW: Nicolas teacher routing
        if is_nicolas_teacher_like:
            if tag == "northwestern_faculty_mentors":
                score += 8
            elif tag in {"nicolas_contribution", "northwestern_academy_incorporators"}: 
                score = max(0, score - 4)

        # NEW: Maximo Caday routing
        if is_caday_relationship_like:
            if tag == "maximo_caday_relationship_with_founders": 
                score += 8
            elif tag == "nicolas_contribution": 
                score = max(0, score - 6)

        # NEW: Sacrifices routing
        if is_academy_sacrifices_like:
            if tag == "northwestern_academy_early_sacrifices":
                score += 8
            elif tag in {"early_years", "nurturing_years"}: 
                score = max(0, score - 4)

        # NEW: Commonwealth Era routing
        if is_commonwealth_planning_like:
            if tag == "northwestern_academy_commonwealth_era":
                score += 8
            elif tag in {"early_years", "northwestern_college_courses", "Accreditation"}: 
                score = max(0, score - 6)

        # NEW: Transition Process routing
        if is_transition_process_like:
            if tag == "transition_process":
                score += 8
            elif tag in {"early_years", "northwestern_college_courses", "Accreditation"}: 
                score = max(0, score - 6)
            
        # NEW: Nurturing Years routing
        if is_nurturing_years_like:
            if tag == "nurturing_years":
                score += 8
            elif tag == "early_years": 
                score = max(0, score - 6)

        # NEW: Student Activism routing
        if is_activist_details_query:
            if tag == "northwestern_student_activists":
                score += 8
            elif tag == "northwestern_martial_law": 
                score = max(0, score - 4)
            
        if any(t in user_tokens for t in {"student", "leaders", "marcos", "protest"}):
            if tag == "student_activism": score += 4
            elif tag == "northwestern_faculty_mentors": score = max(0, score - 4)
        
        # NEW: Generic Courses routing
        if is_generic_course_query:
            if tag == "northwestern_college_courses":
                score += 8
                # Penalty logic for generic courses stealing specific engineering is removed as the specific engineering intent is removed.
            elif tag in {"Accreditation", "Deregulated_Status"}: score = max(0, score - 6)
        # FIX END
            
        # NEW: Massive penalty for Commonwealth stealing Incorporators
        if (is_founders_like or "incorporator" in user_tokens) and tag == "northwestern_academy_commonwealth_era":
             score -= 5.0
        # NEW: Massive penalty for Incorporators stealing Commonwealth
        if is_commonwealth_planning_like and tag == "northwestern_academy_incorporators":
             score -= 5.0
        # NEW: Penalty for Presidents List stealing Founders
        if (is_founders_like or "incorporator" in user_tokens) and tag == "complete_northwestern_presidents_list":
             score -= 5.0


        # Pick best
        if score > max_overlap:
            max_overlap = score
            forced_index = i

    # Only allow hard override on strong, specific detectors; never to general_info/greeting
    def _override_allowed(tag: str) -> bool:
        if tag in {"general_info","greeting"}:
            return False
        return any([
            tag == "northwestern_fiat_lux_meaning" and is_motto_query,
            tag == "northwestern_logo_symbolism" and is_logo_query,
            tag == "northwestern_current_president" and is_current_president_query,
            tag == "complete_northwestern_presidents_list" and (("president" in user_tokens) and (is_presidents_query or is_first_president_query or is_generic_leadership_phrase)),
            tag == "northwestern_academy_incorporators" and (is_founders_query or is_founders_who_query),
            tag == "foundation" and is_foundation_when_query,
            tag == "cresencio_barangan_history" and is_barangan_query,
            tag == "angel_albano_history" and is_albano_query,
            tag == "northwestern_new_school_site" and is_new_site_query,
            tag == "buildings" and is_buildings_overview_query,
            tag == "campus_historical_landmarks" and is_landmark_query,
            tag == "northwestern_college_courses" and is_generic_course_query,
            tag == "transition_process" and is_transition_process_like,
            tag == "nicolas_contribution" and is_nicolas_contrib_like,
            tag == "northwestern_faculty_mentors" and is_nicolas_teacher_like,
            tag == "northwestern_academy_early_sacrifices" and is_academy_sacrifices_like,
            tag == "nurturing_years" and is_nurturing_years_like,
            tag == "2004_Award" and any(t in user_tokens for t in ["award", "2004"]),
            tag == "northwestern_academy_commonwealth_era" and is_commonwealth_planning_like,
            tag == "northwestern_student_activists" and is_activist_details_query,
            tag == "maximo_caday_relationship_with_founders" and is_caday_relationship_like,
            tag == "major_transitions" and strong_univ_status,
            tag == "early_years" and user_input.lower() == "what was the school originally called?",
        ])

    HARD_OVERRIDE_THRESHOLD = 5
    if forced_index is not None and max_overlap >= HARD_OVERRIDE_THRESHOLD:
        forced_tag = _pattern_meta[forced_index]["tag"]
        if _override_allowed(forced_tag):
            # SOFT preference only; do not return early
            preferred_forced_tag = forced_tag
        # continue to semantic ranking

    # 2) Semantic similarity with recent context
    # NEW: freeze context for leadership and academy→college to avoid session drift on repeated asks
    freeze_context = (strong_academy_become_college or is_generic_leadership_phrase or is_presidents_query or leadership_during_college_transition or college_leadership_focus)
    if eval_mode or freeze_context:
        recent_context = ""
    else:
        recent_qs = st.session_state.get('recent_questions', [])
        recent_as = [msg["content"] for msg in st.session_state.get('history', []) if msg["role"]=="assistant"]
        recent_context = " ".join(recent_qs[-2:] + recent_as[-1:]) if recent_as else ""
    contextual_input = _preprocess(user_input + (" " + recent_context if recent_context else ""))

    with torch.no_grad():
        user_embedding = _model.encode([contextual_input], convert_to_tensor=True)
        # Diagram: Semantic Similarity Search 
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

        # Intent-specific reranker tuning
        if is_current_president_query:
            if tag == "northwestern_current_president": score += 0.45
            if tag in {"complete_northwestern_presidents_list", "major_transitions"}: score -= 0.35 
        if (is_presidents_query or is_first_president_query or is_generic_leadership_phrase):
            if tag == "complete_northwestern_presidents_list": score += 0.45 
            if tag == "northwestern_current_president": score -= 0.5 
            if tag in {"major_transitions"}: score -= 0.28
            
        # NEW: All presidents boost
        if is_all_presidents_like and tag == "complete_northwestern_presidents_list":
            score += 0.35
        # NEW: Generic president query push
        if is_generic_president_query:
            if tag == "northwestern_current_president": score += 0.45
            if tag == "complete_northwestern_presidents_list": score -= 0.35

        # HIGH PENALTY for Current Time queries stealing from List
        if is_current_time_like:
            if tag == "complete_northwestern_presidents_list": score -= 0.6 
            
        # Reranker on list should be neutral/slightly positive for "first college president"
        if ("first" in user_tokens and "college" in user_tokens and "president" in user_tokens):
             if tag == "complete_northwestern_presidents_list": score += 0.15


        if (is_founders_query or is_founders_list_query):
            if tag == "northwestern_academy_incorporators": score += 0.42
            if tag in {"major_transitions"}: score -= 0.32
            # NEW: Penalty against Founders landing on Presidents List
            if tag == "complete_northwestern_presidents_list": score -= 0.5

        if is_general_info_query:
            if tag == "general_info": score += 0.35
            if tag in {"major_transitions","northwestern_new_school_site", "northwestern_academy_incorporators", "northwestern_college_courses"}: score -= 0.3

        if is_greeting_query:
            if tag == "greeting": score += 0.35
            if tag in {"northwestern_new_school_site","major_transitions"}: score -= 0.28

        if is_buildings_overview_query:
            if tag == "buildings": score += 0.32
            if tag == "campus_historical_landmarks": score -= 0.26

        if is_landmark_query:
            if tag == "campus_historical_landmarks": score += 0.28
            if tag == "buildings": score -= 0.22
        else:
            if tag == "campus_historical_landmarks": score -= 0.26

        if is_barangan_query:
            if tag == "cresencio_barangan_history": score += 0.34
            if tag in {"major_transitions","northwestern_academy_incorporators"}: score -= 0.28

        # NEW: Nicolas contribution/role routing
        if is_nicolas_contrib_like:
            if tag == "nicolas_contribution": score += 0.45
            elif tag == "northwestern_faculty_mentors": score -= 0.3
            elif tag == "northwestern_academy_incorporators": score -= 0.3
        
        if is_nicolas_teacher_like: # What was he known for as a teacher (faculty_mentors)
             if tag == "northwestern_faculty_mentors": score += 0.45
             elif tag == "nicolas_contribution": score -= 0.3
             elif tag == "northwestern_academy_incorporators": score -= 0.3

        # NEW: Maximo Caday reranker
        if is_caday_relationship_like:
             if tag == "maximo_caday_relationship_with_founders": score += 0.45
             elif tag == "nicolas_contribution": score -= 0.35


        if not is_new_site_query and tag == "northwestern_new_school_site": score -= 0.34
        if is_new_site_query and tag == "northwestern_new_school_site": score += 0.26

        if not is_status_query and tag == "major_transitions": score -= 0.34
        if is_status_query and tag == "major_transitions": score += 0.26

        # Early years stealers guard
        if tag == "early_years" and any(t in user_tokens for t in {"establishment","college","courses","programs","engineering","first","president"}):
            score -= 0.38
        # NEW: Early Sacrifices reranker
        if any(t in user_tokens for t in {"sacrifices", "goal", "vision"}) and tag == "northwestern_academy_early_sacrifices":
            score += 0.4
        if any(t in user_tokens for t in {"sacrifices", "goal", "vision"}) and tag in {"early_years", "nurturing_years"}:
            score -= 0.3

        # Reranker Nudge 1: Final separation of EARLY HISTORY CLUSTER (MISS 1, 2, 3, 4)
        if is_foundation_phrase:
            if tag == "foundation":
                score += 0.30 # Massive boost for explicit foundation query
            if tag == "early_years":
                score -= 0.30 # Massive penalty on early_years stealing explicit foundation
        
        if is_beginnings_query:
            if tag == "early_years":
                score += 0.30 # Force generic 'beginnings' query to early_years
            if tag == "nurturing_years":
                score -= 0.20 # Penalize nurturing years for short ambiguous query
        
        # FINAL check on 1932/when queries (MISS 2, 3, 4)
        if has_1932 or is_when_query:
            if tag == "foundation": score += 0.20 # Reinforce foundation
            if tag == "early_years": score -= 0.20 # Penalize generic early years stealing date
        
        # Academy phases
        # STRONG BOOST if any commonwealth planning keywords are present, to overcome generic early_years steal
        if is_commonwealth_planning_like and tag == "northwestern_academy_commonwealth_era":
             score += 0.5
        if is_commonwealth_planning_like and tag in {"early_years", "northwestern_college_courses", "Accreditation"}:
            score -= 0.45
        # NEW: Massive penalty against Commonwealth stealing Incorporators
        if is_founders_like and tag == "northwestern_academy_commonwealth_era":
            score -= 0.6 
        
        # NEW: Transition Process reranker
        if is_transition_process_like and tag == "transition_process":
            score += 0.4
        if is_transition_process_like and tag in {"early_years", "northwestern_college_courses", "Accreditation"}:
            score -= 0.4

        # NEW: Nurturing years reranker
        if is_nurturing_years_like and tag == "nurturing_years":
            score += 0.4
            if is_operating_like: score += 0.15 # Extra nudge for "how did start operating" to nurturing
        if is_nurturing_years_like and tag == "early_years":
            score -= 0.3
            
        # Engineering reranker (Now routes to courses)
        if is_engineering_program_like and tag == "northwestern_college_courses":
             score += 0.3
        
        # Student Activists reranker
        if is_activist_details_query and tag == "northwestern_student_activists":
            score += 0.4
        if is_activist_details_query and tag == "northwestern_martial_law":
            score -= 0.3
        
        if any(t in user_tokens for t in {"student", "leaders", "marcos", "protest"}) and tag == "student_activism":
             score += 0.35 # Generic activism queries

        # NEW: who/when semantic nudges - INCREASED PENALTIES/BOOSTS
        if is_who_query:
            if tag in {"complete_northwestern_presidents_list","northwestern_current_president","northwestern_academy_incorporators", "cresencio_barangan_history", "northwestern_student_activists", "northwestern_faculty_mentors", "northwestern_classroom_icons"}:
                score += 0.35
            if tag in {"major_transitions","early_years","general_info", "foundation", "northwestern_college_courses"}:
                score -= 0.3
        if is_when_query:
            if tag in {"major_transitions","early_years","foundation", "transition_process", "northwestern_academy_commonwealth_era"}:
                score += 0.35
            if tag in {"complete_northwestern_presidents_list","northwestern_current_president","northwestern_academy_incorporators"}:
                score -= 0.3

        # Favor presidents for leadership during college transition (now routes to complete list)
        if leadership_during_college_transition:
            if tag == "complete_northwestern_presidents_list": score += 0.3
            if tag in {"early_years","major_transitions","general_info"}:
                score -= 0.26

        candidates.append((idx, score))

    candidates.sort(key=lambda x: x[1], reverse=True)

    # Preference picker
    def pick_if_present(prefer_tag_set, max_gap=0.12):
        current_best_idx, current_best_score = candidates[0]
        present = [(idx, sc) for (idx, sc) in candidates if _pattern_meta[idx]["tag"] in prefer_tag_set]
        if present:
            pref_idx, pref_sc = sorted(present, key=lambda x: x[1], reverse=True)[0]
            if pref_sc + 1e-6 >= current_best_score - max_gap:
                candidates[0] = (pref_idx, pref_sc)

    # Apply soft preference from early keyword pass
    if preferred_forced_tag:
        pick_if_present({preferred_forced_tag}, max_gap=0.22)

    # Existing picks
    # HIGH PRIORITY PICK: Current President if time word is used (to counter list steal)
    if is_current_president_query: 
        pick_if_present({"northwestern_current_president"}, max_gap=0.35) 
    # HIGH PRIORITY PICK: Complete List if "all" or "past" or "first college president" used
    if is_all_presidents_like:
        pick_if_present({"complete_northwestern_presidents_list"}, max_gap=0.35)
    # NEW: Generic President Query falls back to current
    if is_generic_president_query:
        pick_if_present({"northwestern_current_president"}, max_gap=0.3)


    if (is_presidents_query or is_first_president_query or is_generic_leadership_phrase): pick_if_present({"complete_northwestern_presidents_list"}, max_gap=0.25) 
    # FIX: Increased max_gap for founders/incorporators
    if (is_founders_query or is_founders_list_query): pick_if_present({"northwestern_academy_incorporators"}, max_gap=0.5) 
    # FIX: Increased max_gap for general_info to overcome semantic stealing
    if is_general_info_query: pick_if_present({"general_info"}, max_gap=0.35)
    
    # FINAL Pick: Ambiguous Early History Terms
    if is_foundation_phrase:
        pick_if_present({"foundation"}, max_gap=0.35)
    if is_beginnings_query:
        pick_if_present({"early_years"}, max_gap=0.35)
    
    if is_greeting_query: pick_if_present({"greeting"}, max_gap=0.22)
    if is_buildings_overview_query: pick_if_present({"buildings"}, max_gap=0.22)
    if is_landmark_query: pick_if_present({"campus_historical_landmarks"}, max_gap=0.22)
    if is_barangan_query: pick_if_present({"cresencio_barangan_history"}, max_gap=0.24)
    # NEW: direct picks for motto/logo and transition process
    if is_motto_query: pick_if_present({"northwestern_fiat_lux_meaning"}, max_gap=0.28)
    if is_logo_query: pick_if_present({"northwestern_logo_symbolism"}, max_gap=0.28)
    if is_transition_process_like: pick_if_present({"transition_process"}, max_gap=0.28)
    if is_albano_query: pick_if_present({"angel_albano_history"}, max_gap=0.26)
    # UPDATED: explicit pick for both variants
    if strong_univ_status: pick_if_present({"major_transitions"}, max_gap=0.3)
    # NEW: explicit pick for founders-establish list phrasing
    if strong_founders_establish_list:
        pick_if_present({"northwestern_academy_incorporators"}, max_gap=0.3)
    # NEW: explicit pick for Academy→College phrasing
    if strong_academy_become_college:
        pick_if_present({"transition_process", "early_years"}, max_gap=0.42)
    # NEW: explicit pick for Nicolas contribution/role
    if is_nicolas_contrib_like: 
        pick_if_present({"nicolas_contribution"}, max_gap=0.3)
    # NEW: explicit pick for early sacrifices
    if is_academy_sacrifices_like:
        pick_if_present({"northwestern_academy_early_sacrifices"}, max_gap=0.3)
    # NEW: explicit pick for nurturing years
    if is_nurturing_years_like:
        pick_if_present({"nurturing_years"}, max_gap=0.3)
    # NEW: explicit pick for engineering program (Now routes to courses)
    if is_engineering_program_like:
         pick_if_present({"northwestern_college_courses"}, max_gap=0.3)
    # NEW: explicit pick for student activists
    if is_activist_details_query:
        pick_if_present({"northwestern_student_activists"}, max_gap=0.3)
    # NEW: explicit pick for commonwealth era
    if is_commonwealth_planning_like:
        pick_if_present({"northwestern_academy_commonwealth_era"}, max_gap=0.3)
    # NEW: explicit pick for generic course query
    if is_generic_course_query:
         pick_if_present({"northwestern_college_courses"}, max_gap=0.35)
    # NEW: explicit pick for great teachers
    if any(t in user_tokens for t in ["teacher", "mentors", "great", "teachers"]):
         pick_if_present({"northwestern_classroom_icons", "northwestern_faculty_mentors"}, max_gap=0.3)


    # Final selection and score adjustments
    best_index = candidates[0][0]
    best_score = candidates[0][1]
    best_meta = _pattern_meta[best_index]
    best_tag = best_meta["tag"]
    responses = best_meta.get("responses", [])
    keywords = set([k.lower() for k in best_meta.get("keywords", [])])

    try:
        orig_intent = next((it for it in _intents.get("intents", []) if it.get("tag") == best_tag), {})
        boost_keywords = set([k.lower() for k in orig_intent.get("boost_keywords", [])])
    except Exception:
        boost_keywords = set()

    keyword_overlap = sum(1 for kw in keywords if kw in user_input.lower())
    boost_overlap = len(set(_tokenizer.tokenize(user_input.lower())).intersection(boost_keywords))
    best_score += min(0.08 * keyword_overlap + 0.12 * boost_overlap, 0.35)

    second_best_score = candidates[1][1] if len(candidates) > 1 else 0.0

    # Collision fixes and penalties (refined)
    # Boost early_years for 1932 questions; only penalize for "when/year founded"
    if best_tag == "early_years" and has_1932:
        best_score += 0.3
    if best_tag == "foundation" and has_1932:
         best_score += 0.3

    if best_tag == "early_years" and is_foundation_when_query:
        best_score = max(-1.0, best_score - 0.35)

    # Presidents vs current/list fixes
    if best_tag == "complete_northwestern_presidents_list" and is_current_president_query and not is_all_presidents_like:
        best_score = max(-1.0, best_score - 0.42)
    if best_tag == "northwestern_current_president" and is_all_presidents_like:
        best_score = max(-1.0, best_score - 0.5)
    if best_tag == "complete_northwestern_presidents_list" and is_generic_president_query:
        best_score = max(-1.0, best_score - 0.42)
    # NEW: Final penalty for Presidents List stealing Founders
    if best_tag == "complete_northwestern_presidents_list" and (is_founders_query or is_founders_list_query):
         best_score = max(-1.0, best_score - 0.7)


    # Nicolas contribution vs incorporators
    if best_tag == "northwestern_academy_incorporators" and is_nicolas_contrib_like:
        best_score = max(-1.0, best_score - 0.36)
    # NEW: Nicolas contribution vs faculty mentors fix
    if best_tag == "nicolas_contribution" and is_nicolas_teacher_like:
        best_score = max(-1.0, best_score - 0.36)
    if best_tag == "northwestern_faculty_mentors" and is_nicolas_contrib_like:
        best_score = max(-1.0, best_score - 0.36)
    
    # Barangan vs founders fix
    if best_tag == "northwestern_academy_incorporators" and is_barangan_query:
        best_score = max(-1.0, best_score - 0.36)
    if best_tag == "cresencio_barangan_history" and is_founders_who_query and not is_barangan_query:
        best_score = max(-1.0, best_score - 0.36)

    # Maximo Caday vs Nicolas Contribution
    if best_tag == "nicolas_contribution" and is_caday_relationship_like:
         best_score = max(-1.0, best_score - 0.4)

    # Landmarks vs general_info
    if best_tag == "general_info" and is_landmark_query:
        best_score = max(-1.0, best_score - 0.24)

    # Engineering/courses pairwise fixes
    # DELETED: if best_tag == "northwestern_college_engineering_program" and is_first_engineering_program_like: best_score += 0.15
    # NEW: courses vs engineering program fix (for the mutual steal)
    if best_tag == "northwestern_college_courses" and is_engineering_program_like:
        # Since engineering now routes here, add a small positive nudge for specific engineering queries
        best_score += 0.1
        
    # UPDATED: extra guards for Academy→College phrasing
    if strong_academy_become_college and best_tag == "early_years":
        best_score += 0.18
    if strong_academy_become_college and best_tag in {"northwestern_college_courses","general_info"}:
        best_score = max(-1.0, best_score - 0.3)
    # NEW: Final hammer for Commonwealth stealing Incorporators
    if (is_founders_query or is_founders_list_query) and best_tag == "northwestern_academy_commonwealth_era":
        best_score = max(-1.0, best_score - 0.7)
    # NEW: Final hammer for Incorporators stealing Commonwealth
    if is_commonwealth_planning_like and best_tag == "northwestern_academy_incorporators":
        best_score = max(-1.0, best_score - 0.7)


    # NEW: Early Sacrifices vs Nurturing Years fix
    if best_tag == "northwestern_academy_early_sacrifices" and is_nurturing_years_like and not is_academy_sacrifices_like:
        best_score = max(-1.0, best_score - 0.3)
    if best_tag == "nurturing_years" and is_academy_sacrifices_like and not is_nurturing_years_like:
        best_score = max(-1.0, best_score - 0.3)

    # NEW: Final hammer to fix early_years/nurturing_years confusion
    if best_tag == "early_years" and is_nurturing_years_like:
        best_score = max(-1.0, best_score - 0.4)
    if best_tag == "nurturing_years" and has_1932:
        best_score = max(-1.0, best_score - 0.4)
        
    # Martial Law vs Activists fix
    if best_tag == "northwestern_martial_law" and is_activist_details_query:
        best_score = max(-1.0, best_score - 0.4)

    # Generic Courses vs Accreditation/Engineering
    if is_generic_course_query:
        if best_tag in {"Accreditation", "Deregulated_Status", "major_transitions"}:
            best_score = max(-1.0, best_score - 0.4)
        # DELETED: if best_tag == "northwestern_college_engineering_program": best_score = max(-1.0, best_score - 0.3)
             
    # General Info Hammer
    if best_tag == "general_info" and any(t in user_tokens for t in ["presidents", "founders", "nicolas", "courses", "engineering"]):
        best_score = max(-1.0, best_score - 0.5)


    # NEW: slightly lower ambiguity threshold when who/when detectors fire
    ambiguous_threshold = 0.02
    if is_who_query or is_when_query or leadership_during_college_transition or strong_academy_become_college:
        ambiguous_threshold = 0.015

    # NEW: Dynamic confidence threshold (ADJUSTED)
    token_count = len(user_tokens)
    if token_count <= 3:
        CONFIDENCE_THRESHOLD = 0.5
    elif token_count <= 8:
        CONFIDENCE_THRESHOLD = 0.61  # Increased from 0.59
    else:
        CONFIDENCE_THRESHOLD = 0.65  # Increased from 0.63

    # Build debug info before fallback
    debug_info = {
        "best_tag": best_tag,
        "best_score": round(best_score, 3),
        "second_best_score": round(second_best_score, 3),
        "threshold": CONFIDENCE_THRESHOLD,
        "original_example": best_meta.get("original_example"),
        "responses": responses
    }

    # 4) Confidence fallback (strong greeting/general-info routing)
    if best_score < CONFIDENCE_THRESHOLD:
        # Prefer hard detector fallback first for Academy→College
        if strong_academy_become_college:
            # Prefer transition_process if possible, fallback to early_years
            fallback_intent = next((i for i in _intents.get("intents", []) if i.get("tag")=="transition_process"), None)
            fallback_tag = "transition_process"
            if not fallback_intent:
                fallback_intent = next((i for i in _intents.get("intents", []) if i.get("tag")=="early_years"), None)
                fallback_tag = "early_years"

            if fallback_intent:
                resp = random.choice(fallback_intent.get("responses", []))
                debug_info["reason"] = "Strong detector fallback (Academy → College date)."
                debug_info["best_tag"] = fallback_tag
                if not eval_mode:
                    st.session_state['last_intent'] = fallback_tag
                return resp, debug_info
        fallback_resp, fallback_tag = keyword_fallback(user_input, _intents)
        if not fallback_resp:
            if is_greeting_query:
                greet_intent = next((i for i in _intents.get("intents", []) if i.get("tag")=="greeting"), None)
                if greet_intent:
                    fallback_resp = random.choice(greet_intent.get("responses", []))
                    fallback_tag = "greeting"
            elif is_general_info_query:
                gi_intent = next((i for i in _intents.get("intents", []) if i.get("tag")=="general_info"), None)
                if gi_intent:
                    fallback_resp = random.choice(gi_intent.get("responses", []))
                    fallback_tag = "general_info"
            # NEW: Route short award/ranking queries if unmatched by semantic
            elif any(t in user_tokens for t in ["award", "2004", "ranking"]) and len(user_tokens) < 5:
                award_intent = next((i for i in _intents.get("intents", []) if i.get("tag")=="2004_Award"), None)
                if award_intent:
                    fallback_resp = random.choice(award_intent.get("responses", []))
                    fallback_tag = "2004_Award"
            # NEW: Block short general fallback for known high-conflict terms
            elif len(user_tokens) <= 3 and any(t in user_tokens for t in ["presidents", "founders", "nicolas", "college"]):
                 # If it failed semantic and contains high-value nouns, let it hit "I don't know" or the ambiguity route.
                 pass
                     
        if fallback_resp:
            debug_info["reason"] = "Detector-driven fallback triggered."
            debug_info["best_tag"] = fallback_tag
            if not eval_mode:
                st.session_state['last_intent'] = fallback_tag
            return fallback_resp, debug_info
        debug_info["reason"] = "Low semantic score, no fallback."
        debug_info["best_tag"] = None
        return "I don't know.", debug_info

    # 5) Conflict-aware ambiguity handling
    is_conflict_pair = False
    if len(candidates) > 1:
        second_tag = _pattern_meta[candidates[1][0]]["tag"]
        for pair in CONFLICT_INTENT_SETS:
            if best_tag in pair and second_tag in pair:
                is_conflict_pair = True
                break

    # UPDATED: accept close-call for Academy→College phrasing
    if strong_academy_become_college and best_tag in ["early_years", "transition_process"]:
        debug_info["reason"] = "Strong detector match (Academy → College date)."
        if not eval_mode:
            st.session_state['last_intent'] = best_tag
        return random.choice(responses) if responses else "I don't know.", debug_info

    # Accept close-call for leadership during college transition (routes to complete list)
    if (college_leadership_focus or leadership_during_college_transition) and best_tag == "complete_northwestern_presidents_list":
        debug_info["reason"] = "Strong detector match (college leadership list)."
        if not eval_mode:
            st.session_state['last_intent'] = best_tag
        return random.choice(responses) if responses else "I don't know.", debug_info

    # NEW: accept close-call for who/when if preferred tag wins
    if is_who_query and best_tag in {"complete_northwestern_presidents_list","northwestern_current_president","northwestern_academy_incorporators", "cresencio_barangan_history", "northwestern_student_activists", "northwestern_faculty_mentors", "northwestern_classroom_icons"}:
        debug_info["reason"] = "Who-question preference accepted."
        if not eval_mode:
            st.session_state['last_intent'] = best_tag
        return random.choice(responses) if responses else "I don't know.", debug_info
    if is_when_query and best_tag in {"major_transitions","early_years","foundation", "transition_process", "northwestern_academy_commonwealth_era"}:
        debug_info["reason"] = "When-question preference accepted."
        if not eval_mode:
            st.session_state['last_intent'] = best_tag
        return random.choice(responses) if responses else "I don't know.", debug_info

    if best_score - second_best_score < ambiguous_threshold:
        if not is_conflict_pair and best_score > 0.7:
            debug_info["reason"] = "High confidence match (close pair accepted)."
            return random.choice(responses) if responses else "I don't know.", debug_info
        debug_info["reason"] = "Ambiguous match."
        return "I see a couple of possible answers. Can you be more specific?", debug_info

    best_response = random.choice(responses) if responses else "I don't know."
    debug_info["reason"] = "High confidence match."
    debug_info["best_tag"] = best_tag
    if not eval_mode:
        st.session_state['last_intent'] = best_tag
    return best_response, debug_info
