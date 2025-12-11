def compute_detectors(user_tokens: set) -> dict:
    # --- Basic Question Type Detectors ---
    is_who_query = "who" in user_tokens
    is_when_query = any(t in user_tokens for t in {"when", "year", "date"})

    # --- Early Greeting/Utility Detectors (for early exit logic) ---
    early_greet = (any(t in user_tokens for t in {"hi", "hello", "hey", "greetings"})
        or ({"how", "are", "you"} <= user_tokens)
        or ("whats" in user_tokens) or (("what" in user_tokens) and ("up" in user_tokens)))

    # --- Institution & Status Detectors ---
    mentions_university = "university" in user_tokens
    mentions_president = "president" in user_tokens
    mentions_institution_for_founders = any(t in user_tokens for t in {"university", "northwestern", "nwu", "academy"})
    mentions_institution_for_president = any(t in user_tokens for t in {"university", "northwestern", "nwu", "college"})
    
    status_core = {"become", "became", "status", "recognized", "recognition", "confirmation", "confirm", "year"}
    is_university_status_query = ("university" in user_tokens) and any(t in user_tokens for t in status_core)
    is_status_like = is_university_status_query

    # --- Founding/Leadership Detectors ---
    founder_query_tokens = {"founder", "founders", "founded", "found", "cofounder", "co-founder", "incorporator"}
    is_founder_query = any(t in user_tokens for t in founder_query_tokens)
    is_founders_list_query = any(t in user_tokens for t in {"list", "name", "founders", "incorporators", "co-founders", "cofounders"}) and is_founder_query
    
    is_president_like = mentions_president or any(t in user_tokens for t in {"led", "leader", "head"})
    is_current_time_like = any(t in user_tokens for t in {"current", "present", "incumbent", "sitting", "now", "today", "right"})
    
    is_current_president_query = (mentions_president and is_current_time_like and mentions_institution_for_president) or (is_current_time_like and any(t in user_tokens for t in {"head", "leader"}) and any(t in user_tokens for t in {"university", "northwestern", "nwu"}))
    is_presidents_query = mentions_president and mentions_institution_for_president and not is_current_president_query
    
    is_list_query = any(t in user_tokens for t in {"past", "present", "all", "list", "history"}) and ("president" in user_tokens)


    # --- Specific Historical Detectors ---
    is_first_president_query = (("first" in user_tokens and "president" in user_tokens) or ("founding" in user_tokens and "president" in user_tokens))
    generic_leader_terms = {"led", "leader", "head"}
    is_generic_leadership_phrase = any(t in user_tokens for t in generic_leader_terms) and any(t in user_tokens for t in {"university", "nwu", "northwestern", "college"})
    
    is_nicolas_title_like = any(t in user_tokens for t in {"mr", "mr.", "referred", "called", "known"})
    
    # Strong/Complex Detectors
    strong_univ_status = (("university" in user_tokens) and any(t in user_tokens for t in {"year", "when"}) and any(t in user_tokens for t in status_core))
    strong_academy_become_college = (("academy" in user_tokens) and ("college" in user_tokens) and any(t in user_tokens for t in {"become", "became", "transition", "when", "year", "date"}))
    strong_first_college_president = (("president" in user_tokens) and ("college" in user_tokens) and ("first" in user_tokens) and any(t in user_tokens for t in {"northwestern", "nwu"}))
    leadership_during_college_transition = (any(t in user_tokens for t in {"led", "leader", "head", "who"}) and ("college" in user_tokens) and any(t in user_tokens for t in {"became", "become", "transition", "when", "year", "date"}))

    # --- Assemble Detectors ---
    # NOTE: This dictionary must include ALL flags used in modules/matcher.py
    return {
        "is_who_query": is_who_query,
        "is_when_query": is_when_query,
        "early_greet": early_greet,
        "is_founder_query": is_founder_query,
        "is_president_like": is_president_like,
        "is_current_time_like": is_current_time_like,
        "is_current_president_query": is_current_president_query,
        "is_presidents_query": is_presidents_query,
        "is_list_query": is_list_query,
        "is_first_president_query": is_first_president_query,
        "is_generic_leadership_phrase": is_generic_leadership_phrase,
        "is_nicolas_title_like": is_nicolas_title_like,
        "strong_univ_status": strong_univ_status,
        "strong_academy_become_college": strong_academy_become_college,
        "strong_first_college_president": strong_first_college_president,
        "leadership_during_college_transition": leadership_during_college_transition,

        # Also include other complex or downstream flags if they were used (many of these were implicitly defined)
        "is_status_like": is_status_like,
        "is_new_site_like": any(t in user_tokens for t in {"site","campus","airport","avenue","hectare","new"}) and ("site" in user_tokens or "campus" in user_tokens),
        "is_founders_like": any(t in user_tokens for t in {"founder","founders","incorporators","cofounders","co-founders"}),
        "is_greeting_like": early_greet or any(t in user_tokens for t in {"hi","hello","hey","greetings","good","day","help"}),
        "is_general_info_like": (("what" in user_tokens and "university" in user_tokens) or ("tell" in user_tokens and "university" in user_tokens) or ("northwestern" in user_tokens and "university" in user_tokens)),
        "is_buildings_like": any(t in user_tokens for t in {"buildings","structures","timeline","completed","major","campus"}) and not any(t in user_tokens for t in {"student","worship","aquino","sc"}),
        "is_barangan_like": any(t in user_tokens for t in {"barangan","cresencio","cashier","funds","finance"}),
        "is_landmark_like": any(t in user_tokens for t in {"landmark","landmarks","historical","historic","site","sites"}),
        "college_leadership_focus": (("college" in user_tokens) and any(t in user_tokens for t in {"who","led","leader","head"})),
        "is_albano_query": any(t in user_tokens for t in {"angel","albano"}),
        "is_transition_process_like": any(t in user_tokens for t in {"convert","conversion","transition","process","steps","petition","apply","application","recognition","recognized","approval","approved","sec","decs","ched"}) and any(t in user_tokens for t in {"college","university","northwestern","nwu"}),
        "is_ban_building_query": (len(user_tokens.intersection({"ban", "ben", "nicolas", "ben", "ben a", "ben a. nicolas"})) > 0) and (len(user_tokens.intersection({"building", "inaugurated", "inauguration", "opened", "completion", "when"})) > 0),
        "is_student_center_query": (("when" in user_tokens) or ("completed" in user_tokens) or ("finished" in user_tokens)) and any(t in user_tokens for t in {"student", "center", "aquino", "multipurpose", "sc"}),
        "is_worship_center_query": (("when" in user_tokens) or ("finished" in user_tokens) or ("completion" in user_tokens) or ("date" in user_tokens)) and all(t in user_tokens for t in {"worship", "center"}),
        "is_motto_query": ("fiat" in user_tokens and "lux" in user_tokens) or ("motto" in user_tokens) or ("let" in user_tokens and "light" in user_tokens),
        "is_logo_query": any(t in user_tokens for t in {"logo", "symbol", "seal", "emblem", "mascot", "owl"}),
        "is_buildings_overview_query": any(t in user_tokens for t in {"buildings", "structures", "timeline", "completed", "major", "campus"}) and not (("student" in user_tokens) or ("worship" in user_tokens) or ("aquino" in user_tokens) or ("sc" in user_tokens)),
        "is_new_site_query": (("site" in user_tokens or "campus" in user_tokens) and any(t in user_tokens for t in {"airport", "avenue", "hectare", "new"})),
        "is_landmark_query": any(t in user_tokens for t in {"landmark","landmarks","historical","historic","site","sites"}),
        "is_barangan_query": is_barangan_like,
        "is_foundation_when_query": any(t in user_tokens for t in {"founded","foundation"}) and any(t in user_tokens for t in {"when","year","date"}),
        "is_founders_who_query": (any(t in user_tokens for t in {"founder","founders","incorporators"}) and "who" in user_tokens),
        "is_college_president_query": ("president" in user_tokens) and ("college" in user_tokens) and ("university" not in user_tokens),
        "is_first_college_president_query": ("president" in user_tokens) and ("college" in user_tokens) and ("first" in user_tokens),
        "is_nicolas_contrib_like": ("founder" in user_tokens and "what" in user_tokens and (("did" in user_tokens) or ("do" in user_tokens)) and any(t in user_tokens for t in {"northwestern","nwu","university"})),
        "has_1932": "1932" in user_tokens,
        "strong_founders_establish_list": (any(t in user_tokens for t in {"name", "everyone", "all", "list"}) and any(t in user_tokens for t in {"helped", "establish", "established", "incorporators", "founders"}) and any(t in user_tokens for t in {"academy", "northwestern", "nwu"}))
    }
