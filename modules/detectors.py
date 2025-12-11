def compute_detectors(user_tokens: set):
    # --- Basic Token Check for Institutional Mention ---
    mentions_university = "university" in user_tokens
    mentions_president = "president" in user_tokens
    mentions_institution_for_founders = any(t in user_tokens for t in {"university","northwestern","nwu","academy"})
    mentions_institution_for_president = any(t in user_tokens for t in {"university","northwestern","nwu","college"})
    founder_query_tokens = {"founder","founders","founded","find","cofounder","co-founder","incorporator"}
    is_founder_query_base = any(t in user_tokens for t in founder_query_tokens)
    
    # --- Core Query Type Detectors ---
    is_who_query = "who" in user_tokens
    is_when_query = any(t in user_tokens for t in {"when","year","date"})
    is_what_query = "what" in user_tokens
    has_1932 = "1932" in user_tokens

    # --- Time and Leadership Detectors ---
    is_current_time_like = any(t in user_tokens for t in {"current","present","incumbent","sitting","now","today","right"})
    is_president_like = mentions_president or any(t in user_tokens for t in {"led","leader","head"})

    # --- Status and Transition Detectors ---
    status_core = {"become","became","status","recognized","recognition","confirmation","confirm","year"}
    is_university_status_query = mentions_university and any(t in user_tokens for t in status_core)
    
    strong_univ_status = (
        (mentions_university) and
        ("year" in user_tokens or "when" in user_tokens) and
        any(t in user_tokens for t in {"become","became","recognized","recognition","status","confirmation","confirm"})
    )
    process_terms = {"convert","conversion","transition","process","steps","petition","apply","application","recognition","recognized","approval","approved","sec","decs","ched"}
    is_transition_process_like = any(t in user_tokens for t in process_terms) and any(t in user_tokens for t in {"college","university","northwestern","nwu", "academy", "how"})
    strong_academy_become_college = (
        ("academy" in user_tokens) and
        ("college" in user_tokens) and
        any(t in user_tokens for t in {"become","became","becoming","transition","convert","conversion","when","year","date"})
    )
    is_commonwealth_planning_like = any(t in user_tokens for t in {"planning","expand","expansion","programs","courses","why", "commonwealth", "1935", "constitution", "surge", "affect"})

    # --- Founders and Barangan Detectors ---
    is_founders_like = is_founder_query_base
    is_founders_query = is_founder_query_base and mentions_institution_for_founders
    is_founders_who_query = is_founder_query_base and is_who_query
    founders_list_terms = {"list", "name", "founders", "incorporators", "co-founders", "cofounders", "ten"}
    is_founders_list_query = any(t in user_tokens for t in founders_list_terms) and is_founders_query
    strong_founders_establish_list = (
        any(t in user_tokens for t in {"name","everyone","all","list"}) and
        any(t in user_tokens for t in {"helped","establish","established","incorporators","founders"}) and
        any(t in user_tokens for t in {"academy","northwestern","nwu"})
    )
    is_barangan_like = any(t in user_tokens for t in {"barangan","cresencio","cashier","funds","finance"})
    is_albano_query = any(t in user_tokens for t in {"angel","albano"})

    # --- President Detectors (Routing to Current/List) ---
    is_current_president_query = is_president_like and is_current_time_like
    is_all_presidents_like = any(t in user_tokens for t in {"all", "list", "past", "first", "college"}) and is_president_like
    is_generic_president_query = is_president_like and is_who_query and not is_current_time_like and not is_all_presidents_like and not any(t in user_tokens for t in {"first", "past", "list", "all", "of", "college"})
    is_presidents_query = is_president_like and mentions_institution_for_president and not is_current_president_query
    
    # --- Nicolas and Faculty Detectors ---
    is_nicolas_contrib_like = ("nicolas" in user_tokens) and any(t in user_tokens for t in {"contribution", "contributions", "do", "did", "help", "impact", "expansion", "role"})
    is_nicolas_teacher_like = ("nicolas" in user_tokens) and any(t in user_tokens for t in {"teacher", "instructor", "faculty", "known"})
    is_nicolas_who_in_college = (("nicolas" in user_tokens) and is_who_query and ("college" in user_tokens))
    is_caday_relationship_like = any(t in user_tokens for t in ["maximo", "caday", "relationship", "colleagues", "get along"])
    
    # --- Early Years and Academy Detectors ---
    is_academy_sacrifices_like = any(t in user_tokens for t in {"sacrifices", "goal", "vision"}) and ("academy" in user_tokens or is_founders_like)
    is_nurturing_years_like = (any(t in user_tokens for t in {"early","beginnings","like","where","held","challenges","face"}) and (mentions_institution_for_founders)) and not is_academy_sacrifices_like
    
    # --- Program and Logo Detectors ---
    is_engineering_program_like = "engineering" in user_tokens and any(t in user_tokens for t in {"program", "courses", "dean", "flagship", "engineering"})
    is_generic_course_query = any(t in user_tokens for t in {"courses", "programs", "degree", "associate"}) and (mentions_institution_for_president) and not is_engineering_program_like
    is_motto_query = ("fiat" in user_tokens and "lux" in user_tokens) or ("motto" in user_tokens) or ("let" in user_tokens and "light" in user_tokens)
    is_logo_query = any(t in user_tokens for t in {"logo", "symbol", "seal", "emblem", "mascot", "owl"})
    
    # --- Campus and General Detectors ---
    is_landmark_like = any(t in user_tokens for t in {"landmark","landmarks","historical","historic","site","sites"})
    is_general_info_like = (is_what_query and mentions_university) or ("tell" in user_tokens and mentions_university) or (mentions_university and "northwestern" in user_tokens)
    early_greet = any(t in user_tokens for t in {"hi","hello","hey","greetings"}) or ({"how","are","you"} <= user_tokens) or ("whats" in user_tokens) or (("what" in user_tokens) and ("up" in user_tokens))
    is_greeting_query = early_greet
    
    # --- Activism Detectors ---
    is_activist_details_query = any(t in user_tokens for t in {"velasco", "pascual", "became", "support", "notable", "leaders", "marcos"})

    # --- Compile all detectors into the dictionary ---
    detectors = {
        "is_who_query": is_who_query,
        "is_when_query": is_when_query,
        "is_what_query": is_what_query,
        "has_1932": has_1932,
        "is_current_time_like": is_current_time_like,
        "is_president_like": is_president_like,
        "is_university_status_query": is_university_status_query,
        "strong_univ_status": strong_univ_status,
        "is_transition_process_like": is_transition_process_like,
        "strong_academy_become_college": strong_academy_become_college,
        "is_commonwealth_planning_like": is_commonwealth_planning_like,
        "is_founders_like": is_founders_like,
        "is_founders_query": is_founders_query,
        "is_founders_who_query": is_founders_who_query,
        "is_founders_list_query": is_founders_list_query,
        "strong_founders_establish_list": strong_founders_establish_list,
        "is_barangan_like": is_barangan_like,
        "is_albano_query": is_albano_query,
        "is_current_president_query": is_current_president_query,
        "is_all_presidents_like": is_all_presidents_like,
        "is_generic_president_query": is_generic_president_query,
        "is_presidents_query": is_presidents_query,
        "is_nicolas_contrib_like": is_nicolas_contrib_like,
        "is_nicolas_teacher_like": is_nicolas_teacher_like,
        "is_nicolas_who_in_college": is_nicolas_who_in_college,
        "is_caday_relationship_like": is_caday_relationship_like,
        "is_academy_sacrifices_like": is_academy_sacrifices_like,
        "is_nurturing_years_like": is_nurturing_years_like,
        "is_engineering_program_like": is_engineering_program_like,
        "is_generic_course_query": is_generic_course_query,
        "is_motto_query": is_motto_query,
        "is_logo_query": is_logo_query,
        "is_landmark_like": is_landmark_like,
        "is_general_info_like": is_general_info_like,
        "is_greeting_query": is_greeting_query,
        "is_activist_details_query": is_activist_details_query,
    }
    
    # Exclude intermediate variables used above but not required downstream (e.g., mentions_university, early_greet)
    # Filter the dictionary to only include boolean values and relevant names.
    return {k: v for k, v in detectors.items() if isinstance(v, bool)}
