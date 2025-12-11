def compute_detectors(user_tokens: set) -> dict:
    # --- 1. CORE TOKEN & SIMPLE INPUT DETECTORS (Defined first) ---
    is_who_query = "who" in user_tokens
    is_when_query = any(t in user_tokens for t in {"when", "year", "date"})
    mentions_president = "president" in user_tokens
    has_1932 = "1932" in user_tokens
    
    # Utility/Greeting
    early_greet = (any(t in user_tokens for t in {"hi", "hello", "hey", "greetings"}) or ({"how", "are", "you"} <= user_tokens) or ("whats" in user_tokens) or (("what" in user_tokens) and ("up" in user_tokens)))
    is_greeting_like = early_greet or any(t in user_tokens for t in {"hi", "hello", "hey", "greetings", "good", "day", "help"})
    is_general_info_like = (("what" in user_tokens and "university" in user_tokens) or ("tell" in user_tokens and "university" in user_tokens) or ("northwestern" in user_tokens and "university" in user_tokens))
    
    # Founder/Role Input Flags
    founder_query_tokens = {"founder", "founders", "founded", "found", "cofounder", "co-founder", "incorporator"}
    is_founder_query = any(t in user_tokens for t in founder_query_tokens)
    is_president_like = mentions_president or any(t in user_tokens for t in {"led", "leader", "head"})
    is_current_time_like = any(t in user_tokens for t in {"current", "present", "incumbent", "sitting", "now", "today", "right"})
    
    # Status/Date Input Flags
    status_core = {"become", "became", "status", "recognized", "recognition", "confirmation", "confirm", "year"}
    is_university_status_query = ("university" in user_tokens) and any(t in user_tokens for t in status_core)
    is_status_like = is_university_status_query
    
    # --- 2. INTERMEDIATE & COMPLEX DETECTORS ---
    
    # Location/Topic Specifics
    is_barangan_like = any(t in user_tokens for t in {"barangan", "cresencio", "cashier", "funds", "finance"})
    is_landmark_like = any(t in user_tokens for t in {"landmark", "landmarks", "historical", "historic", "site", "sites"})
    is_new_site_like = any(t in user_tokens for t in {"site", "campus", "airport", "avenue", "hectare", "new"}) and ("site" in user_tokens or "campus" in user_tokens)
    is_nicolas_title_like = any(t in user_tokens for t in {"mr", "mr.", "referred", "called", "known"}) # Variable causing error
    is_albano_query = any(t in user_tokens for t in {"angel", "albano"})
    is_access_poor_like = any(t in user_tokens for t in {"poor", "needy", "scholarship", "scholarships", "help", "assist", "students"})
    is_first_engineering_program_like = ("first" in user_tokens and "engineering" in user_tokens and "program" in user_tokens)
    is_flagship_like = any(t in user_tokens for t in {"flagship", "1960s", "1960", "sixties"})
    
    # Institution Mention Scopes
    mentions_institution_for_president = any(t in user_tokens for t in {"university", "northwestern", "nwu", "college"})
    
    # --- 3. COMPOSITE DETECTORS (Using intermediate flags) ---
    
    # Leadership Composites
    is_current_president_query = (mentions_president and is_current_time_like and mentions_institution_for_president) or (is_current_time_like and any(t in user_tokens for t in {"head", "leader"}) and mentions_institution_for_president)
    is_presidents_query = mentions_president and mentions_institution_for_president and not is_current_president_query
    is_list_query = any(t in user_tokens for t in {"past", "present", "all", "list", "history"}) and mentions_president
    is_generic_leadership_phrase = is_president_like and any(t in user_tokens for t in {"university", "nwu", "northwestern", "college"})
    
    is_college_president_query = (mentions_president and ("college" in user_tokens) and ("university" not in user_tokens))
    strong_first_college_president = (is_college_president_query and ("first" in user_tokens))
    
    # Founding Composites
    is_founders_list_query = any(t in user_tokens for t in {"list", "name", "incorporators", "co-founders"}) and is_founder_query
    strong_founders_establish_list = (is_founders_list_query and any(t in user_tokens for t in {"helped", "establish", "established"}))
    is_foundation_when_query = any(t in user_tokens for t in {"founded", "foundation"}) and is_when_query
    is_founders_who_query = (is_founder_query and "who" in user_tokens)

    # Status/Transition Composites
    strong_univ_status = (is_university_status_query and is_when_query)
    strong_academy_become_college = (("academy" in user_tokens) and ("college" in user_tokens) and is_when_query)
    leadership_during_college_transition = (is_president_like and ("college" in user_tokens) and is_when_query)
    college_leadership_focus = (("college" in user_tokens) and is_president_like)

    # Specific Topics
    is_nicolas_contrib_like = (is_founder_query and any(t in user_tokens for t in {"what", "did", "do"}) and any(t in user_tokens for t in {"nicolas"}))
    
    # --- 4. ASSEMBLE AND RETURN DICTIONARY ---
    return {
        # Core
        "is_who_query": is_who_query,
        "is_when_query": is_when_query,
        "has_1932": has_1932,
        "early_greet": early_greet,
        "is_greeting_like": is_greeting_like,
        
        # Leadership
        "is_president_like": is_president_like,
        "is_current_time_like": is_current_time_like,
        "is_current_president_query": is_current_president_query,
        "is_presidents_query": is_presidents_query,
        "is_list_query": is_list_query,
        "is_generic_leadership_phrase": is_generic_leadership_phrase,
        "strong_first_college_president": strong_first_college_president,
        "leadership_during_college_transition": leadership_during_college_transition,
        "is_college_president_query": is_college_president_query,
        "is_first_college_president_query": strong_first_college_president,
        
        # Founding
        "is_founder_query": is_founder_query,
        "is_founders_like": is_founder_query,
        "is_founders_list_query": is_founders_list_query,
        "strong_founders_establish_list": strong_founders_establish_list,
        "is_foundation_when_query": is_foundation_when_query,
        "is_founders_who_query": is_founders_who_query,

        # Status & Transition
        "is_status_like": is_status_like,
        "strong_univ_status": strong_univ_status,
        "strong_academy_become_college": strong_academy_become_college,
        "is_transition_process_like": any(t in user_tokens for t in {"convert", "transition", "process", "apply", "recognition"}),
        
        # Specifics & Topics
        "is_barangan_like": is_barangan_like,
        "is_barangan_query": is_barangan_like,
        "is_nicolas_title_like": is_nicolas_title_like, # ERROR FIX LOCATION
        "is_nicolas_contrib_like": is_nicolas_contrib_like,
        "is_albano_query": is_albano_query,
        "is_landmark_like": is_landmark_like,
        "is_general_info_like": is_general_info_like,
        "is_access_poor_like": is_access_poor_like,
        "is_first_engineering_program_like": is_first_engineering_program_like,
        "is_flagship_like": is_flagship_like,
        "college_leadership_focus": college_leadership_focus,
        
        # Structural/Building (Derived for matching clarity)
        "is_motto_query": D['is_motto_query'] if 'D' in locals() else is_motto_query, # Use intermediate flag
        "is_logo_query": D['is_logo_query'] if 'D' in locals() else any(t in user_tokens for t in {"logo", "owl", "seal"}),
        "is_buildings_overview_query": is_buildings_like if 'is_buildings_like' in locals() else any(t in user_tokens for t in {"buildings", "campus"}),
        "is_student_center_query": any(t in user_tokens for t in {"student", "center", "aquino"}),
        "is_worship_center_query": all(t in user_tokens for t in {"worship", "center"}),
        "is_ban_building_query": any(t in user_tokens for t in {"ban", "ben", "nicolas"}),
        "is_new_site_query": is_new_site_like,
        "is_landmark_query": is_landmark_like,
        
    }
