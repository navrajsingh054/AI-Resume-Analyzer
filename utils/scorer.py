# utils/scorer.py
# ─────────────────────────────────────────────────────────────────────
# PURPOSE : Calculate ATS match score between a resume and job description
#           using TF-IDF vectorization and cosine similarity.
#
# ALGORITHM SUMMARY:
#   1. Clean both texts using cleaner.py (Day 3)
#   2. Fit a TF-IDF vectorizer on both documents together
#   3. Transform each document into a TF-IDF vector
#   4. Compute cosine similarity between the two vectors
#   5. Multiply by 100 → ATS score (0 to 100)
#
# ALSO PRODUCES:
#   - Keyword overlap analysis
#   - Section-level sub-scores (skills, experience, education)
#   - Score breakdown explaining what drove the score
#
# USED BY : app.py (Day 8)
# ─────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

# sklearn = scikit-learn, the standard Python ML library
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Our own modules
from utils.cleaner import clean_text_as_string, clean_text
from utils.keyword_extractor import extract_skills_from_text


# ── Vectorizer configuration ───────────────────────────────────────────
# We define these settings once at module level.
# ngram_range=(1,2) means include both single words AND two-word phrases.
# This is critical: "machine learning" should be one feature, not two.
#
# min_df=1 means include a word even if it appears in only 1 document.
# For only 2 documents (resume + JD), min_df=1 is the correct setting.
# If min_df=2, a word must appear in BOTH docs — too strict for our use case.
#
# sublinear_tf=True applies log normalization to term frequency:
#   TF = 1 + log(raw_count) instead of raw_count
# This prevents a word appearing 10 times from being 10x more important
# than a word appearing once. Diminishing returns is more realistic.
VECTORIZER_CONFIG = {
    "ngram_range"  : (1, 2),   # unigrams + bigrams
    "min_df"       : 1,        # include rare words (important for 2-doc corpus)
    "sublinear_tf" : True,     # log-normalize term frequencies
    "strip_accents": "unicode",# handle accented characters (é → e)
    "analyzer"     : "word",   # analyze at word level (not character level)
}


# ════════════════════════════════════════════════════════════════════
# SECTION 1 : CORE TF-IDF SCORING
# ════════════════════════════════════════════════════════════════════

def compute_tfidf_score(resume_text: str, job_text: str) -> float:
    """
    Core function: compute cosine similarity between resume and job description.

    Step-by-step:
        1. Clean both texts → clean strings
        2. Create TF-IDF vectorizer
        3. fit_transform on [resume, job_desc] → 2×N matrix
           (2 rows = 2 documents, N columns = unique words in vocabulary)
        4. Extract row 0 (resume vector) and row 1 (job vector)
        5. Compute cosine similarity between them
        6. Return as float 0.0 → 1.0

    Args:
        resume_text : raw text from resume (extractor.py output)
        job_text    : raw text from job description

    Returns:
        Float from 0.0 to 1.0 (multiply by 100 for percentage)
    """
    # ── Step 1: Clean both texts ───────────────────────────────────
    # clean_text_as_string returns a space-joined string of lemmatized tokens
    # e.g. "python flask docker machine learn api build"
    clean_resume = clean_text_as_string(resume_text)
    clean_job    = clean_text_as_string(job_text)

    # Guard: if either text is empty after cleaning, score is 0
    if not clean_resume.strip() or not clean_job.strip():
        return 0.0

    # ── Step 2: Create vectorizer ──────────────────────────────────
    vectorizer = TfidfVectorizer(**VECTORIZER_CONFIG)
    # **VECTORIZER_CONFIG unpacks the dict as keyword arguments
    # Equivalent to: TfidfVectorizer(ngram_range=(1,2), min_df=1, ...)

    # ── Step 3: Fit and transform ──────────────────────────────────
    # fit_transform does TWO things in one call:
    #   fit      → learns the vocabulary from both documents
    #              vocabulary = every unique word across resume + job desc
    #   transform → converts each document into a TF-IDF vector
    #
    # We pass a LIST of both documents so the vectorizer sees the full
    # vocabulary. If we fit on resume alone, job description words
    # not in resume would be ignored → wrong scores.
    #
    # tfidf_matrix shape: (2, V) where V = vocabulary size
    #   row 0 = resume vector    [0.0, 0.3, 0.0, 0.7, ...]
    #   row 1 = job desc vector  [0.2, 0.3, 0.5, 0.0, ...]
    tfidf_matrix = vectorizer.fit_transform([clean_resume, clean_job])

    # ── Step 4: Extract individual vectors ────────────────────────
    # tfidf_matrix[0] = resume vector (still a sparse matrix row)
    # We keep them as sparse matrices — cosine_similarity handles them
    resume_vector  = tfidf_matrix[0]   # shape: (1, V)
    job_vector     = tfidf_matrix[1]   # shape: (1, V)

    # ── Step 5: Cosine similarity ──────────────────────────────────
    # cosine_similarity returns a 2D array: [[score]]
    # We extract the single float with [0][0]
    similarity_matrix = cosine_similarity(resume_vector, job_vector)
    score = similarity_matrix[0][0]

    # ── Step 6: Clamp to [0, 1] ────────────────────────────────────
    # Due to floating point arithmetic, score might be 1.0000000002
    # np.clip ensures it stays within bounds
    score = float(np.clip(score, 0.0, 1.0))

    return score


# ════════════════════════════════════════════════════════════════════
# SECTION 2 : KEYWORD OVERLAP ANALYSIS
# ════════════════════════════════════════════════════════════════════

def get_keyword_overlap(
    resume_text : str,
    job_text    : str
) -> Dict[str, any]:
    """
    Find which specific words from the job description appear in the resume.

    This supplements the TF-IDF score with explainability — telling the
    user WHICH keywords matched and which are missing.

    Approach:
        - Tokenize both texts into sets of unique words
        - Compute intersection (matched) and difference (missing)
        - Identify HIGH-VALUE keywords using TF-IDF weights

    Args:
        resume_text : raw resume text
        job_text    : raw job description text

    Returns:
        Dict with:
            matched_keywords : list of words present in both
            missing_keywords : list of words in JD but not resume
            match_rate       : percentage of JD keywords found in resume
            top_missing      : top 10 missing keywords by TF-IDF weight
    """
    # Get clean token sets (unique words only, hence set())
    resume_tokens = set(clean_text(resume_text))
    job_tokens    = set(clean_text(job_text))

    matched = resume_tokens & job_tokens    # intersection
    missing = job_tokens - resume_tokens    # in JD but not resume

    # Calculate match rate
    if len(job_tokens) == 0:
        match_rate = 0.0
    else:
        match_rate = round((len(matched) / len(job_tokens)) * 100, 1)

    # ── Find top missing keywords by TF-IDF weight ─────────────────
    # We want to rank the missing keywords by importance.
    # Strategy: run TF-IDF on just the job description,
    # then rank missing words by their TF-IDF score in the JD.
    top_missing = _rank_keywords_by_tfidf(
        keywords = list(missing),
        reference_text = clean_text_as_string(job_text)
    )

    return {
        "matched_keywords" : sorted(list(matched)),
        "missing_keywords" : sorted(list(missing)),
        "match_rate"       : match_rate,
        "top_missing"      : top_missing[:10],   # top 10 most important missing
    }


def _rank_keywords_by_tfidf(
    keywords       : List[str],
    reference_text : str
) -> List[Tuple[str, float]]:
    """
    Private helper: rank a list of keywords by their TF-IDF score
    within a reference text.

    Used to find the MOST IMPORTANT missing keywords — so we tell the
    user "add kubernetes and terraform" not "add 'the' and 'a'".

    Args:
        keywords       : list of words to rank
        reference_text : the text to compute TF-IDF scores from

    Returns:
        List of (keyword, score) tuples, sorted by score descending
    """
    if not keywords or not reference_text.strip():
        return []

    # Create a single-document vectorizer
    # We fit on reference_text alone to get word importance within JD
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)

    try:
        # fit_transform on a single document returns shape (1, V)
        tfidf_matrix = vectorizer.fit_transform([reference_text])

        # get_feature_names_out() returns the vocabulary as an array
        # e.g. ["api", "aws", "docker", "machine learning", ...]
        feature_names = vectorizer.get_feature_names_out()

        # toarray() converts sparse matrix → dense numpy array
        # [0] gets the first (only) row → 1D array of TF-IDF scores
        scores = tfidf_matrix.toarray()[0]

        # Build a dict: word → score
        word_scores = dict(zip(feature_names, scores))

        # For each keyword, look up its score in the JD
        # .get(kw, 0.0) returns 0.0 if keyword not in vocabulary
        ranked = [(kw, round(word_scores.get(kw, 0.0), 4)) for kw in keywords]

        # Sort by score descending (highest importance first)
        ranked.sort(key=lambda x: x[1], reverse=True)

        return ranked

    except Exception:
        # If vectorizer fails (e.g. empty input), return unsorted list
        return [(kw, 0.0) for kw in keywords]


# ════════════════════════════════════════════════════════════════════
# SECTION 3 : SUB-SCORES BY RESUME SECTION
# ════════════════════════════════════════════════════════════════════

def compute_section_scores(
    resume_text : str,
    job_text    : str
) -> Dict[str, float]:
    """
    Break the resume into logical sections and score each one separately.

    Why section scores matter:
        A resume can score 65% overall but have 90% skill match and only
        20% experience match. The user needs to know WHERE to improve.

    Strategy:
        We use keyword-based heuristics to extract text belonging to
        each section, then compute TF-IDF score for each section vs JD.

    Sections detected:
        skills     : text near "skills", "technologies", "tools"
        experience : text near "experience", "work", "employment"
        education  : text near "education", "degree", "university"

    Args:
        resume_text : raw resume text
        job_text    : raw job description text

    Returns:
        Dict: {"skills": 72.5, "experience": 48.0, "education": 31.0}
    """
    lines = resume_text.lower().splitlines()

    # ── Section keyword markers ────────────────────────────────────
    # These words signal that the following lines belong to that section
    section_markers = {
        "skills"     : {"skill", "technolog", "tool", "proficien",
                        "competenc", "technical", "stack"},
        "experience" : {"experience", "work", "employ", "position",
                        "career", "job", "intern", "project"},
        "education"  : {"education", "degree", "university", "college",
                        "school", "academic", "qualification", "study"},
    }

    # ── Extract text per section using sliding window ──────────────
    section_texts = {s: [] for s in section_markers}
    current_section = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Check if this line is a section header
        for section, markers in section_markers.items():
            # any() returns True if at least one marker is in the line
            if any(marker in stripped for marker in markers):
                current_section = section
                break   # found the section — no need to check others

        # If we know which section we're in, add this line to it
        if current_section:
            section_texts[current_section].append(stripped)

    # ── Compute TF-IDF score for each section vs job description ──
    section_scores = {}
    clean_job = clean_text_as_string(job_text)

    for section, text_lines in section_texts.items():
        if text_lines:
            section_text = " ".join(text_lines)
            clean_section = clean_text_as_string(section_text)

            if clean_section.strip():
                # Score this section against full job description
                raw_score = compute_tfidf_score(section_text, job_text)
                section_scores[section] = round(raw_score * 100, 1)
            else:
                section_scores[section] = 0.0
        else:
            section_scores[section] = 0.0   # section not found in resume

    return section_scores


# ════════════════════════════════════════════════════════════════════
# SECTION 4 : WEIGHTED FINAL SCORE
# ════════════════════════════════════════════════════════════════════

def compute_weighted_score(
    tfidf_score    : float,
    skill_coverage : float,
    section_scores : Dict[str, float]
) -> Dict[str, float]:
    """
    Combine multiple signals into a single weighted ATS score.

    Raw TF-IDF alone is a good signal but not perfect — it doesn't know
    which parts of a resume matter most. We blend:

        40% TF-IDF cosine similarity   → vocabulary overlap quality
        35% Skill coverage score       → exact skill keyword matching
        15% Experience section score   → how well exp matches JD
        10% Skills section score       → dedicated skills section match

    Weights were chosen based on how real ATS systems prioritize:
        - Overall vocabulary match is the strongest signal
        - Exact skill keywords are heavily weighted in ATS parsing
        - Experience section is crucial for seniority matching
        - Skills section is a bonus (some resumes don't have one)

    Args:
        tfidf_score    : raw cosine similarity (0.0 to 1.0)
        skill_coverage : fraction of job skills found in resume (0.0 to 1.0)
        section_scores : {"skills": 72.5, "experience": 48.0, "education": 31.0}

    Returns:
        Dict with:
            weighted_score   : final 0-100 score
            component_scores : breakdown of each component's contribution
    """
    # Normalize section scores from 0-100 to 0-1 for consistent weighting
    exp_score   = section_scores.get("experience", 0.0) / 100
    skills_sec  = section_scores.get("skills", 0.0) / 100

    # ── Apply weights ──────────────────────────────────────────────
    weights = {
        "tfidf"      : 0.40,
        "skills"     : 0.35,
        "experience" : 0.15,
        "skills_sec" : 0.10,
    }

    weighted = (
        tfidf_score    * weights["tfidf"]      +
        skill_coverage * weights["skills"]     +
        exp_score      * weights["experience"] +
        skills_sec     * weights["skills_sec"]
    )

    # Convert to 0-100 and clamp
    final_score = float(np.clip(weighted * 100, 0.0, 100.0))

    # ── Component breakdown (for UI display) ──────────────────────
    component_scores = {
        "TF-IDF Match"         : round(tfidf_score * 100, 1),
        "Skill Coverage"       : round(skill_coverage * 100, 1),
        "Experience Match"     : round(exp_score * 100, 1),
        "Skills Section Match" : round(skills_sec * 100, 1),
        "Final Weighted Score" : round(final_score, 1),
    }

    return {
        "weighted_score"   : round(final_score, 1),
        "component_scores" : component_scores,
    }


# ════════════════════════════════════════════════════════════════════
# SECTION 5 : SCORE INTERPRETATION
# ════════════════════════════════════════════════════════════════════

def interpret_score(score: float) -> Dict[str, str]:
    """
    Convert a numeric score into a human-readable grade and advice.

    Used in the UI to show a colored badge and actionable message.

    Args:
        score : float 0 to 100

    Returns:
        Dict with:
            grade  : letter grade (A, B, C, D, F)
            label  : text label ("Excellent", "Good", etc.)
            color  : CSS color name for Streamlit badge
            advice : one-line actionable recommendation
    """
    if score >= 80:
        return {
            "grade"  : "A",
            "label"  : "Excellent Match",
            "color"  : "green",
            "advice" : "Strong alignment. Focus on tailoring your summary section.",
        }
    elif score >= 65:
        return {
            "grade"  : "B",
            "label"  : "Good Match",
            "color"  : "blue",
            "advice" : "Add missing keywords from the job description to boost score.",
        }
    elif score >= 50:
        return {
            "grade"  : "C",
            "label"  : "Moderate Match",
            "color"  : "orange",
            "advice" : "Significant gaps found. Prioritize the missing skills section.",
        }
    elif score >= 35:
        return {
            "grade"  : "D",
            "label"  : "Weak Match",
            "color"  : "red",
            "advice" : "Low overlap with job requirements. Consider upskilling first.",
        }
    else:
        return {
            "grade"  : "F",
            "label"  : "Poor Match",
            "color"  : "red",
            "advice" : "Very low match. This role may need different skills entirely.",
        }


# ════════════════════════════════════════════════════════════════════
# SECTION 6 : MASTER FUNCTION
# ════════════════════════════════════════════════════════════════════

def get_full_score_report(
    resume_text : str,
    job_text    : str
) -> dict:
    """
    Master function: runs the complete scoring pipeline.
    This is the ONLY function app.py needs to import from scorer.py.

    Pipeline:
        1. TF-IDF cosine similarity (raw match score)
        2. Keyword overlap analysis (which words matched/missing)
        3. Section-level scores (skills / experience / education)
        4. Skill coverage from keyword_extractor (Day 4)
        5. Weighted final score combining all signals
        6. Score interpretation (grade + advice)

    Args:
        resume_text : raw text from resume
        job_text    : raw text from job description

    Returns:
        Dict with complete scoring report — all keys documented inline
    """
    # ── Step 1: Raw TF-IDF score ───────────────────────────────────
    tfidf_raw   = compute_tfidf_score(resume_text, job_text)
    tfidf_score = round(tfidf_raw * 100, 1)

    # ── Step 2: Keyword overlap ────────────────────────────────────
    overlap = get_keyword_overlap(resume_text, job_text)

    # ── Step 3: Section scores ─────────────────────────────────────
    section_scores = compute_section_scores(resume_text, job_text)

    # ── Step 4: Skill coverage (bridges Day 4 → Day 5) ────────────
    # Use the skill extractor from Day 4 to measure exact skill overlap
    resume_skills = extract_skills_from_text(resume_text)
    job_skills    = extract_skills_from_text(job_text)

    if job_skills:
        # What fraction of required job skills does resume have?
        matched_skills   = resume_skills & job_skills
        skill_coverage   = len(matched_skills) / len(job_skills)
        skill_coverage_pct = round(skill_coverage * 100, 1)
    else:
        skill_coverage     = 0.0
        skill_coverage_pct = 0.0

    # ── Step 5: Weighted final score ───────────────────────────────
    weighted = compute_weighted_score(tfidf_raw, skill_coverage, section_scores)

    # ── Step 6: Interpretation ─────────────────────────────────────
    interpretation = interpret_score(weighted["weighted_score"])

    # ── Assemble full report ───────────────────────────────────────
    return {
        # Core scores
        "tfidf_score"        : tfidf_score,
        "final_score"        : weighted["weighted_score"],
        "skill_coverage_pct" : skill_coverage_pct,

        # Section breakdown
        "section_scores"     : section_scores,
        "component_scores"   : weighted["component_scores"],

        # Keyword analysis
        "keyword_overlap"    : overlap,
        "resume_skills"      : resume_skills,
        "job_skills"         : job_skills,

        # Interpretation
        "grade"              : interpretation["grade"],
        "label"              : interpretation["label"],
        "color"              : interpretation["color"],
        "advice"             : interpretation["advice"],
    }