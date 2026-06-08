# utils/keyword_extractor.py
# ─────────────────────────────────────────────────────────────────────
# PURPOSE : Match resume tokens against the skills database.
#           Produces found skills, missing skills, and domain scores.
#
# KEY CONCEPT — Multi-word skill matching:
#   Simple token matching finds "python" but misses "machine learning"
#   because that's TWO tokens. We use two strategies:
#     1. Single-token match  : token in ALL_SKILLS
#     2. Bigram match        : (token[i] + " " + token[i+1]) in ALL_SKILLS
#   This catches "deep learning", "system design", "github actions" etc.
#
# USED BY : app.py (Day 8), utils/scorer.py (Day 5)
# ─────────────────────────────────────────────────────────────────────

from typing import List, Dict, Set, Tuple
import pandas as pd

# Import our skills database
from data.skills_db import SKILLS_DB, ALL_SKILLS, DOMAIN_LABELS, TCS_PRIORITY_DOMAINS

# Import cleaner to process job description text too
from utils.cleaner import clean_text, clean_text_as_string


# ════════════════════════════════════════════════════════════════════
# SECTION 1 : CORE MATCHING ENGINE
# ════════════════════════════════════════════════════════════════════

def extract_skills_from_tokens(tokens: List[str]) -> Set[str]:
    """
    Given a list of clean tokens, find all skills present.
    Uses both single-token and bigram (two-word) matching.

    Args:
        tokens : list of clean lemmatized tokens from cleaner.py

    Returns:
        Set of matched skill strings, e.g. {"python", "flask", "machine learning"}

    Example:
        tokens = ["python", "machine", "learning", "flask", "docker"]
        → single matches: "python", "flask", "docker"
        → bigram match:   "machine learning"
        → returns: {"python", "machine learning", "flask", "docker"}
    """
    found = set()    # use a set so duplicates are automatically ignored

    # ── Strategy 1: Single-token matching ─────────────────────────
    # Check if each individual token exists in our ALL_SKILLS set
    # Set lookup is O(1) — instant regardless of how big ALL_SKILLS is
    for token in tokens:
        if token in ALL_SKILLS:
            found.add(token)

    # ── Strategy 2: Bigram matching ───────────────────────────────
    # A bigram is a pair of consecutive tokens
    # We slide a window of size 2 across the token list
    #
    # tokens = ["machine", "learning", "deep", "learning"]
    # bigrams = [("machine","learning"), ("learning","deep"), ("deep","learning")]
    #
    # range(len(tokens) - 1) stops one before the end
    # so tokens[i+1] never goes out of bounds
    for i in range(len(tokens) - 1):
        bigram = tokens[i] + " " + tokens[i + 1]
        # e.g. "machine" + " " + "learning" = "machine learning"

        if bigram in ALL_SKILLS:
            found.add(bigram)

    return found


def match_skills_by_domain(found_skills: Set[str]) -> Dict[str, Set[str]]:
    """
    Take a flat set of found skills and group them back into domains.

    Why we need this:
        extract_skills_from_tokens() returns {"python", "flask", "docker"}
        But we want to know: python is in "languages", flask is in "web", etc.
        This function does that grouping.

    Args:
        found_skills : set of matched skill strings

    Returns:
        Dict mapping domain_key → set of found skills in that domain
        e.g. {
            "languages": {"python", "java"},
            "web":       {"flask", "react"},
            "cloud_devops": {"docker"},
        }
    """
    domain_matches = {}

    for domain, skill_set in SKILLS_DB.items():
        # Set intersection: elements that exist in BOTH sets
        # found_skills & skill_set = skills that are found AND in this domain
        matched = found_skills & skill_set

        if matched:    # only include domains where at least one skill was found
            domain_matches[domain] = matched

    return domain_matches


# ════════════════════════════════════════════════════════════════════
# SECTION 2 : SCORING ENGINE
# ════════════════════════════════════════════════════════════════════

def calculate_domain_scores(found_skills: Set[str]) -> Dict[str, float]:
    """
    For each domain, calculate what percentage of its skills appear in resume.

    Formula:
        score = (skills found in domain / total skills in domain) * 100

    Example:
        languages domain has 25 skills.
        Resume has: python, java, c++ → 3 found.
        Score = (3 / 25) * 100 = 12.0%

    Note: A 12% score sounds low, but no one knows ALL 25 languages.
    This score is relative — use it to COMPARE two resumes, not as absolute grade.

    Args:
        found_skills : set of all found skills

    Returns:
        Dict mapping domain_key → score (float, 0.0 to 100.0)
        e.g. {"languages": 12.0, "web": 35.0, "ml_ai": 8.0}
    """
    scores = {}

    for domain, skill_set in SKILLS_DB.items():
        total_in_domain = len(skill_set)        # how many skills exist in this domain
        found_in_domain = len(found_skills & skill_set)  # how many resume has

        # Avoid division by zero (shouldn't happen but defensive programming)
        if total_in_domain == 0:
            scores[domain] = 0.0
        else:
            score = (found_in_domain / total_in_domain) * 100
            scores[domain] = round(score, 1)    # round to 1 decimal place

    return scores


def calculate_overall_skill_score(found_skills: Set[str]) -> float:
    """
    Calculate a single overall skill coverage score (0-100).

    Formula:
        overall = (total unique skills found / total skills in DB) * 100

    This is a breadth score — rewards knowing skills across many domains.

    Args:
        found_skills : set of all found skills

    Returns:
        Float from 0.0 to 100.0
    """
    total_skills = len(ALL_SKILLS)
    found_count  = len(found_skills)

    if total_skills == 0:
        return 0.0

    return round((found_count / total_skills) * 100, 1)


# ════════════════════════════════════════════════════════════════════
# SECTION 3 : GAP ANALYSIS
# ════════════════════════════════════════════════════════════════════

def find_skill_gaps(
    resume_skills: Set[str],
    job_skills: Set[str]
) -> Dict[str, Set[str]]:
    """
    Compare resume skills against job description skills to find gaps.

    This is the most valuable output for the user — it tells them
    exactly WHAT to add to their resume.

    Set operations used:
        A & B  = intersection = skills in BOTH (resume has what job needs)
        A - B  = difference   = skills in A but not B
        B - A  = difference   = skills in B but not A (THE GAPS)

    Args:
        resume_skills : skills found in resume
        job_skills    : skills found in job description

    Returns:
        Dict with three keys:
            "matched" : skills present in both (good!)
            "missing" : job needs these but resume doesn't have them (add these!)
            "extra"   : resume has these but job didn't ask for (bonus)
    """
    matched = resume_skills & job_skills      # intersection
    missing = job_skills - resume_skills      # in job but NOT in resume → gaps
    extra   = resume_skills - job_skills      # in resume but NOT in job → bonus

    return {
        "matched" : matched,
        "missing" : missing,
        "extra"   : extra,
    }


def get_missing_by_domain(missing_skills: Set[str]) -> Dict[str, List[str]]:
    """
    Group missing skills by domain so we can show targeted recommendations.

    Instead of showing a flat list of 15 missing skills, we show:
        "You're missing these Machine Learning skills: pytorch, mlflow"
        "You're missing these Cloud skills: kubernetes, terraform"

    Args:
        missing_skills : set of skill strings not in resume but in job desc

    Returns:
        Dict mapping domain_label → sorted list of missing skills
        e.g. {
            "Machine Learning & AI": ["pytorch", "mlflow"],
            "Cloud & DevOps": ["kubernetes", "terraform"],
        }
    """
    grouped = {}

    for domain, skill_set in SKILLS_DB.items():
        # skills that are missing AND belong to this domain
        domain_missing = list(missing_skills & skill_set)

        if domain_missing:
            label = DOMAIN_LABELS[domain]        # human-readable name
            grouped[label] = sorted(domain_missing)  # sorted for consistent display

    return grouped


# ════════════════════════════════════════════════════════════════════
# SECTION 4 : JOB DESCRIPTION ANALYSIS
# ════════════════════════════════════════════════════════════════════

def extract_skills_from_text(raw_text: str) -> Set[str]:
    """
    Full pipeline: raw text → clean tokens → skill set.
    Works for BOTH resume text and job description text.

    Args:
        raw_text : any raw text string

    Returns:
        Set of found skills
    """
    tokens = clean_text(raw_text)           # Day 3's cleaner
    skills = extract_skills_from_tokens(tokens)
    return skills


# ════════════════════════════════════════════════════════════════════
# SECTION 5 : DATAFRAME OUTPUT (for UI tables)
# ════════════════════════════════════════════════════════════════════

def skills_to_dataframe(
    found_skills   : Set[str],
    domain_scores  : Dict[str, float]
) -> pd.DataFrame:
    """
    Convert skill analysis results into a DataFrame for display in Streamlit.

    Each row = one skill, with its domain and whether it was found.

    Args:
        found_skills  : set of skills found in resume
        domain_scores : dict of domain → score from calculate_domain_scores()

    Returns:
        DataFrame with columns: skill, domain, found, domain_score
    """
    rows = []

    for domain, skill_set in SKILLS_DB.items():
        label = DOMAIN_LABELS[domain]
        score = domain_scores.get(domain, 0.0)

        for skill in sorted(skill_set):   # sorted for consistent ordering
            rows.append({
                "skill"        : skill,
                "domain"       : label,
                "found"        : skill in found_skills,   # True/False
                "domain_score" : score,
            })

    df = pd.DataFrame(rows)

    # Sort: found=True first, then alphabetically within domain
    # ascending=[False, True] means: found DESC (True before False),
    # skill ASC (a before z)
    df = df.sort_values(["found", "skill"], ascending=[False, True])
    df = df.reset_index(drop=True)   # reset row numbers after sorting

    return df


def get_analysis_summary(
    resume_text : str,
    job_text    : str = ""
) -> dict:
    """
    Master function: runs complete skill analysis and returns all results.
    This is what app.py (Day 8) will call — one function, all results.

    Args:
        resume_text : raw text from resume (extractor.py output)
        job_text    : raw text from job description (optional)

    Returns:
        Dict with all analysis results:
            resume_skills   : set of skills found in resume
            job_skills      : set of skills found in job desc (empty if no JD)
            domain_matches  : {domain → set of found skills}
            domain_scores   : {domain → percentage score}
            overall_score   : single float 0-100
            gap_analysis    : {matched, missing, extra}
            missing_by_domain : {domain_label → [missing skills]}
            skills_df       : full DataFrame for table display
    """
    # ── Step 1: Extract skills from resume ────────────────────────
    resume_skills = extract_skills_from_text(resume_text)

    # ── Step 2: Extract skills from job description ───────────────
    job_skills = extract_skills_from_text(job_text) if job_text.strip() else set()

    # ── Step 3: Domain grouping and scoring ───────────────────────
    domain_matches = match_skills_by_domain(resume_skills)
    domain_scores  = calculate_domain_scores(resume_skills)
    overall_score  = calculate_overall_skill_score(resume_skills)

    # ── Step 4: Gap analysis (only meaningful if JD provided) ─────
    if job_skills:
        gap_analysis     = find_skill_gaps(resume_skills, job_skills)
        missing_by_domain = get_missing_by_domain(gap_analysis["missing"])
    else:
        gap_analysis     = {"matched": set(), "missing": set(), "extra": set()}
        missing_by_domain = {}

    # ── Step 5: Build DataFrame ───────────────────────────────────
    skills_df = skills_to_dataframe(resume_skills, domain_scores)

    return {
        "resume_skills"    : resume_skills,
        "job_skills"       : job_skills,
        "domain_matches"   : domain_matches,
        "domain_scores"    : domain_scores,
        "overall_score"    : overall_score,
        "gap_analysis"     : gap_analysis,
        "missing_by_domain": missing_by_domain,
        "skills_df"        : skills_df,
    }