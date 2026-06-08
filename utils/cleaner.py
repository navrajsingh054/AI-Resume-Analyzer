# utils/cleaner.py
# ─────────────────────────────────────────────────────────────────────
# PURPOSE : Clean and preprocess raw resume text using NLP.
#           Converts messy extracted text into clean tokens
#           that the scorer and keyword extractor can work with.
#
# PIPELINE:
#   raw text → lowercase → remove noise → tokenize →
#   remove stopwords → lemmatize → clean token list
#
# USED BY : utils/scorer.py (Day 5), utils/keyword_extractor.py (Day 4)
# ─────────────────────────────────────────────────────────────────────

import re                    # built-in: regular expressions for pattern matching
import spacy                 # industrial NLP library
from typing import List      # for type hints — makes code self-documenting


# ── Load spaCy model ONCE at module level ─────────────────────────────
# Loading a model is expensive (takes ~0.5 seconds).
# By doing it here (module level), it loads once when the file is first
# imported, and every function call reuses the same loaded model.
# If you loaded it inside a function, it would reload on every call — very slow.
nlp = spacy.load("en_core_web_sm")


# ── Custom stopwords ──────────────────────────────────────────────────
# spaCy has 326 built-in English stopwords.
# We ADD resume-specific filler words that spaCy doesn't know are noise.
# These words appear constantly in resumes but carry zero meaning for matching.
RESUME_STOPWORDS = {
    "experience", "work", "working", "worked",
    "responsible", "responsibility", "responsibilities",
    "include", "includes", "including", "included",
    "use", "used", "using", "utilize", "utilized",
    "develop", "developed", "developing",  # too generic; skills matter, not the verb
    "manage", "managed", "managing",
    "company", "team", "project", "role",
    "year", "years", "month", "months",
    "good", "great", "excellent", "strong", "ability",
    "etc", "also", "well", "various",
}
# NOTE: We keep technical terms like "python", "machine", "learning" — those matter!


# ════════════════════════════════════════════════════════════════════
# SECTION 1 : NOISE REMOVAL
# ════════════════════════════════════════════════════════════════════

def remove_noise(text: str) -> str:
    """
    Remove non-content elements from raw text using regular expressions.

    We remove these in a specific order — each step's output feeds the next.

    Args:
        text : raw extracted string from extractor.py

    Returns:
        Cleaner string with noise removed (still one big string, not tokens yet)
    """

    # ── 1. Email addresses ─────────────────────────────────────────
    # Pattern: word chars + @ + word chars + . + word chars
    # Example match: "rahul.sharma@gmail.com"
    # Why remove: emails are unique per person, not useful for skill matching
    text = re.sub(r'\S+@\S+', '', text)
    # \S+ means "one or more non-whitespace characters"
    # @ is literal @
    # So this matches anything like "abc@def.com" or "user@company.org"

    # ── 2. URLs ────────────────────────────────────────────────────
    # Pattern: http or https, then ://, then anything non-whitespace
    # Example match: "https://github.com/rahul/project"
    text = re.sub(r'https?://\S+', '', text)
    # https? means "http" OR "https" (? makes the 's' optional)
    # :// is literal
    # \S+ matches the rest of the URL

    # ── 3. LinkedIn / GitHub URLs that don't start with http ──────
    # Example match: "linkedin.com/in/rahulsharma"
    text = re.sub(r'(linkedin|github|twitter)\.com/\S+', '', text, flags=re.IGNORECASE)
    # flags=re.IGNORECASE means "LinkedIn.com" also matches

    # ── 4. Phone numbers ───────────────────────────────────────────
    # Pattern covers: +91-9876543210, (415) 555-0172, 9876543210
    # Why remove: phone numbers are personal identifiers, not skills
    text = re.sub(r'[\+]?[\d]?[\s\-\.]?[\(]?[\d]{3}[\)]?[\s\-\.]?[\d]{3}[\s\-\.]?[\d]{4}', '', text)

    # ── 5. Special characters — keep only letters, digits, spaces ─
    # Replace anything that isn't a-z, A-Z, 0-9, or whitespace with a space
    # This removes: @#$%^&*()[]{}|<>, commas, exclamation marks, etc.
    # We KEEP digits here (removed in next step) because "Python 3" is meaningful context
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

    # ── 6. Standalone numbers ──────────────────────────────────────
    # Remove numbers that stand alone (not part of a word like "Python3")
    # \b is a word boundary — matches position between word and non-word char
    # So \b\d+\b matches "3" in "3 years" but NOT "3" in "Python3"
    text = re.sub(r'\b\d+\b', '', text)

    # ── 7. Extra whitespace ────────────────────────────────────────
    # After all the removals above, we have lots of extra spaces.
    # \s+ matches one or more whitespace characters (spaces, tabs, newlines)
    # Replace all with a single space
    text = re.sub(r'\s+', ' ', text)

    return text.strip()   # remove leading/trailing whitespace


# ════════════════════════════════════════════════════════════════════
# SECTION 2 : TOKENIZATION
# ════════════════════════════════════════════════════════════════════

def tokenize(text: str) -> List[str]:
    """
    Split a string into a list of individual word tokens using spaCy.

    Why spaCy instead of text.split()?
    - spaCy handles contractions: "I'm" → ["I", "'m"]
    - spaCy handles punctuation: "Python." → ["Python", "."]
    - text.split() would give ["Python."] — dot stays attached

    Args:
        text : cleaned string (output of remove_noise)

    Returns:
        List of token strings, e.g. ["python", "developer", "flask"]
    """
    text = text.lower()          # lowercase BEFORE passing to spaCy
                                  # spaCy is case-sensitive for some lookups

    doc = nlp(text)              # nlp() runs the full spaCy pipeline on text
                                  # 'doc' is a Doc object — a sequence of Token objects
                                  # Token objects have: .text, .lemma_, .is_stop, .pos_, etc.

    tokens = [token.text for token in doc]
    # List comprehension: for each Token object in doc, get its .text string
    # This gives us the raw text of each token (not yet lemmatized)

    return tokens


# ════════════════════════════════════════════════════════════════════
# SECTION 3 : STOPWORD REMOVAL
# ════════════════════════════════════════════════════════════════════

def remove_stopwords(tokens: List[str]) -> List[str]:
    """
    Filter out stopwords from a token list.

    Stopwords are very common words that carry no meaning for matching:
    "the", "a", "is", "in", "and", "with", "for", "of" ...

    We combine:
    1. spaCy's built-in 326 English stopwords (token.is_stop check)
    2. Our custom RESUME_STOPWORDS set defined above

    Args:
        tokens : list of word strings (output of tokenize)

    Returns:
        Filtered list with stopwords removed
    """
    # We need spaCy token objects to use .is_stop, so we re-process
    # Joining and re-parsing is slightly wasteful but keeps functions clean
    doc = nlp(" ".join(tokens))

    clean_tokens = []
    for token in doc:
        # Condition 1: token.is_stop → spaCy says this is a common English word
        # Condition 2: token.text in RESUME_STOPWORDS → our custom list
        # Condition 3: len(token.text) <= 1 → single characters like "a", "i", "x"
        # If ANY of these is True, we skip this token
        if token.is_stop:
            continue
        if token.text in RESUME_STOPWORDS:
            continue
        if len(token.text) <= 1:
            continue

        clean_tokens.append(token.text)

    return clean_tokens


# ════════════════════════════════════════════════════════════════════
# SECTION 4 : LEMMATIZATION
# ════════════════════════════════════════════════════════════════════

def lemmatize(tokens: List[str]) -> List[str]:
    """
    Convert each token to its base/root form using spaCy's lemmatizer.

    Examples:
        "databases"  → "database"
        "algorithms" → "algorithm"
        "built"      → "build"
        "designing"  → "design"
        "faster"     → "fast"

    Why this matters:
    If a resume says "databases" and job description says "database",
    without lemmatization they won't match. After lemmatization, both
    become "database" and they match correctly.

    Args:
        tokens : list of token strings (output of remove_stopwords)

    Returns:
        List with each token replaced by its lemma
    """
    doc = nlp(" ".join(tokens))    # re-parse so we get Token objects with .lemma_

    lemmas = []
    for token in doc:
        lemma = token.lemma_       # .lemma_ gives the base form (underscore = string version)
                                    # token.lemma (no underscore) gives an integer ID — not what we want

        # Some lemmas start with "-" (spaCy artifact for punctuation)
        # e.g. lemma of "." is "." — we skip these
        if lemma.startswith("-"):
            continue

        lemmas.append(lemma)

    return lemmas


# ════════════════════════════════════════════════════════════════════
# SECTION 5 : MAIN PIPELINE — the only function other files need
# ════════════════════════════════════════════════════════════════════

def clean_text(text: str) -> List[str]:
    """
    Master function: runs the complete NLP pipeline end-to-end.

    Pipeline:
        raw text
          → remove_noise()      removes emails, URLs, special chars
          → tokenize()          splits into word list
          → remove_stopwords()  filters filler words
          → lemmatize()         converts to root forms
          → final clean tokens

    This is the ONLY function that scorer.py and keyword_extractor.py import.
    They don't need to know about the individual steps.

    Args:
        text : raw string from extractor.py

    Returns:
        List of clean, meaningful tokens ready for analysis
    """
    if not text or not text.strip():
        return []                   # guard against empty input — return empty list

    step1 = remove_noise(text)      # step 1: strip emails, URLs, special chars
    step2 = tokenize(step1)         # step 2: split into token list
    step3 = remove_stopwords(step2) # step 3: remove filler words
    step4 = lemmatize(step3)        # step 4: convert to base forms

    # Final filter: remove any empty strings or whitespace-only strings
    # that might have slipped through
    final = [t for t in step4 if t.strip()]

    return final


def clean_text_as_string(text: str) -> str:
    """
    Same as clean_text() but returns a single joined string instead of a list.

    Why: TF-IDF vectorizer (Day 5) needs a string, not a list.
    Keyword extractor (Day 4) needs a list.
    So we provide both versions.

    Returns:
        Space-joined string of clean tokens, e.g. "python flask api build"
    """
    tokens = clean_text(text)
    return " ".join(tokens)


# ════════════════════════════════════════════════════════════════════
# SECTION 6 : ANALYSIS HELPER — for showing stats in the UI
# ════════════════════════════════════════════════════════════════════

def get_text_stats(raw_text: str, clean_tokens: List[str]) -> dict:
    """
    Compute before/after statistics to show in the Streamlit UI (Day 8).
    Helps users understand what the cleaning step did.

    Args:
        raw_text     : original text before cleaning
        clean_tokens : token list after full pipeline

    Returns:
        Dictionary with keys:
            raw_char_count    : total characters in raw text
            raw_word_count    : rough word count (split by spaces)
            clean_token_count : tokens after full pipeline
            reduction_pct     : how much we reduced the text (percentage)
            top_tokens        : 10 most frequent tokens (for word cloud)
    """
    raw_words = len(raw_text.split())
    clean_count = len(clean_tokens)

    # Calculate reduction percentage
    # Formula: ((original - cleaned) / original) * 100
    if raw_words > 0:
        reduction = round(((raw_words - clean_count) / raw_words) * 100, 1)
    else:
        reduction = 0.0

    # Count token frequency using a dictionary
    # We'll use pandas Series for easy sorting
    import pandas as pd
    freq = pd.Series(clean_tokens).value_counts()  # counts occurrences of each token

    # .head(10) gives the top 10 most frequent
    # .to_dict() converts Series → {"python": 5, "develop": 3, ...}
    top_tokens = freq.head(10).to_dict()

    return {
        "raw_char_count"    : len(raw_text),
        "raw_word_count"    : raw_words,
        "clean_token_count" : clean_count,
        "reduction_pct"     : reduction,
        "top_tokens"        : top_tokens,
    }