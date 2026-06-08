# utils/section_detector.py
# ─────────────────────────────────────────────────────────────────────
# PURPOSE : Detect and parse resume sections using rule-based heading
#           detection combined with spaCy Named Entity Recognition.
#
# STRATEGY:
#   1. Split resume into lines
#   2. Classify each line as HEADING or BODY using font heuristics
#      (from extractor.py Day 2) OR keyword pattern matching
#   3. Group consecutive body lines under their heading → sections
#   4. For each section, run spaCy NER to extract structured data
#
# SECTIONS DETECTED:
#   contact, summary, skills, experience, education, projects,
#   certifications, achievements, languages
#
# USED BY : app.py (Day 8), utils/scorer.py (Day 5 section scores)
# ─────────────────────────────────────────────────────────────────────

import re
import spacy
import pandas as pd
from typing import Dict, List, Tuple, Optional

nlp = spacy.load("en_core_web_sm")   # load once at module level (expensive)


# ════════════════════════════════════════════════════════════════════
# SECTION 1 : HEADING PATTERN DEFINITIONS
# ════════════════════════════════════════════════════════════════════

# Each key is our internal section name.
# Each value is a set of lowercase keywords that signal that section's heading.
# We use substring matching — "work experience" contains "experience" → match.
SECTION_PATTERNS = {
    "contact": {
        "contact", "personal", "profile", "about me", "information"
    },
    "summary": {
        "summary", "objective", "overview", "introduction",
        "professional summary", "career objective", "about"
    },
    "skills": {
        "skill", "technical skill", "core competenc", "technology",
        "tool", "stack", "expertise", "proficien", "competenc",
        "programming language", "framework"
    },
    "experience": {
        "experience", "work experience", "employment", "work history",
        "professional experience", "career", "intern", "position",
        "job history", "industrial"
    },
    "education": {
        "education", "academic", "qualification", "degree",
        "university", "college", "school", "study", "studies"
    },
    "projects": {
        "project", "personal project", "academic project",
        "side project", "portfolio", "work sample"
    },
    "certifications": {
        "certif", "course", "training", "license", "accreditation",
        "mooc", "udemy", "coursera"
    },
    "achievements": {
        "achievement", "award", "honor", "honour", "recognition",
        "accomplishment", "scholarship", "rank", "winner"
    },
    "languages": {
        "language", "spoken language", "linguistic"
    },
}

# These are patterns that STRONGLY indicate a line is a heading
# even without font size data — all-caps words, short lines ending
# with colon, lines with only title-case words, etc.
HEADING_REGEX_PATTERNS = [
    r'^[A-Z][A-Z\s&/]{3,30}$',          # ALL CAPS: "WORK EXPERIENCE"
    r'^[A-Z][a-z]+(\s[A-Z][a-z]+){0,4}:?$',  # Title Case: "Work Experience"
    r'^\w[\w\s]{2,25}:$',                # ends with colon: "Skills:"
]


# ════════════════════════════════════════════════════════════════════
# SECTION 2 : LINE CLASSIFICATION
# ════════════════════════════════════════════════════════════════════

def classify_line(line: str) -> Tuple[bool, Optional[str]]:
    """
    Decide if a line is a section heading, and if so which section.

    Returns:
        (is_heading: bool, section_name: Optional[str])

        If is_heading=True  → section_name is the matched section key
        If is_heading=False → section_name is None (just a body line)

    Strategy (in order of priority):
        1. Check if line text matches any SECTION_PATTERNS keyword
        2. Check if line matches HEADING_REGEX_PATTERNS (structural check)
        3. Otherwise → body line
    """
    stripped = line.strip()
    if not stripped:
        return False, None

    lower = stripped.lower()

    # ── Priority 1: keyword match ──────────────────────────────────
    # Check if any section's keywords appear in this line
    for section, keywords in SECTION_PATTERNS.items():
        for kw in keywords:
            if kw in lower:
                # Extra check: the line should be SHORT (headings are rarely
                # full sentences). If it's a sentence with 10+ words,
                # the keyword is probably inside body text, not a heading.
                word_count = len(stripped.split())
                if word_count <= 7:           # "Work Experience 2024" = 3 words ✓
                    return True, section      # "Developed skills in Python" = 5 words ✗
                                              # Wait — 5 words with "skill" → heading?
                                              # We check word_count <= 7 to be safe.
                                              # Could miss "Professional Work Experience"
                                              # but avoids false positives in body text.

    # ── Priority 2: structural regex match ────────────────────────
    # Even if no keyword matched, all-caps / title-case short lines
    # are almost always headings in a resume
    for pattern in HEADING_REGEX_PATTERNS:
        if re.match(pattern, stripped):
            # We don't know WHICH section — return generic heading signal
            return True, "unknown"

    return False, None


# ════════════════════════════════════════════════════════════════════
# SECTION 3 : SECTION SPLITTER
# ════════════════════════════════════════════════════════════════════

def split_into_sections(text: str) -> Dict[str, str]:
    """
    Split a full resume text into named sections.

    Algorithm:
        1. Split text into lines
        2. For each line, call classify_line()
        3. When a heading is found, start a new section bucket
        4. All following body lines go into current section bucket
        5. Continue until next heading found

    Args:
        text : full raw resume text (extractor.py output)

    Returns:
        Dict mapping section_name → raw text block for that section
        e.g. {
            "skills":     "Python, Java, Flask, Docker...",
            "experience": "Software Intern at TCS...",
            "education":  "B.Tech CSE, PTU, CGPA 8.4",
        }
    """
    lines = text.splitlines()

    sections: Dict[str, List[str]] = {}   # section_name → list of lines
    current_section = "header"            # lines before first heading go to "header"
    sections["header"] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue                       # skip blank lines

        is_heading, section_name = classify_line(stripped)

        if is_heading and section_name and section_name != "unknown":
            # Found a known section heading → switch current section
            current_section = section_name
            if current_section not in sections:
                sections[current_section] = []
            # Don't add the heading line itself to the content

        elif is_heading and section_name == "unknown":
            # Structural heading but unknown type → keep in current section
            # (avoids losing content from unrecognised headings)
            sections.setdefault(current_section, []).append(stripped)

        else:
            # Body line → add to current section
            sections.setdefault(current_section, []).append(stripped)

    # Join each section's line list into a single string
    return {
        section: "\n".join(lines_list)
        for section, lines_list in sections.items()
        if lines_list                      # skip empty sections
    }


# ════════════════════════════════════════════════════════════════════
# SECTION 4 : NER-BASED ENTITY EXTRACTION
# ════════════════════════════════════════════════════════════════════

def extract_entities(text: str) -> Dict[str, List[str]]:
    """
    Run spaCy NER on a text block and return entities grouped by type.

    Args:
        text : any text string (usually one section's content)

    Returns:
        Dict mapping entity_label → list of entity strings
        e.g. {
            "ORG"  : ["TCS", "Google", "Punjab Technical University"],
            "DATE" : ["June 2024", "2021", "3 years"],
            "GPE"  : ["Pune", "Hyderabad"],
        }
    """
    if not text.strip():
        return {}

    doc = nlp(text)           # run full spaCy pipeline including NER

    entities: Dict[str, List[str]] = {}

    for ent in doc.ents:
        # ent.label_ → entity type string e.g. "ORG", "DATE", "GPE"
        # ent.text   → the actual text span e.g. "TCS"

        label = ent.label_
        entity_text = ent.text.strip()

        # Filter out very short or very long entities (usually noise)
        if len(entity_text) < 2 or len(entity_text) > 60:
            continue

        if label not in entities:
            entities[label] = []

        # Avoid duplicates
        if entity_text not in entities[label]:
            entities[label].append(entity_text)

    return entities


def extract_candidate_name(header_text: str) -> str:
    """
    Extract the candidate's name from the resume header.

    Strategy:
        1. Try spaCy PERSON entity on the first 3 lines (name is usually first)
        2. Fall back to the first non-empty line (heuristic)

    Args:
        header_text : text before the first section heading

    Returns:
        Candidate name string, or "Unknown" if not found
    """
    if not header_text.strip():
        return "Unknown"

    # Only look at first 3 lines — name is almost always at the top
    first_lines = "\n".join(header_text.strip().splitlines()[:3])
    doc = nlp(first_lines)

    # Look for PERSON entity first
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text.strip()

    # Fallback: first non-empty line is usually the name
    for line in header_text.splitlines():
        cleaned = line.strip()
        # Name line: not too long, no @ (email), no digits (phone)
        if cleaned and len(cleaned) < 50 and "@" not in cleaned and not re.search(r'\d{5,}', cleaned):
            return cleaned

    return "Unknown"


def extract_contact_info(text: str) -> Dict[str, str]:
    """
    Extract email, phone, LinkedIn, and GitHub from resume text.

    Uses regex patterns — more reliable than NER for structured contact info.

    Args:
        text : full resume text or header section text

    Returns:
        Dict with keys: email, phone, linkedin, github
        Missing values are empty strings.
    """
    contact = {
        "email"   : "",
        "phone"   : "",
        "linkedin": "",
        "github"  : "",
    }

    # Email pattern
    email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    if email_match:
        contact["email"] = email_match.group()

    # Phone: matches +91-9876543210, 9876543210, (415) 555-0172
    phone_match = re.search(
        r'[\+]?[\d]?[\s\-\.]?[\(]?[\d]{3}[\)]?[\s\-\.]?[\d]{3}[\s\-\.]?[\d]{4}',
        text
    )
    if phone_match:
        contact["phone"] = phone_match.group().strip()

    # LinkedIn URL
    linkedin_match = re.search(
        r'linkedin\.com/in/[\w\-]+', text, re.IGNORECASE
    )
    if linkedin_match:
        contact["linkedin"] = "https://" + linkedin_match.group()

    # GitHub URL
    github_match = re.search(
        r'github\.com/[\w\-]+', text, re.IGNORECASE
    )
    if github_match:
        contact["github"] = "https://" + github_match.group()

    return contact


# ════════════════════════════════════════════════════════════════════
# SECTION 5 : SECTION-SPECIFIC PARSERS
# ════════════════════════════════════════════════════════════════════

def parse_skills_section(text: str) -> List[str]:
    """
    Extract individual skills from the skills section text.

    Handles multiple formats:
        Comma-separated : "Python, Flask, Docker, PostgreSQL"
        Bullet points   : "- Python\n- Flask\n- Docker"
        Category labels : "Languages: Python, Java\nFrameworks: Flask, Django"

    Args:
        text : raw skills section text

    Returns:
        List of individual skill strings, cleaned and deduplicated
    """
    skills = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        # If line has a category label like "Languages: Python, Java"
        # strip ONLY the label part before the colon, keep everything after
        # re.sub with count=1 replaces only the FIRST match on this line
        if re.match(r'^[\w\s]+:', stripped):
            # e.g. "Programming Languages: Python, Java" 
            # → strip "Programming Languages: " → "Python, Java"
            stripped = re.sub(r'^[\w\s]+:\s*', '', stripped, count=1)

        # Now split on common delimiters
        raw_skills = re.split(r'[,|\n•\-;/]+', stripped)

        for skill in raw_skills:
            cleaned = skill.strip()
            # Keep alphanumeric + common tech chars: #, +, ., space
            cleaned = re.sub(r'[^\w\s\+\#\.\(\)]', '', cleaned)
            cleaned = cleaned.strip()

            # Keep if: non-empty, reasonable length, not a pure number
            if cleaned and 1 < len(cleaned) < 40 and not cleaned.isdigit():
                skills.append(cleaned)

    # Deduplicate while preserving order
    skills = list(dict.fromkeys(skills))
    return skills


def parse_experience_section(text: str) -> List[Dict]:
    """
    Parse the experience section into structured job entries.

    Each job entry is detected by looking for lines that contain
    typical job header patterns — a company name or role title
    usually followed by a date range.

    Typical formats:
        "Software Engineer — Google, Pune (Jan 2023 – Jun 2024)"
        "Intern | TCS | 2024"
        "Developer at Infosys, January 2023 to December 2023"

    Args:
        text : raw experience section text

    Returns:
        List of dicts, each with:
            role       : job title / role name
            company    : company name (from NER or heuristic)
            duration   : date range string
            bullets    : list of bullet point strings
            entities   : raw NER output for this entry
    """
    entries = []
    lines   = text.splitlines()

    current_entry  = None
    current_bullets = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Heuristic: job header lines usually contain a year (4 digits)
        # AND are not too long (not a full bullet sentence)
        has_year      = bool(re.search(r'\b(19|20)\d{2}\b', stripped))
        is_short_line = len(stripped.split()) <= 12
        is_bullet     = stripped.startswith(("-", "•", "*", "·"))

        if has_year and is_short_line and not is_bullet:
            # Save previous entry before starting new one
            if current_entry:
                current_entry["bullets"] = current_bullets
                entries.append(current_entry)

            # Start new entry — run NER to extract company and dates
            ents = extract_entities(stripped)
            orgs  = ents.get("ORG",  [])
            dates = ents.get("DATE", [])

            current_entry = {
                "role"    : stripped,               # full line as role for now
                "company" : orgs[0] if orgs else "", # first ORG entity
                "duration": ", ".join(dates),        # join date entities
                "bullets" : [],
                "entities": ents,
            }
            current_bullets = []

        elif is_bullet and current_entry:
            # Bullet point — clean and add to current entry
            bullet_text = re.sub(r'^[-•*·]\s*', '', stripped)
            current_bullets.append(bullet_text)

        elif current_entry and not has_year:
            # Non-bullet, non-header body line — could be continuation
            current_bullets.append(stripped)

    # Don't forget the last entry
    if current_entry:
        current_entry["bullets"] = current_bullets
        entries.append(current_entry)

    return entries


def parse_education_section(text: str) -> List[Dict]:
    """
    Parse the education section into structured degree entries.

    Args:
        text : raw education section text

    Returns:
        List of dicts, each with:
            institution : university/college name
            degree      : degree title
            year        : graduation year
            gpa         : GPA/CGPA string if present
            entities    : raw NER output
    """
    entries = []
    lines   = text.splitlines()
    current = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        ents   = extract_entities(stripped)
        orgs   = ents.get("ORG",  [])
        dates  = ents.get("DATE", [])

        # GPA pattern: "8.4", "3.9/4.0", "CGPA: 8.4", "GPA 3.8"
        gpa_match = re.search(
            r'(?:cgpa|gpa|grade)[:\s]*(\d+\.?\d*)', stripped, re.IGNORECASE
        )
        gpa = gpa_match.group(1) if gpa_match else ""

        # Degree keywords signal a new education entry
        degree_keywords = {
            "b.tech", "btech", "b.e", "be", "m.tech", "mtech",
            "bsc", "b.sc", "msc", "m.sc", "mba", "phd", "ph.d",
            "bachelor", "master", "diploma", "degree",
            "b.com", "bca", "mca",
        }
        lower = stripped.lower()
        has_degree = any(kw in lower for kw in degree_keywords)
        has_year   = bool(re.search(r'\b(19|20)\d{2}\b', stripped))

        if has_degree or (orgs and has_year):
            if current:
                entries.append(current)

            current = {
                "institution": orgs[0] if orgs else "",
                "degree"     : stripped,
                "year"       : dates[0] if dates else "",
                "gpa"        : gpa,
                "entities"   : ents,
            }
        elif current and (gpa and not current["gpa"]):
            current["gpa"] = gpa   # update GPA if found on a following line

    if current:
        entries.append(current)

    return entries


def parse_projects_section(text: str) -> List[Dict]:
    """
    Parse the projects section into structured project entries.

    Detects project name lines (short, Title Case or followed by dash/colon)
    and collects following bullets as description.

    Args:
        text : raw projects section text

    Returns:
        List of dicts, each with:
            name        : project name
            tech_stack  : technologies mentioned (from NER + skill matching)
            description : bullet points describing the project
    """
    from utils.keyword_extractor import extract_skills_from_text

    entries = []
    lines   = text.splitlines()

    current_project = None
    current_bullets = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        is_bullet = stripped.startswith(("-", "•", "*", "·"))

        # Project name heuristics:
        #   - short line (≤ 8 words)
        #   - not a bullet
        #   - contains capital letters (proper noun / title)
        #   - often has "—", ":", "|" separating name from tech stack
        is_separator  = any(sep in stripped for sep in ["—", "–", ":", "|"])
        is_short      = len(stripped.split()) <= 10
        has_caps      = bool(re.search(r'[A-Z]', stripped))
        looks_like_header = is_short and has_caps and not is_bullet

        if looks_like_header and not is_bullet:
            # Save previous project
            if current_project:
                current_project["description"] = current_bullets
                entries.append(current_project)

            # Extract tech stack from the project header line itself
            tech = list(extract_skills_from_text(stripped))

            current_project = {
                "name"       : stripped,
                "tech_stack" : tech,
                "description": [],
            }
            current_bullets = []

        elif is_bullet and current_project:
            bullet_text = re.sub(r'^[-•*·]\s*', '', stripped)
            current_bullets.append(bullet_text)

            # Also mine tech from bullet points
            more_tech = extract_skills_from_text(bullet_text)
            for t in more_tech:
                if t not in current_project["tech_stack"]:
                    current_project["tech_stack"].append(t)

    if current_project:
        current_project["description"] = current_bullets
        entries.append(current_project)

    return entries


# ════════════════════════════════════════════════════════════════════
# SECTION 6 : MASTER FUNCTION
# ════════════════════════════════════════════════════════════════════

def analyze_resume_structure(text: str) -> dict:
    """
    Master function: runs complete structural analysis on a resume.
    This is the ONLY function app.py imports from section_detector.py.

    Pipeline:
        1. Split into sections (split_into_sections)
        2. Extract candidate name and contact info
        3. Parse each section with its specific parser
        4. Run NER on each section
        5. Compute a completeness score (how many sections are present)

    Args:
        text : full raw resume text from extractor.py

    Returns:
        Dict with all structural analysis results
    """
    # ── Step 1: Split into sections ───────────────────────────────
    sections = split_into_sections(text)

    # ── Step 2: Name and contact ───────────────────────────────────
    header_text  = sections.get("header", "")
    name         = extract_candidate_name(header_text)
    contact_info = extract_contact_info(text)   # search full text for contact

    # ── Step 3: Parse each section ─────────────────────────────────
    skills_raw    = sections.get("skills", "")
    experience_raw = sections.get("experience", "")
    education_raw  = sections.get("education", "")
    projects_raw   = sections.get("projects", "")

    parsed_skills     = parse_skills_section(skills_raw)      if skills_raw     else []
    parsed_experience = parse_experience_section(experience_raw) if experience_raw else []
    parsed_education  = parse_education_section(education_raw)   if education_raw  else []
    parsed_projects   = parse_projects_section(projects_raw)     if projects_raw   else []

    # ── Step 4: NER on each section ────────────────────────────────
    section_entities = {}
    for section_name, section_text in sections.items():
        if section_text.strip():
            section_entities[section_name] = extract_entities(section_text)

    # ── Step 5: Completeness score ─────────────────────────────────
    # Award points for each important section present
    # This tells the user which sections their resume is missing
    important_sections = {
        "skills"          : 25,   # 25 points — most important for ATS
        "experience"      : 30,   # 30 points — most important for recruiters
        "education"       : 20,   # 20 points — baseline requirement
        "projects"        : 15,   # 15 points — important for freshers
        "summary"         : 5,    # 5 points  — nice to have
        "certifications"  : 5,    # 5 points  — bonus
    }

    completeness_score = 0
    section_presence   = {}

    for sec, points in important_sections.items():
        present = sec in sections and bool(sections[sec].strip())
        section_presence[sec] = present
        if present:
            completeness_score += points

    # ── Assemble result ────────────────────────────────────────────
    return {
        # Identity
        "candidate_name"   : name,
        "contact_info"     : contact_info,

        # Raw section texts
        "sections"         : sections,

        # Parsed structured data
        "skills"           : parsed_skills,
        "experience"       : parsed_experience,
        "education"        : parsed_education,
        "projects"         : parsed_projects,

        # NER results per section
        "section_entities" : section_entities,

        # Completeness
        "completeness_score" : completeness_score,
        "section_presence"   : section_presence,

        # Quick stats
        "section_count"    : len(sections),
        "skills_count"     : len(parsed_skills),
        "experience_count" : len(parsed_experience),
        "education_count"  : len(parsed_education),
        "projects_count"   : len(parsed_projects),
    }