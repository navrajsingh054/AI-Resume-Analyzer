# utils/ai_feedback.py
# ─────────────────────────────────────────────────────────────────────
# PURPOSE : Call Google Gemini API to generate AI-powered resume
#           feedback based on resume content, ATS score, and skill gaps.
#
# KEY CONCEPTS:
#   - Prompt Engineering : structuring the input to get reliable output
#   - API error handling : retries, rate limits, empty responses
#   - Response parsing   : extracting structured data from AI text
#
# MODEL USED: gemini-1.5-flash
#   - Free tier: 15 requests/minute, 1 million tokens/day
#   - Fast response: ~2-3 seconds per call
#   - Good at following structured instructions
#
# USED BY : app.py (Day 8)
# ─────────────────────────────────────────────────────────────────────

import os
import re
import time
from typing import Dict, List, Optional

import google.generativeai as genai     # Google Gemini SDK
from dotenv import load_dotenv          # loads .env file into os.environ


# ── Load environment variables from .env file ─────────────────────────
# load_dotenv() reads .env and sets each line as an environment variable
# Must be called BEFORE os.getenv() or the key won't be found
load_dotenv()


# ── Configure Gemini SDK ──────────────────────────────────────────────
# os.getenv() reads the key from environment (set by load_dotenv above)
# If key is missing, we set it to empty string — the API call will fail
# gracefully with a clear error message rather than a cryptic crash
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    # genai.configure sets the key globally for all subsequent API calls
    # You only need to call this once per session


# ── Model configuration ───────────────────────────────────────────────
# GenerativeModel wraps the API with model-specific settings
# "gemini-1.5-flash" is the free-tier fast model
# "gemini-1.5-pro" is slower but more capable (same free tier limits)
MODEL_NAME = "gemini-2.0-flash"

# generation_config controls how the model generates text
GENERATION_CONFIG = {
    "temperature"     : 0.7,   # 0.0 = deterministic, 1.0 = creative
                                # 0.7 = balanced: consistent but not robotic
    "top_p"           : 0.85,  # nucleus sampling: consider tokens making up
                                # top 85% of probability mass → focused output
    "top_k"           : 40,    # consider only top 40 tokens at each step
    "max_output_tokens": 2048, # max ~1500 words of output per call
}


# ════════════════════════════════════════════════════════════════════
# SECTION 1 : PROMPT BUILDERS
# ════════════════════════════════════════════════════════════════════
# Prompt engineering is the skill of writing instructions that reliably
# produce structured, useful output from an LLM.
#
# Our strategy:
#   1. Give the model a ROLE ("You are an expert ATS consultant")
#   2. Give it CONTEXT (resume text, scores, missing skills)
#   3. Give it STRICT OUTPUT FORMAT (numbered sections with labels)
#   4. Give it CONSTRAINTS (be specific, no generic advice)
# ════════════════════════════════════════════════════════════════════

def build_full_feedback_prompt(
    resume_text      : str,
    job_description  : str,
    ats_score        : float,
    matched_skills   : List[str],
    missing_skills   : List[str],
    section_scores   : Dict[str, float],
    candidate_name   : str = "the candidate",
) -> str:
    """
    Build the main feedback prompt.

    This is the most important function in this file.
    The quality of Gemini's output depends entirely on prompt quality.

    Prompt structure:
        [ROLE ASSIGNMENT]
        [CONTEXT BLOCK]
        [ANALYSIS DATA]
        [OUTPUT FORMAT INSTRUCTIONS]
        [CONSTRAINTS]

    Args:
        resume_text     : full raw resume text
        job_description : full raw job description text
        ats_score       : final weighted ATS score (0-100)
        matched_skills  : list of skills found in both resume and JD
        missing_skills  : list of skills in JD but not in resume
        section_scores  : {"skills": 72.0, "experience": 45.0, ...}
        candidate_name  : extracted candidate name for personalisation

    Returns:
        Complete prompt string ready to send to Gemini
    """

    # Format lists for readable prompt injection
    # [:15] limits to 15 items — prompt tokens cost money and time
    matched_str = ", ".join(matched_skills[:15]) if matched_skills else "none detected"
    missing_str = ", ".join(missing_skills[:15]) if missing_skills else "none detected"

    # Format section scores as readable lines
    section_str = "\n".join(
        f"  - {section.capitalize()}: {score:.1f}%"
        for section, score in section_scores.items()
    )

    # Truncate resume and JD to avoid exceeding context window
    # 3000 chars ≈ 750 tokens — well within gemini-1.5-flash's 1M token limit
    # but keeps cost and latency low
    resume_snippet = resume_text[:3000]
    jd_snippet     = job_description[:2000]

    prompt = f"""You are an expert ATS (Applicant Tracking System) consultant and
professional resume coach with 10 years of experience helping software
engineering candidates land jobs at top tech companies.

Analyse the following resume against the job description and provide
specific, actionable feedback. Be direct and concrete — avoid generic
advice like "improve your skills". Every suggestion must reference
specific content from the resume or job description.

════════════════════════════════════════
CANDIDATE: {candidate_name}
ATS SCORE: {ats_score:.1f}/100
════════════════════════════════════════

RESUME TEXT:
{resume_snippet}

JOB DESCRIPTION:
{jd_snippet}

════════════════════════════════════════
ANALYSIS DATA (already computed):
Matched skills  : {matched_str}
Missing skills  : {missing_str}
Section scores  :
{section_str}
════════════════════════════════════════

Provide feedback in EXACTLY this format with these EXACT section headers:

## OVERALL ASSESSMENT
Write 2-3 sentences summarising the resume's strengths and biggest
weakness relative to this specific job. Mention the ATS score and
what is driving it up or down.

## SKILLS FEEDBACK
List exactly 3 specific observations about the skills section.
Format each as: "• [observation]: [specific recommendation]"
Reference actual skills from the resume and job description.

## EXPERIENCE FEEDBACK
List exactly 3 specific observations about the experience section.
Format each as: "• [observation]: [specific recommendation]"
Mention specific bullet points or roles from the resume.

## TOP 5 ACTION ITEMS
List exactly 5 things the candidate should do TODAY to improve this
resume for this job. Number them 1-5. Be extremely specific.
Example: "1. Add 'Kubernetes' to your skills section — it appears
3 times in the job description and is completely absent from your resume."

## REWRITTEN SUMMARY
Write a 3-sentence professional summary for this candidate tailored
to this specific job. Use keywords from the job description naturally.
Make it sound human, not like an AI wrote it.

## MISSING KEYWORDS
List the 8 most important keywords from the job description that are
missing from the resume. Format as a comma-separated list.

IMPORTANT RULES:
- Every bullet point must reference something SPECIFIC from the resume
- Do not give advice that doesn't apply to this specific resume
- Do not use phrases like "consider adding" — be direct: "Add X"
- The rewritten summary must sound like a real person wrote it
- Keep total response under 600 words
"""
    return prompt


def build_quick_tips_prompt(
    resume_text  : str,
    ats_score    : float,
    missing_skills: List[str],
) -> str:
    """
    Shorter prompt for a quick 3-tip response.
    Used when the user hasn't provided a job description.

    Returns a simpler prompt asking for just 3 quick improvements.
    """
    missing_str = ", ".join(missing_skills[:10]) if missing_skills else "none detected"

    prompt = f"""You are a professional resume coach.

Review this resume and give exactly 3 quick, specific improvements
the candidate can make TODAY. The current ATS score is {ats_score:.1f}/100.

Skills that could be added: {missing_str}

RESUME:
{resume_text[:2000]}

Respond in EXACTLY this format:

## QUICK TIPS

**Tip 1:** [specific action]
[1-2 sentences explaining why]

**Tip 2:** [specific action]
[1-2 sentences explaining why]

**Tip 3:** [specific action]
[1-2 sentences explaining why]

Keep total response under 200 words. Be direct and specific.
"""
    return prompt


# ════════════════════════════════════════════════════════════════════
# SECTION 2 : API CALLER WITH ERROR HANDLING
# ════════════════════════════════════════════════════════════════════

def call_gemini(prompt: str, retries: int = 3) -> str:
    """
    Send a prompt to Gemini and return the response text.

    Includes retry logic for transient errors (network issues,
    rate limits) and clear error messages for permanent failures.

    Args:
        prompt  : the complete prompt string
        retries : number of retry attempts on failure (default 3)

    Returns:
        Response text string from Gemini.
        Returns an error message string if all retries fail —
        so the app can display the error instead of crashing.
    """

    # ── Check API key first ────────────────────────────────────────
    if not GEMINI_API_KEY:
        return (
            "ERROR: GEMINI_API_KEY not found.\n"
            "1. Get a free key at https://aistudio.google.com/app/apikey\n"
            "2. Add it to your .env file: GEMINI_API_KEY=your_key_here\n"
            "3. Restart the app"
        )

    # ── Initialize model ───────────────────────────────────────────
    # We create a new model instance per call so generation_config
    # can theoretically vary per call in future
    model = genai.GenerativeModel(
        model_name        = MODEL_NAME,
        generation_config = GENERATION_CONFIG,
    )

    last_error = ""

    for attempt in range(retries):
        try:
            # model.generate_content() sends the prompt and waits for response
            # This is a synchronous (blocking) call — execution pauses here
            # until Gemini responds (usually 2-5 seconds)
            response = model.generate_content(prompt)

            # response.text contains the generated string
            # If the model refused to answer (safety filters),
            # response.text might be empty
            if response.text and response.text.strip():
                return response.text.strip()
            else:
                last_error = "Empty response from Gemini (possible safety filter)"

        except Exception as e:
            last_error = str(e)
            error_lower = last_error.lower()

            # Rate limit error → wait longer before retry
            # Gemini free tier: 15 requests per minute
            if "quota" in error_lower or "rate" in error_lower or "429" in error_lower:
                wait_time = (attempt + 1) * 10   # 10s, 20s, 30s
                print(f"    Rate limit hit. Waiting {wait_time}s before retry...")
                time.sleep(wait_time)

            # Transient server error → short wait
            elif "500" in error_lower or "503" in error_lower or "unavailable" in error_lower:
                wait_time = (attempt + 1) * 3    # 3s, 6s, 9s
                print(f"    Server error. Retrying in {wait_time}s...")
                time.sleep(wait_time)

            # Authentication error → don't retry (key is wrong)
            elif "api_key" in error_lower or "invalid" in error_lower or "403" in error_lower:
                return (
                    f"ERROR: Invalid API key.\n"
                    f"Check your GEMINI_API_KEY in .env file.\n"
                    f"Details: {last_error}"
                )

            # Unknown error → short wait and retry
            else:
                if attempt < retries - 1:
                    time.sleep(2)

    # All retries exhausted
    return f"ERROR: Gemini API failed after {retries} attempts.\nLast error: {last_error}"


# ════════════════════════════════════════════════════════════════════
# SECTION 3 : RESPONSE PARSER
# ════════════════════════════════════════════════════════════════════

def parse_feedback_response(raw_response: str) -> Dict[str, str]:
    """
    Parse Gemini's raw text response into a structured dict.

    Strategy:
        Find each ## header by scanning line by line.
        Collect all lines after a header until the next header.
        This is more robust than re.split which breaks on
        headers containing numbers or special characters.

    Args:
        raw_response : raw text string from Gemini

    Returns:
        Dict with keys:
            overall_assessment, skills_feedback, experience_feedback,
            action_items, rewritten_summary, missing_keywords, raw
    """
    result = {
        "overall_assessment" : "Assessment not available.",
        "skills_feedback"    : "Skills feedback not available.",
        "experience_feedback": "Experience feedback not available.",
        "action_items"       : "Action items not available.",
        "rewritten_summary"  : "Summary rewrite not available.",
        "missing_keywords"   : "",
        "raw"                : raw_response,
    }

    if not raw_response or raw_response.startswith("ERROR:"):
        result["overall_assessment"] = raw_response
        return result

    # ── Scan line by line ──────────────────────────────────────────
    # Each time we see a line starting with ##, we know which section
    # starts. We collect all following lines until the next ## line.
    lines           = raw_response.splitlines()
    current_section = None   # which result key we are filling
    buffer          = []     # lines collected for current section

    def flush_buffer(section_key, buf):
        """Join buffer lines and store in result if non-empty."""
        content = "\n".join(buf).strip()
        if content and section_key:
            result[section_key] = content

    for line in lines:
        stripped = line.strip()

        # Check if this line is a ## header
        if stripped.startswith("##"):
            # Save whatever we collected for the previous section
            flush_buffer(current_section, buffer)
            buffer = []   # reset buffer for new section

            # Normalise header for matching:
            # "## TOP 5 ACTION ITEMS" → "top 5 action items"
            header_text = stripped.lstrip("#").strip().lower()

            # Map header text to result dict key
            if "overall" in header_text or "assessment" in header_text:
                current_section = "overall_assessment"

            elif "skill" in header_text:
                current_section = "skills_feedback"

            elif "experience" in header_text:
                current_section = "experience_feedback"

            elif "action" in header_text:
                current_section = "action_items"

            elif "summary" in header_text or "rewritten" in header_text:
                current_section = "rewritten_summary"

            elif "missing" in header_text or "keyword" in header_text:
                current_section = "missing_keywords"

            elif "quick" in header_text or "tip" in header_text:
                # For the quick tips prompt format
                current_section = "action_items"

            else:
                # Unknown header — don't lose the content, put in raw
                current_section = None

        else:
            # Body line — add to buffer for current section
            if current_section and stripped:
                buffer.append(stripped)

    # Don't forget the last section after the loop ends
    flush_buffer(current_section, buffer)

    return result

def parse_action_items(action_items_text: str) -> List[str]:
    """
    Convert the action items text block into a clean list of strings.

    Handles formats:
        "1. Add kubernetes to skills\n2. Rewrite experience bullets..."
        "1) Add kubernetes\n2) Rewrite..."

    Args:
        action_items_text : the action_items value from parse_feedback_response

    Returns:
        List of individual action item strings (without numbering)
    """
    if not action_items_text or action_items_text == "Action items not available.":
        return []

    items = []

    for line in action_items_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        # Remove leading numbering: "1.", "1)", "1 -" etc.
        cleaned = re.sub(r'^\d+[\.\)]\s*', '', stripped)
        cleaned = re.sub(r'^[-•]\s*', '', cleaned)

        if len(cleaned) > 10:    # filter out very short lines (noise)
            items.append(cleaned)

    return items[:5]             # return max 5 items


# ════════════════════════════════════════════════════════════════════
# SECTION 4 : MASTER FUNCTION
# ════════════════════════════════════════════════════════════════════

def get_ai_feedback(
    resume_text      : str,
    job_description  : str  = "",
    ats_score        : float = 0.0,
    matched_skills   : List[str] = None,
    missing_skills   : List[str] = None,
    section_scores   : Dict[str, float] = None,
    candidate_name   : str = "the candidate",
) -> Dict:
    """
    Master function: builds prompt, calls Gemini, parses response.
    This is the ONLY function app.py imports from ai_feedback.py.

    Args:
        resume_text     : raw resume text
        job_description : raw job description (optional but recommended)
        ats_score       : ATS score from scorer.py
        matched_skills  : list of matched skills from keyword_extractor.py
        missing_skills  : list of missing skills
        section_scores  : section score dict from scorer.py
        candidate_name  : from section_detector.py

    Returns:
        Dict with all feedback fields plus metadata:
            overall_assessment  : 2-3 sentence summary
            skills_feedback     : 3 bullet observations
            experience_feedback : 3 bullet observations
            action_items        : text block of 5 numbered items
            action_items_list   : parsed list of 5 action strings
            rewritten_summary   : AI-rewritten professional summary
            missing_keywords    : comma-separated missing keyword string
            raw                 : full raw Gemini response
            success             : True if API call succeeded
            error               : error message if success=False
    """
    # Safe defaults for optional parameters
    matched_skills = matched_skills or []
    missing_skills = missing_skills or []
    section_scores = section_scores or {}

    # ── Choose prompt based on whether JD was provided ────────────
    if job_description.strip():
        prompt = build_full_feedback_prompt(
            resume_text     = resume_text,
            job_description = job_description,
            ats_score       = ats_score,
            matched_skills  = matched_skills,
            missing_skills  = missing_skills,
            section_scores  = section_scores,
            candidate_name  = candidate_name,
        )
    else:
        # No JD provided — use shorter quick tips prompt
        prompt = build_quick_tips_prompt(
            resume_text   = resume_text,
            ats_score     = ats_score,
            missing_skills= missing_skills,
        )

    # ── Call Gemini ────────────────────────────────────────────────
    print(f"    Calling Gemini API ({MODEL_NAME})...")
    raw_response = call_gemini(prompt)

    # ── Check for errors ──────────────────────────────────────────
    success = not raw_response.startswith("ERROR:")

    # ── Parse response ────────────────────────────────────────────
    parsed = parse_feedback_response(raw_response)

    # ── Add parsed action items list ──────────────────────────────
    parsed["action_items_list"] = parse_action_items(
        parsed.get("action_items", "")
    )

    # ── Add metadata ──────────────────────────────────────────────
    parsed["success"] = success
    parsed["error"]   = "" if success else raw_response
    parsed["model"]   = MODEL_NAME

    return parsed