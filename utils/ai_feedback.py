# utils/ai_feedback.py
import os
import time
from typing import Dict, List
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL_NAME   = "llama3-8b-8192"   # free, fast, no quota issues


def build_full_feedback_prompt(
    resume_text     : str,
    job_description : str,
    ats_score       : float,
    matched_skills  : List[str],
    missing_skills  : List[str],
    section_scores  : Dict[str, float],
    candidate_name  : str = "the candidate",
) -> str:
    matched_str  = ", ".join(matched_skills[:15]) if matched_skills else "none"
    missing_str  = ", ".join(missing_skills[:15]) if missing_skills else "none"
    section_str  = "\n".join(
        f"  - {k.capitalize()}: {v:.1f}%"
        for k, v in section_scores.items()
    )
    resume_snippet = resume_text[:3000]
    jd_snippet     = job_description[:2000]

    return f"""You are an expert ATS consultant and resume coach.
Analyse this resume against the job description and give specific feedback.

CANDIDATE: {candidate_name}
ATS SCORE: {ats_score:.1f}/100
Matched skills : {matched_str}
Missing skills : {missing_str}
Section scores :
{section_str}

RESUME:
{resume_snippet}

JOB DESCRIPTION:
{jd_snippet}

Respond in EXACTLY this format:

## OVERALL ASSESSMENT
2-3 sentences about strengths and biggest weakness.

## SKILLS FEEDBACK
- [observation]: [specific recommendation]
- [observation]: [specific recommendation]
- [observation]: [specific recommendation]

## EXPERIENCE FEEDBACK
- [observation]: [specific recommendation]
- [observation]: [specific recommendation]
- [observation]: [specific recommendation]

## TOP 5 ACTION ITEMS
1. [specific action]
2. [specific action]
3. [specific action]
4. [specific action]
5. [specific action]

## REWRITTEN SUMMARY
3-sentence professional summary tailored to this job.

## MISSING KEYWORDS
keyword1, keyword2, keyword3, keyword4, keyword5, keyword6, keyword7, keyword8

Keep total response under 600 words. Be specific, not generic.
"""


def build_quick_tips_prompt(
    resume_text   : str,
    ats_score     : float,
    missing_skills: List[str],
) -> str:
    missing_str = ", ".join(missing_skills[:10]) if missing_skills else "none"
    return f"""You are a resume coach. Give exactly 3 specific improvements.
ATS score: {ats_score:.1f}/100. Skills to add: {missing_str}

RESUME:
{resume_text[:2000]}

## QUICK TIPS

**Tip 1:** [action]
[1-2 sentences why]

**Tip 2:** [action]
[1-2 sentences why]

**Tip 3:** [action]
[1-2 sentences why]

Under 200 words total.
"""


def call_groq(prompt: str, retries: int = 3) -> str:
    if not GROQ_API_KEY:
        return (
            "ERROR: GROQ_API_KEY not found.\n"
            "1. Get free key at https://console.groq.com\n"
            "2. Add to .env: GROQ_API_KEY=your_key\n"
            "3. Add to Streamlit secrets: GROQ_API_KEY = \"your_key\""
        )

    client = Groq(api_key=GROQ_API_KEY)

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model    = MODEL_NAME,
                messages = [{"role": "user", "content": prompt}],
                max_tokens  = 2048,
                temperature = 0.7,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            error = str(e).lower()
            if "rate" in error or "429" in error:
                time.sleep((attempt + 1) * 5)
            elif attempt < retries - 1:
                time.sleep(2)
            else:
                return f"ERROR: Groq API failed. {str(e)}"

    return "ERROR: All retries failed."


def parse_feedback_response(raw_response: str) -> Dict[str, str]:
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

    lines           = raw_response.splitlines()
    current_section = None
    buffer          = []

    def flush(section_key, buf):
        content = "\n".join(buf).strip()
        if content and section_key:
            result[section_key] = content

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("##"):
            flush(current_section, buffer)
            buffer = []
            header = stripped.lstrip("#").strip().lower()
            if "overall" in header or "assessment" in header:
                current_section = "overall_assessment"
            elif "skill" in header:
                current_section = "skills_feedback"
            elif "experience" in header:
                current_section = "experience_feedback"
            elif "action" in header or "tip" in header:
                current_section = "action_items"
            elif "summary" in header or "rewritten" in header:
                current_section = "rewritten_summary"
            elif "missing" in header or "keyword" in header:
                current_section = "missing_keywords"
            else:
                current_section = None
        elif current_section and stripped:
            buffer.append(stripped)

    flush(current_section, buffer)
    return result


def parse_action_items(action_items_text: str) -> List[str]:
    import re
    if not action_items_text or action_items_text == "Action items not available.":
        return []
    items = []
    for line in action_items_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        cleaned = re.sub(r'^\d+[\.\)]\s*', '', stripped)
        cleaned = re.sub(r'^[-•]\s*', '', cleaned)
        if len(cleaned) > 10:
            items.append(cleaned)
    return items[:5]


def get_ai_feedback(
    resume_text    : str,
    job_description: str = "",
    ats_score      : float = 0.0,
    matched_skills : List[str] = None,
    missing_skills : List[str] = None,
    section_scores : Dict[str, float] = None,
    candidate_name : str = "the candidate",
) -> Dict:
    matched_skills = matched_skills or []
    missing_skills = missing_skills or []
    section_scores = section_scores or {}

    if job_description.strip():
        prompt = build_full_feedback_prompt(
            resume_text, job_description, ats_score,
            matched_skills, missing_skills, section_scores, candidate_name,
        )
    else:
        prompt = build_quick_tips_prompt(
            resume_text, ats_score, missing_skills
        )

    print(f"    Calling Groq API ({MODEL_NAME})...")
    raw_response = call_groq(prompt)
    success      = not raw_response.startswith("ERROR:")
    parsed       = parse_feedback_response(raw_response)
    parsed["action_items_list"] = parse_action_items(parsed.get("action_items", ""))
    parsed["success"] = success
    parsed["error"]   = "" if success else raw_response
    parsed["model"]   = MODEL_NAME

    return parsed