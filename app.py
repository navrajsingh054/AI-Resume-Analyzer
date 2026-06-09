# app.py
# ─────────────────────────────────────────────────────────────────────
# PURPOSE : Main Streamlit web application.
#           Wires all modules together into a single UI.
#
# RUN WITH : streamlit run app.py
# URL      : http://localhost:8501
#
# LAYOUT:
#   Sidebar  → file upload + job description input + settings
#   Main     → tabbed results: Score | Skills | Structure | AI Feedback
# ─────────────────────────────────────────────────────────────────────

import os
import tempfile
import streamlit as st

# ── Our modules ────────────────────────────────────────────────────────
from utils.extractor        import extract_text
from utils.scorer           import get_full_score_report
from utils.keyword_extractor import get_analysis_summary
from utils.section_detector  import analyze_resume_structure
from utils.ai_feedback       import get_ai_feedback


# ════════════════════════════════════════════════════════════════════
# SECTION 1 : PAGE CONFIGURATION
# Must be the FIRST Streamlit call in the file — before any other st.*
# ════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title = "AI Resume Analyzer",
    page_icon  = "📄",
    layout     = "wide",        # use full browser width
    initial_sidebar_state = "expanded",
)


# ════════════════════════════════════════════════════════════════════
# SECTION 2 : CUSTOM CSS
# Streamlit's default styling is functional but plain.
# We inject a small CSS block to improve visual polish.
# ════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
/* Score circle */
.score-circle {
    width: 140px; height: 140px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    flex-direction: column;
    margin: 0 auto;
    font-weight: bold;
}
.score-excellent { background: #d4edda; border: 4px solid #28a745; color: #155724; }
.score-good      { background: #cce5ff; border: 4px solid #004085; color: #004085; }
.score-moderate  { background: #fff3cd; border: 4px solid #ff8c00; color: #856404; }
.score-weak      { background: #f8d7da; border: 4px solid #dc3545; color: #721c24; }

/* Skill badges */
.skill-badge-found   { background:#d4edda; color:#155724; padding:3px 10px;
                        border-radius:12px; margin:2px; display:inline-block;
                        font-size:13px; }
.skill-badge-missing { background:#f8d7da; color:#721c24; padding:3px 10px;
                        border-radius:12px; margin:2px; display:inline-block;
                        font-size:13px; }

/* Section cards */
.info-card {
    background: #f8f9fa;
    border-left: 4px solid #007bff;
    padding: 12px 16px;
    margin: 8px 0;
    border-radius: 0 8px 8px 0;
}

/* Action item cards */
.action-card {
    background: #fff8e1;
    border-left: 4px solid #ffc107;
    padding: 10px 14px;
    margin: 6px 0;
    border-radius: 0 8px 8px 0;
}
</style>
""", unsafe_allow_html=True)
# unsafe_allow_html=True needed to inject raw HTML/CSS into Streamlit


# ════════════════════════════════════════════════════════════════════
# SECTION 3 : CACHED ANALYSIS FUNCTIONS
# @st.cache_data caches the return value keyed by input arguments.
# Same input → returns cached result instantly without re-running.
# ════════════════════════════════════════════════════════════════════

@st.cache_data
def run_extraction(file_bytes: bytes, file_ext: str) -> str:
    """Extract text from uploaded file bytes with full error handling."""
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=file_ext
    ) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        text = extract_text(tmp_path)
        if not text or not text.strip():
            raise ValueError(
                "No text could be extracted from this file.\n"
                "Possible reasons:\n"
                "• The PDF is scanned (image-based) — try a text-based PDF\n"
                "• The file is corrupted or password-protected\n"
                "• The DOCX file has no text content"
            )
        return text
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Extraction failed: {str(e)}")
    finally:
        os.unlink(tmp_path)


@st.cache_data
def run_scoring(resume_text: str, job_text: str) -> dict:
    """Cached wrapper for Day 5 scorer."""
    return get_full_score_report(resume_text, job_text)


@st.cache_data
def run_skill_analysis(resume_text: str, job_text: str) -> dict:
    """Cached wrapper for Day 4 keyword extractor."""
    return get_analysis_summary(resume_text, job_text)


@st.cache_data
def run_structure_analysis(resume_text: str) -> dict:
    """Cached wrapper for Day 6 section detector."""
    return analyze_resume_structure(resume_text)


@st.cache_data
def run_ai_feedback(
    resume_text    : str,
    job_text       : str,
    ats_score      : float,
    matched_skills : tuple,   # tuple not list — lists aren't hashable for cache
    missing_skills : tuple,
    section_scores : str,     # JSON string — dicts aren't hashable for cache
    candidate_name : str,
) -> dict:
    """
    Cached wrapper for Day 7 AI feedback.

    Note on cache keys:
        st.cache_data hashes all arguments to create a cache key.
        Lists and dicts are not hashable — they can't be hashed.
        We convert them to tuple and JSON string so they can be cached.
    """
    import json
    scores_dict = json.loads(section_scores)

    return get_ai_feedback(
        resume_text    = resume_text,
        job_description= job_text,
        ats_score      = ats_score,
        matched_skills = list(matched_skills),
        missing_skills = list(missing_skills),
        section_scores = scores_dict,
        candidate_name = candidate_name,
    )


# ════════════════════════════════════════════════════════════════════
# SECTION 4 : UI HELPER FUNCTIONS
# Small functions that render specific UI components.
# Keeping them separate makes the main flow readable.
# ════════════════════════════════════════════════════════════════════

def render_score_circle(score: float, grade: str, label: str) -> None:
    """Render the big score circle in the center of the dashboard."""
    if score >= 80:
        css_class = "score-excellent"
    elif score >= 65:
        css_class = "score-good"
    elif score >= 50:
        css_class = "score-moderate"
    else:
        css_class = "score-weak"

    st.markdown(f"""
    <div class="score-circle {css_class}">
        <div style="font-size:36px">{score:.0f}%</div>
        <div style="font-size:14px">Grade {grade}</div>
    </div>
    <p style="text-align:center; margin-top:8px; color:gray">{label}</p>
    """, unsafe_allow_html=True)


def render_skill_badges(skills: list, badge_class: str) -> None:
    """Render a row of colored skill badges."""
    if not skills:
        st.write("None detected")
        return
    badges = " ".join(
        f'<span class="{badge_class}">{s}</span>'
        for s in sorted(skills)
    )
    st.markdown(badges, unsafe_allow_html=True)


def render_section_card(title: str, content: str) -> None:
    """Render a styled info card with a left blue border."""
    st.markdown(
        f'<div class="info-card"><strong>{title}</strong><br>{content}</div>',
        unsafe_allow_html=True
    )


def render_action_card(number: int, text: str) -> None:
    """Render a single action item with yellow left border."""
    st.markdown(
        f'<div class="action-card"><strong>#{number}</strong> {text}</div>',
        unsafe_allow_html=True
    )


# ════════════════════════════════════════════════════════════════════
# SECTION 5 : SIDEBAR
# ════════════════════════════════════════════════════════════════════

def render_sidebar():
    """
    Render the sidebar with file upload and job description input.

    Returns:
        (uploaded_file, job_description, analyze_clicked)
        uploaded_file  : Streamlit UploadedFile object or None
        job_description: string from text area
        analyze_clicked: True if user clicked the Analyze button
    """
    with st.sidebar:
        st.title("📄 Resume Analyzer")
        st.markdown("---")

        # ── File uploader ──────────────────────────────────────────
        st.subheader("Step 1 — Upload Resume")
        uploaded_file = st.file_uploader(
            label       = "Choose your resume",
            type        = ["pdf", "docx"],   # only allow PDF and DOCX
            help        = "Supported formats: PDF, DOCX (max 10MB)",
        )
        # st.file_uploader returns None if no file, or an UploadedFile object

        if uploaded_file:
            # Show file info
            file_size_kb = len(uploaded_file.getvalue()) / 1024
            st.success(f"✓ {uploaded_file.name} ({file_size_kb:.0f} KB)")

        st.markdown("---")

        # ── Job description ────────────────────────────────────────
        st.subheader("Step 2 — Paste Job Description")
        job_description = st.text_area(
            label       = "Job Description (optional but recommended)",
            height      = 200,
            placeholder = "Paste the full job description here...\n\nExample:\nWe are looking for a Python developer with experience in FastAPI, Docker, AWS...",
            help        = "Providing a job description enables ATS scoring and gap analysis",
        )

        st.markdown("---")

        # ── Settings ───────────────────────────────────────────────
        st.subheader("Step 3 — Settings")
        enable_ai = st.checkbox(
            "Enable AI Feedback (Gemini)",
            value = True,
            help  = "Requires GEMINI_API_KEY in .env file",
        )

        st.markdown("---")

        # ── Analyze button ─────────────────────────────────────────
        # st.button returns True only on the rerun immediately after clicking
        analyze_clicked = st.button(
            "🔍 Analyze Resume",
            type      = "primary",   # filled blue button
            use_container_width = True,
            disabled  = (uploaded_file is None),  # disabled until file uploaded
        )

        # ── About section ──────────────────────────────────────────
        st.markdown("---")
        st.markdown("""
        **About this tool**
        - 📊 ATS score via TF-IDF + Cosine Similarity
        - 🎯 Skill gap analysis (200+ tech skills)
        - 🧠 NLP section detection (spaCy)
        - 🤖 AI feedback (Google Gemini)
        """)

    return uploaded_file, job_description, enable_ai, analyze_clicked


# ════════════════════════════════════════════════════════════════════
# SECTION 6 : TAB RENDERERS
# One function per tab — keeps the main flow clean.
# ════════════════════════════════════════════════════════════════════

def render_score_tab(score_report: dict) -> None:
    """Render the ATS Score tab."""

    # ── Top row: score circle + key metrics ───────────────────────
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.subheader("ATS Score")
        render_score_circle(
            score_report["final_score"],
            score_report["grade"],
            score_report["label"],
        )

    with col2:
        st.subheader("Score Breakdown")
        components = score_report["component_scores"]
        for component, value in components.items():
            if component != "Final Weighted Score":
                # st.metric shows a big number with a label
                st.metric(label=component, value=f"{value}%")

    with col3:
        st.subheader("Advice")
        st.info(score_report["advice"])

        st.markdown("**Keyword Match Rate**")
        match_rate = score_report["keyword_overlap"]["match_rate"]
        # st.progress takes a value 0.0 to 1.0
        st.progress(match_rate / 100)
        st.caption(f"{match_rate}% of job description keywords found in resume")

    st.markdown("---")

    # ── Section scores bar chart ───────────────────────────────────
    st.subheader("Section-Level Scores")
    section_scores = score_report["section_scores"]
    for section, val in section_scores.items():
        st.markdown(f"**{section.capitalize()}**")
        st.progress(val / 100)
        st.caption(f"{val:.1f}%")

    st.markdown("---")

    # ── Keyword overlap detail ─────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("✅ Matched Keywords")
        matched = score_report["keyword_overlap"]["matched_keywords"]
        render_skill_badges(matched[:20], "skill-badge-found")

    with col_b:
        st.subheader("❌ Top Missing Keywords")
        top_missing = [kw for kw, _ in
                       score_report["keyword_overlap"]["top_missing"][:15]]
        render_skill_badges(top_missing, "skill-badge-missing")


def render_skills_tab(skill_summary: dict) -> None:
    """Render the Skill Gap Analysis tab."""

    # ── Overview metrics ──────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Resume Skills",  len(skill_summary["resume_skills"]))
    with col2:
        st.metric("Job Skills",     len(skill_summary["job_skills"]))
    with col3:
        st.metric("Matched",
                  len(skill_summary["gap_analysis"]["matched"]))
    with col4:
        st.metric("Missing",
                  len(skill_summary["gap_analysis"]["missing"]))

    st.markdown("---")

    # ── Domain scores chart ────────────────────────────────────────
    st.subheader("Domain Coverage")
    domain_scores = skill_summary["domain_scores"]
    if domain_scores:
        for domain, val in sorted(domain_scores.items(), key=lambda x: -x[1]):
            if val > 0:
                st.markdown(f"**{domain.capitalize()}**")
                st.progress(val / 100)
                st.caption(f"{val:.1f}%")

    st.markdown("---")

    # ── Missing skills by domain ───────────────────────────────────
    if skill_summary["missing_by_domain"]:
        st.subheader("Missing Skills by Domain")
        for domain_label, missing_list in skill_summary["missing_by_domain"].items():
            with st.expander(f"📌 {domain_label} ({len(missing_list)} missing)"):
                render_skill_badges(missing_list, "skill-badge-missing")

    st.markdown("---")

    # ── Full skills DataFrame ──────────────────────────────────────
    st.subheader("Full Skills Database Match")
    df = skill_summary["skills_df"]
    # Show only found skills by default
    show_all = st.checkbox("Show all skills (including not found)")
    if not show_all:
        df = df[df["found"] == True]

    st.dataframe(
        df[["skill", "domain", "found", "domain_score"]],
        use_container_width = True,
        height = 300,
    )


def render_structure_tab(structure: dict) -> None:
    """Render the Resume Structure tab."""

    # ── Candidate info ─────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Candidate Info")
        render_section_card("Name",  structure["candidate_name"])
        contact = structure["contact_info"]
        if contact["email"]:
            render_section_card("Email", contact["email"])
        if contact["phone"]:
            render_section_card("Phone", contact["phone"])
        if contact["linkedin"]:
            render_section_card("LinkedIn", contact["linkedin"])
        if contact["github"]:
            render_section_card("GitHub", contact["github"])

    with col2:
        st.subheader("Resume Completeness")
        completeness = structure["completeness_score"]
        st.progress(completeness / 100)
        st.metric("Completeness Score", f"{completeness}/100")

        st.markdown("**Section Checklist**")
        for section, present in structure["section_presence"].items():
            icon = "✅" if present else "❌"
            st.markdown(f"{icon} {section.capitalize()}")

    st.markdown("---")

    # ── Parsed sections ────────────────────────────────────────────
    tab_exp, tab_edu, tab_proj, tab_skills = st.tabs([
        "💼 Experience", "🎓 Education", "🚀 Projects", "🛠 Skills"
    ])

    with tab_exp:
        experience = structure["experience"]
        if experience:
            for i, job in enumerate(experience, 1):
                with st.expander(f"Job {i}: {job['role'][:60]}"):
                    if job["company"]:
                        st.markdown(f"**Company:** {job['company']}")
                    if job["duration"]:
                        st.markdown(f"**Duration:** {job['duration']}")
                    if job["bullets"]:
                        st.markdown("**Key contributions:**")
                        for bullet in job["bullets"]:
                            st.markdown(f"• {bullet}")
        else:
            st.info("No experience entries detected")

    with tab_edu:
        education = structure["education"]
        if education:
            for edu in education:
                with st.expander(edu["degree"][:60]):
                    if edu["institution"]:
                        st.markdown(f"**Institution:** {edu['institution']}")
                    if edu["year"]:
                        st.markdown(f"**Year:** {edu['year']}")
                    if edu["gpa"]:
                        st.markdown(f"**GPA:** {edu['gpa']}")
        else:
            st.info("No education entries detected")

    with tab_proj:
        projects = structure["projects"]
        if projects:
            for proj in projects:
                with st.expander(proj["name"][:60]):
                    if proj["tech_stack"]:
                        st.markdown(f"**Tech stack:** {', '.join(proj['tech_stack'])}")
                    if proj["description"]:
                        for bullet in proj["description"]:
                            st.markdown(f"• {bullet}")
        else:
            st.info("No projects detected")

    with tab_skills:
        skills = structure["skills"]
        if skills:
            render_skill_badges(skills, "skill-badge-found")
        else:
            st.info("No skills detected in skills section")


def render_ai_tab(feedback: dict) -> None:
    """Render the AI Feedback tab."""

    if not feedback["success"]:
        st.error(f"AI feedback unavailable: {feedback['error']}")
        st.info("Add your GEMINI_API_KEY to .env to enable AI feedback")
        return

    # ── Overall assessment ─────────────────────────────────────────
    st.subheader("🎯 Overall Assessment")
    st.info(feedback["overall_assessment"])

    st.markdown("---")

    # ── Action items ───────────────────────────────────────────────
    st.subheader("⚡ Top 5 Action Items")
    st.caption("Do these TODAY to improve your resume")

    action_items = feedback["action_items_list"]
    if action_items:
        for i, item in enumerate(action_items, 1):
            render_action_card(i, item)
    else:
        st.markdown(feedback["action_items"])

    st.markdown("---")

    # ── Two columns: skills + experience feedback ──────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🛠 Skills Feedback")
        st.markdown(feedback["skills_feedback"])

    with col2:
        st.subheader("💼 Experience Feedback")
        st.markdown(feedback["experience_feedback"])

    st.markdown("---")

    # ── Rewritten summary ──────────────────────────────────────────
    st.subheader("✍️ AI-Rewritten Professional Summary")
    st.caption("Copy this into your resume's summary section")
    st.success(feedback["rewritten_summary"])

    # Copy button using st.code (easy to select + copy)
    with st.expander("📋 Click to copy summary"):
        st.code(feedback["rewritten_summary"], language=None)

    st.markdown("---")

    # ── Missing keywords ───────────────────────────────────────────
    if feedback["missing_keywords"]:
        st.subheader("🔑 Missing Keywords to Add")
        keywords = [k.strip() for k in
                    feedback["missing_keywords"].split(",") if k.strip()]
        render_skill_badges(keywords, "skill-badge-missing")


# ════════════════════════════════════════════════════════════════════
# SECTION 7 : MAIN APP FLOW
# This is the entry point — Streamlit runs this top to bottom
# on every interaction.
# ════════════════════════════════════════════════════════════════════

def main():
    # ── Render sidebar, get inputs ─────────────────────────────────
    uploaded_file, job_description, enable_ai, analyze_clicked = render_sidebar()

    # ── Landing page (no file uploaded yet) ───────────────────────
    if uploaded_file is None:
        st.title("📄 AI Resume Analyzer")
        st.markdown("""
        ### Welcome! Upload your resume to get started.

        **What this tool does:**
        - 📊 Calculates your **ATS match score** against any job description
        - 🎯 Finds **skill gaps** — exactly what to add to your resume
        - 🧠 Detects **resume structure** — sections, experience, education
        - 🤖 Generates **AI feedback** — specific, actionable improvements

        **How to use:**
        1. Upload your resume PDF or DOCX in the sidebar
        2. Paste the job description you are applying for
        3. Click **Analyze Resume**

        ---
        *Built with Python · spaCy · TF-IDF · Streamlit · Google Gemini*
        """)
        return   # stop here — nothing to analyze yet

    # ── Store analysis results in session_state ────────────────────
    # st.session_state persists values across reruns within the same session
    # This prevents re-running analysis when user switches tabs
    if analyze_clicked:
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()

        # ── Validate file size (max 10MB) ──────────────────────────
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        if file_size_mb > 10:
            st.error(f"File too large ({file_size_mb:.1f}MB). Maximum is 10MB.")
            st.stop()

        # ── Validate file extension ────────────────────────────────
        if file_ext not in [".pdf", ".docx"]:
            st.error(f"Unsupported format '{file_ext}'. Upload PDF or DOCX only.")
            st.stop()
        # ── Extract text ───────────────────────────────────────────
        with st.spinner("📄 Extracting text from resume..."):
            try:
                resume_text = run_extraction(
                    uploaded_file.getvalue(), file_ext
                )
            except ValueError as e:
                st.error(str(e))
                st.stop()
            except Exception as e:
                st.error(f"Unexpected error during extraction: {e}")
                st.stop()

        # ── Validate minimum content ───────────────────────────────
        word_count = len(resume_text.split())
        if word_count < 50:
            st.warning(
                f"Only {word_count} words extracted — this seems too short "
                f"for a resume. Results may be inaccurate."
            )

        # ── Run analysis ───────────────────────────────────────────
        with st.spinner("📊 Calculating ATS score..."):
            try:
                score_report = run_scoring(resume_text, job_description)
            except Exception as e:
                st.error(f"Scoring failed: {e}")
                st.stop()

        with st.spinner("🎯 Analysing skills..."):
            try:
                skill_summary = run_skill_analysis(resume_text, job_description)
            except Exception as e:
                st.error(f"Skill analysis failed: {e}")
                st.stop()

        with st.spinner("🧠 Detecting resume structure..."):
            try:
                structure = run_structure_analysis(resume_text)
            except Exception as e:
                st.error(f"Structure analysis failed: {e}")
                st.stop()

        # ── Store in session_state ─────────────────────────────────
        st.session_state.resume_text   = resume_text
        st.session_state.score_report  = score_report
        st.session_state.skill_summary = skill_summary
        st.session_state.structure     = structure
        st.session_state.analysis_done = True
        st.session_state.enable_ai     = enable_ai
        st.session_state.job_desc      = job_description

        # Clear previous AI feedback when re-analyzing
        if "ai_feedback" in st.session_state:
            del st.session_state["ai_feedback"]
        st.session_state.enable_ai    = enable_ai
        st.session_state.job_desc     = job_description

    # ── Show results if analysis has been done ─────────────────────
    if st.session_state.get("analysis_done"):

        resume_text  = st.session_state.resume_text
        score_report = st.session_state.score_report
        skill_summary= st.session_state.skill_summary
        structure    = st.session_state.structure
        job_desc     = st.session_state.job_desc

        # ── Page header with candidate name ───────────────────────
        name = structure["candidate_name"]
        st.title(f"Analysis Results — {name}")
        st.caption(f"File: {uploaded_file.name}  |  "
                   f"ATS Score: {score_report['final_score']}%  |  "
                   f"Grade: {score_report['grade']}")

        st.markdown("---")

        # ── Four main tabs ─────────────────────────────────────────
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 ATS Score",
            "🎯 Skill Gap",
            "🧠 Structure",
            "🤖 AI Feedback",
        ])

        with tab1:
            render_score_tab(score_report)

        with tab2:
            render_skills_tab(skill_summary)

        with tab3:
            render_structure_tab(structure)

        with tab4:
            # AI feedback is expensive — only run when user opens this tab
            # AND clicks a button (so they opt-in to the API call)
            if not st.session_state.enable_ai:
                st.info("AI Feedback is disabled. Enable it in the sidebar.")
            elif "ai_feedback" not in st.session_state:
                st.info("Click below to generate AI feedback (uses Gemini API)")
                if st.button("🤖 Generate AI Feedback", type="primary"):
                    with st.spinner("🤖 Calling Gemini AI... (5-10 seconds)"):
                        import json
                        matched = list(
                            score_report["resume_skills"] &
                            score_report["job_skills"]
                        )
                        missing = list(
                            score_report["job_skills"] -
                            score_report["resume_skills"]
                        )
                        feedback = run_ai_feedback(
                            resume_text    = resume_text,
                            job_text       = job_desc,
                            ats_score      = score_report["final_score"],
                            matched_skills = tuple(matched),
                            missing_skills = tuple(missing),
                            section_scores = json.dumps(
                                score_report["section_scores"]
                            ),
                            candidate_name = structure["candidate_name"],
                        )
                        st.session_state.ai_feedback = feedback
                        st.rerun()  # rerun to display results
            else:
                render_ai_tab(st.session_state.ai_feedback)


# ── Entry point ────────────────────────────────────────────────────────
# When Streamlit runs this file, it calls main() on every rerun.
if __name__ == "__main__":
    main()