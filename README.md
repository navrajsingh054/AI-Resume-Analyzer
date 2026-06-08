# 🤖 AI Resume Analyzer

An intelligent resume analysis tool that scores your resume against
any job description using NLP and AI, identifies skill gaps, and
provides actionable improvement suggestions powered by Google Gemini.

**Live Demo:** [your-app-name.streamlit.app](https://your-app-name.streamlit.app)

---

## ✨ Features

- **ATS Score** — TF-IDF cosine similarity scoring (0-100) against job descriptions
- **Skill Gap Analysis** — Matches 200+ tech skills across 9 domains
- **Resume Structure Detection** — spaCy NER extracts companies, dates, education
- **AI Feedback** — Google Gemini generates specific, actionable improvements
- **PDF Report** — Downloadable 4-page analysis report
- **Zero data storage** — your resume never leaves your browser session

---

## 🛠 Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| NLP Pipeline | spaCy (en_core_web_sm) |
| ATS Scoring | TF-IDF + Cosine Similarity (scikit-learn) |
| PDF Extraction | PyMuPDF, python-docx |
| AI Feedback | Google Gemini 1.5 Flash API |
| Data Processing | Pandas, NumPy |
| Report Generation | fpdf2 |

---

## 🚀 Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/ai-resume-analyzer.git
cd ai-resume-analyzer
```

**2. Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**4. Add your Gemini API key**
```bash
# Create .env file
echo GEMINI_API_KEY=your_key_here > .env
```
Get a free key at: https://aistudio.google.com/app/apikey

**5. Run the app**
```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---
