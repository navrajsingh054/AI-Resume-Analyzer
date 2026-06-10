# 🤖 AI Resume Analyzer

An intelligent resume analysis tool that scores your resume against
any job description using NLP and AI, identifies skill gaps, and
provides actionable improvement suggestions powered by Groq LLaMA3.

**Live Demo:** [ai-resume-analyzer-rhy36hsuxiexzpcnuslddq.streamlit.app](https://ai-resume-analyzer-rhy36hsuxiexzpcnuslddq.streamlit.app)

---

## ✨ Features

- **ATS Score** — TF-IDF cosine similarity scoring (0-100) against job descriptions
- **Skill Gap Analysis** — Matches 200+ tech skills across 9 domains
- **Resume Structure Detection** — spaCy NER extracts companies, dates, education
- **AI Feedback** — Groq LLaMA3 generates specific, actionable improvements
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
| AI Feedback | Groq API (LLaMA 3.1 8B Instant) |
| Data Processing | Pandas, NumPy |
| Report Generation | fpdf2 |

---

## 🚀 Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/navrajsingh054/AI-Resume-Analyzer.git
cd AI-Resume-Analyzer
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

**4. Add your Groq API key**
```bash
# Create .env file
echo GROQ_API_KEY=your_key_here > .env
```
Get a free key at: https://console.groq.com

**5. Run the app**
```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---
## AI-Resume-Analyzer/
│
├── app.py                    # Streamlit web UI
├── requirements.txt          # Dependencies
├── runtime.txt               # Python version for deployment
├── .env                      # API keys (never commit this)
│
├── utils/
│   ├── extractor.py          # PDF/DOCX text extraction (PyMuPDF)
│   ├── cleaner.py            # NLP preprocessing (spaCy)
│   ├── keyword_extractor.py  # Skill matching and gap analysis
│   ├── scorer.py             # TF-IDF ATS scoring
│   ├── section_detector.py   # Resume structure analysis
│   ├── ai_feedback.py        # Groq LLaMA3 AI feedback
│   └── report_generator.py   # PDF report generation
│
└── data/
    └── skills_db.py          # 200+ skills database

───────────────────────────────────────

## How It Works:

Resume PDF/DOCX
      ↓
Text Extraction (PyMuPDF)
      ↓
NLP Cleaning (spaCy: tokenize → stopwords → lemmatize)
      ↓
┌─────────────────────────────────────┐
│  Skill Extraction  (set matching)   │
│  ATS Scoring       (TF-IDF + cos)   │
│  Section Detection (NER + regex)    │
│  AI Feedback       (Groq LLaMA3)    │
└─────────────────────────────────────┘
      ↓
## Streamlit Dashboard + PDF Report
---

## ⚙️ Environment Variables

| Variable | Description | Get it from |
|---|---|---|
| `GROQ_API_KEY` | Groq API key for AI feedback | https://console.groq.com |

---

## 🎯 Built For

This project was built to demonstrate:
- Real-world Python application development
- NLP pipeline design and implementation
- API integration and prompt engineering
- Software engineering best practices

---

## 📄 License

MIT License — free to use, modify, and distribute.