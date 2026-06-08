# data/skills_db.py
# ─────────────────────────────────────────────────────────────────────
# PURPOSE : Master database of 200+ technical skills grouped by domain.
#           Each domain is a dict key mapping to a set of skill strings.
#
# WHY A SET (not a list):
#   - Sets have O(1) lookup: "python" in SKILLS_DB["languages"] is instant
#   - Lists have O(n) lookup: must scan every element
#   - We check membership thousands of times, so O(1) matters
#
# WHY LOWERCASE:
#   All skills stored lowercase so we can compare against lowercased
#   resume tokens directly — no .lower() needed at match time.
#
# USED BY : utils/keyword_extractor.py
# ─────────────────────────────────────────────────────────────────────

SKILLS_DB = {

    # ── Programming Languages ────────────────────────────────────────
    "languages": {
        "python", "java", "javascript", "typescript", "c", "c++", "c#",
        "go", "golang", "rust", "swift", "kotlin", "scala", "ruby",
        "php", "r", "matlab", "perl", "bash", "shell", "powershell",
        "dart", "julia", "haskell", "lua", "groovy", "elixir",
    },

    # ── Web Development ──────────────────────────────────────────────
    "web": {
        # Frontend
        "html", "css", "react", "reactjs", "angular", "vue", "vuejs",
        "nextjs", "nuxtjs", "svelte", "jquery", "bootstrap", "tailwind",
        "sass", "webpack", "vite", "redux", "graphql",
        # Backend
        "flask", "django", "fastapi", "express", "nodejs", "springboot",
        "spring", "laravel", "rails", "asp.net", "rest", "api",
        "restful", "microservices", "websocket", "oauth", "jwt",
    },

    # ── Databases ────────────────────────────────────────────────────
    "databases": {
        # Relational
        "sql", "mysql", "postgresql", "postgres", "sqlite", "oracle",
        "mssql", "mariadb",
        # NoSQL
        "mongodb", "cassandra", "redis", "elasticsearch", "dynamodb",
        "firebase", "couchdb", "neo4j",
        # Big Data
        "hadoop", "hive", "spark", "hbase", "kafka", "flink",
    },

    # ── Machine Learning & AI ────────────────────────────────────────
    "ml_ai": {
        # Core ML
        "machine learning", "deep learning", "neural network",
        "natural language processing", "nlp", "computer vision",
        "reinforcement learning", "transfer learning",
        # Algorithms
        "regression", "classification", "clustering", "svm",
        "random forest", "xgboost", "gradient boosting", "knn",
        "decision tree", "naive bayes",
        # Frameworks
        "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn",
        "huggingface", "transformers", "bert", "gpt", "llm",
        "langchain", "opencv", "nltk", "spacy", "fastai",
        # MLOps
        "mlflow", "kubeflow", "airflow", "feature engineering",
        "model deployment", "onnx",
    },

    # ── Data Science & Analytics ─────────────────────────────────────
    "data_science": {
        "pandas", "numpy", "matplotlib", "seaborn", "plotly",
        "scipy", "statsmodels", "data analysis", "data visualization",
        "data wrangling", "etl", "data pipeline", "tableau", "powerbi",
        "excel", "statistics", "probability", "hypothesis testing",
        "a/b testing", "jupyter", "data mining", "feature selection",
    },

    # ── Cloud & DevOps ───────────────────────────────────────────────
    "cloud_devops": {
        # Cloud Providers
        "aws", "azure", "gcp", "google cloud", "heroku", "vercel",
        "netlify", "digitalocean", "cloudflare",
        # AWS Services
        "ec2", "s3", "lambda", "rds", "sqs", "sns", "cloudwatch",
        "iam", "vpc",
        # DevOps Tools
        "docker", "kubernetes", "k8s", "terraform", "ansible",
        "jenkins", "github actions", "gitlab ci", "circleci",
        "prometheus", "grafana", "nginx", "apache",
        # Practices
        "ci/cd", "devops", "devsecops", "infrastructure as code",
        "serverless", "helm",
    },

    # ── Version Control & Collaboration ──────────────────────────────
    "tools": {
        "git", "github", "gitlab", "bitbucket", "svn",
        "jira", "confluence", "trello", "notion", "slack",
        "postman", "swagger", "figma", "linux", "unix",
        "vim", "vscode", "intellij", "eclipse", "jupyter",
    },

    # ── Software Engineering Concepts ────────────────────────────────
    "concepts": {
        "data structures", "algorithms", "object oriented",
        "oop", "design patterns", "solid", "system design",
        "distributed systems", "concurrency", "multithreading",
        "agile", "scrum", "tdd", "unit testing", "integration testing",
        "code review", "documentation", "mvc", "api design",
        "load balancing", "caching", "message queue",
    },

    # ── Soft Skills (for completeness) ───────────────────────────────
    "soft_skills": {
        "communication", "leadership", "teamwork", "problem solving",
        "critical thinking", "time management", "adaptability",
        "collaboration", "presentation", "mentoring",
    },
}


# ── Flat set of ALL skills ─────────────────────────────────────────────
# We build this once here using a set comprehension.
# set.union(*list_of_sets) merges all domain sets into one big set.
# Used when we don't care about domains — just "is this word a skill?"
ALL_SKILLS = set().union(*SKILLS_DB.values())
# Example: ALL_SKILLS = {"python", "java", "flask", "docker", ...}  (200+ items)


# ── Domain display names (for UI labels) ──────────────────────────────
DOMAIN_LABELS = {
    "languages"    : "Programming Languages",
    "web"          : "Web Development",
    "databases"    : "Databases",
    "ml_ai"        : "Machine Learning & AI",
    "data_science" : "Data Science",
    "cloud_devops" : "Cloud & DevOps",
    "tools"        : "Tools & Version Control",
    "concepts"     : "CS Concepts",
    "soft_skills"  : "Soft Skills",
}


# ── Priority domains for TCS NQT roles ────────────────────────────────
# These are the domains TCS weights most heavily in screening.
# Used to highlight gaps most relevant to the target role.
TCS_PRIORITY_DOMAINS = [
    "languages",
    "web",
    "databases",
    "concepts",
    "tools",
]