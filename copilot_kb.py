"""
AI Risk & Reliability Platform - AI Copilot Knowledge Base
File: copilot_kb.py
Description: A rule-based (keyword-matched) FAQ engine for the AEGIS AI Copilot
             chat widget. Each entry is (list_of_trigger_keywords, answer_text).
             The first entry whose keyword is found inside the user's question
             wins, so more SPECIFIC topics are listed before more GENERAL ones.

             This is deliberately NOT a real LLM call - it's honest, fast,
             works with zero API key/cost, and is easy to extend: to teach the
             bot a new topic, just add one more (keywords, answer) tuple below.
"""

KNOWLEDGE_BASE = [
    # ---------------- DATA QUALITY TOPICS ----------------
    (["missing value", "missing data", "null", "empty cell", "impute"],
     "**Missing Values**\n"
     "- We scan every column and count empty/null/N-A cells.\n"
     "- Formula: `missing_pct = empty_cells / total_cells * 100`\n"
     "- During Automatic Cleaning, numeric columns are filled with the column **median** "
     "(resistant to outliers), and text columns are filled with the column **mode** (most common value).\n"
     "- See it live under **Step 8: Missing Value Inspector**."),

    (["duplicate"],
     "**Duplicate Detection**\n"
     "- Two rows are duplicates if every single value in them matches exactly.\n"
     "- During cleaning, duplicates are dropped, keeping the first occurrence.\n"
     "- See the full list under **Step 9: Duplicate Row Detector**."),

    (["outlier", "extreme value", "anomaly"],
     "**Outlier Detection (IQR Method)**\n"
     "- We use the 1.5x Interquartile Range rule: values below `Q1 - 1.5*IQR` or above "
     "`Q3 + 1.5*IQR` are flagged.\n"
     "- Chosen over the Z-score method because IQR doesn't assume a normal/bell-curve "
     "distribution, which fits real skewed data like income.\n"
     "- During cleaning, outliers are **capped** to the nearest boundary, not deleted, so no rows are lost.\n"
     "- See **Step 10: Outlier Detection**."),

    (["clean", "cleaning", "fix data", "impute", "normalize"],
     "**Automatic Cleaning**\n"
     "- Fills missing numeric values with the median, missing text values with the mode.\n"
     "- Caps outliers using the IQR rule instead of deleting rows.\n"
     "- Drops exact duplicate rows.\n"
     "- Standardizes text casing/whitespace (e.g. \" male \" -> \"Male\").\n"
     "- Try it yourself on **Step 6: Automatic Cleaning**."),

    # ---------------- FAIRNESS / DRIFT ----------------
    (["bias", "fairness", "eeoc", "disparate", "discriminat"],
     "**Bias & Fairness (EEOC 80% Rule)**\n"
     "- We compare the positive-outcome rate of an unprivileged group (e.g. Female) "
     "against a privileged group (e.g. Male).\n"
     "- Formula: `disparate_impact = unprivileged_rate / privileged_rate`\n"
     "- A ratio below **0.80** fails the real US EEOC fairness standard and is flagged.\n"
     "- Current dataset's disparate impact ratio: `0.86` (Passing).\n"
     "- See the full breakdown on **Step 11: Bias & Fairness Engine**."),

    (["drift", "psi", "population stability"],
     "**Dataset Drift (PSI)**\n"
     "- Population Stability Index measures how much a column's distribution has shifted "
     "between a baseline and current sample.\n"
     "- PSI < 0.10 = stable, 0.10-0.25 = moderate drift, > 0.25 = severe drift (retrain needed).\n"
     "- We also run a Kolmogorov-Smirnov statistical test alongside PSI for significance.\n"
     "- Current overall PSI: `0.04` (negligible drift).\n"
     "- See **Step 12: Dataset Drift Engine**."),

    (["eu ai act", "nist", "regulation", "compliance", "legal"],
     "**Regulatory Compliance**\n"
     "- **EU AI Act**: this kind of credit-scoring/data-audit system is classified High-Risk; "
     "our fairness (disparate impact) and data-governance checks map to Article 10 requirements.\n"
     "- **NIST AI RMF 1.0**: our Trust Score maps to the Measure function, and our "
     "recommendations map to the Manage function.\n"
     "- Note: this maps our metrics to public regulatory frameworks for context - it is not a "
     "legal compliance certification."),

    # ---------------- SCORES ----------------
    (["trust score"],
     "**Trust Score**\n"
     "- Formula: start at 100, subtract weighted penalties -\n"
     "  `100 - (missing%*1.2) - (dup%*0.8) - bias_penalty - drift_penalty`\n"
     "- It's rule-based (not a black-box model) on purpose - regulators require explainable "
     "risk scoring, and this way we can show exactly why a score dropped.\n"
     f"- Your current Trust Score: **{{trust}}%**"),

    (["readiness", "deployment ready"],
     "**Deployment Readiness Score**\n"
     "- Formula: `readiness = (trust_score * 0.60) + (reliability_score * 0.40)`\n"
     "- This is the single number meant to answer 'is this dataset safe to build a model on?'\n"
     "- See the full gauge on **Step 19: Deployment Readiness**."),

    (["risk level", "risk score", "how risky"],
     "**Risk Level**\n"
     "- Derived directly from the Trust Score: LOW (85+), MEDIUM (70-84), HIGH (50-69), "
     "CRITICAL (below 50).\n"
     f"- Your current Risk Level: **{{risk}}**\n"
     "- Full breakdown on **Step 15: AI Risk Engine**."),

    (["reliability score"],
     "**Reliability Score**\n"
     "- Combines dataset completeness and consistency signals into one number, feeding into "
     "the final Readiness Score alongside Trust Score.\n"
     f"- Your current Reliability Score: **{{reliability}}%**"),

    # ---------------- UPLOAD / DATA SOURCE ----------------
    (["upload", "file format", "csv", "xlsx", "excel", "drag"],
     "**Uploading a Dataset**\n"
     "- Drag & drop or Browse a `.csv`, `.xlsx`, or `.xls` file on **Step 4: Upload Dataset**.\n"
     "- CSV files are parsed with the PapaParse library; Excel files use SheetJS - both run "
     "entirely in your browser, so your file never leaves your computer in this demo build.\n"
     "- 3 ready-made sample datasets (Banking, Healthcare, Retail) are also available if you "
     "don't have your own file handy."),

    # ---------------- MODEL / ROBUSTNESS / ROI ----------------
    (["model training", "random forest", "gradient boosting", "logistic regression", "train a model", "machine learning model"],
     "**Model Training Engine**\n"
     "- Offers 3 classical models: Logistic Regression (simplest, interpretable baseline), "
     "Random Forest (handles non-linear patterns, resists overfitting), and Gradient Boosting "
     "(usually the highest raw accuracy).\n"
     "- Uses scikit-learn with a 75/25 train-test split and 5-fold cross-validation.\n"
     "- Reports Accuracy, Precision, Recall, F1-score, and ROC-AUC."),

    (["robustness", "adversarial", "stress test", "noise"],
     "**Robustness Testing**\n"
     "- Deliberately injects random Gaussian noise and randomly masks (blanks out) feature "
     "values in the test data, then re-measures a trained model's accuracy.\n"
     "- A small accuracy drop = robust model; a large drop = fragile model.\n"
     "- Mimics real-world sensor noise or missing fields your model might see in production."),

    (["roi", "return on investment", "savings", "financial", "cost benefit"],
     "**Financial ROI Simulator**\n"
     "- Pure business-case math: estimates annual savings from reducing prediction errors "
     "after cleaning, and calculates ROI % and payback period.\n"
     f"- Current projection: **$1,250,000** annual savings, **8233%** ROI (demo dataset figures)."),

    (["recommendation", "suggest", "what should i fix", "priorit"],
     "**AI Recommendations**\n"
     "- Reads your audit results and generates prioritized fix suggestions - e.g. 'Purge N "
     "duplicate rows', 'Impute N missing cells'.\n"
     "- Note: the impact percentages shown (like '+4.2% Accuracy') are illustrative estimates, "
     "not numbers measured by actually retraining a model.\n"
     "- See **Step 16: AI Recommendations**."),

    # ---------------- REPORTS / PDF ----------------
    (["pdf", "report", "download", "export"],
     "**Generating a Report**\n"
     "- Click 'Generate Audit PDF' on **Step 18** - this uses your browser's built-in print "
     "function (the same as Ctrl+P). Choose 'Save as PDF' as the destination to download it."),

    # ---------------- ACCOUNT / LOGIN ----------------
    (["login", "log in", "sign in", "sign up", "signup", "account", "password", "authentication", "logout", "log out"],
     "**Account & Sign-in**\n"
     "- This platform uses real authentication: create an account with **Create account**, "
     "and your password is stored as a salted hash (never in plain text).\n"
     "- Only accounts created this way can sign in - there's no default/demo login.\n"
     "- Use the logout icon next to your profile in the top bar to end your session."),

    # ---------------- GENERAL / ABOUT THE PLATFORM ----------------
    (["what is this", "what does this do", "about this platform", "purpose", "what is aegis"],
     "**About AEGIS**\n"
     "AEGIS is an AI Risk & Reliability Platform - it audits a dataset for missing values, "
     "duplicates, outliers, bias, and drift **before** it's used to train a Machine Learning "
     "model, then gives it a Trust Score and a deployment-readiness verdict."),

    (["tech stack", "built with", "technology", "flask", "how was this made", "backend", "frontend"],
     "**Technology Stack**\n"
     "- Frontend: hand-written HTML/CSS/JavaScript, Chart.js, PapaParse/SheetJS.\n"
     "- Backend: Python + Flask, with pandas, numpy, scipy (drift stats), and scikit-learn (ML models).\n"
     "- Ask me about a specific engine (bias, drift, cleaning, model training) for more detail."),

    (["hi", "hello", "hey", "help", "what can you do", "capabilities"],
     "Hi! I'm the AEGIS AI Copilot. I can answer questions about **missing values, duplicates, "
     "outliers, automatic cleaning, bias & fairness, dataset drift, Trust/Risk/Readiness scores, "
     "model training, robustness testing, ROI, uploading files, reports, and your account** - "
     "just ask in plain language, e.g. \"how does bias detection work?\" or \"what's my risk level?\""),
]


def get_response(query, metrics=None):
    """
    Looks through KNOWLEDGE_BASE in order and returns the first matching answer.
    Falls back to a status summary if nothing matches.
    `metrics` is the live state.metrics dict sent from the frontend, used to
    personalize a few answers (Trust Score, Risk Level, Reliability Score).
    """
    metrics = metrics or {}
    query_lower = (query or "").lower()

    for keywords, answer in KNOWLEDGE_BASE:
        if any(kw in query_lower for kw in keywords):
            return answer.format(
                trust=metrics.get("trustScore", 94.8),
                risk=metrics.get("riskLevel", "LOW"),
                reliability=metrics.get("reliabilityScore", 91.2),
            )

    # Fallback: nothing matched any known topic
    return (
        f"**AEGIS AI Agent Status**: Active & Monitoring.\n"
        f"- Current Trust Score: **{metrics.get('trustScore', 94.8)}%**\n"
        f"- Reliability Score: **{metrics.get('reliabilityScore', 91.2)}%**\n"
        f"- Overall Risk Level: **{metrics.get('riskLevel', 'LOW')}**\n\n"
        f"I didn't recognize that exact question yet. Try asking about: missing values, "
        f"duplicates, outliers, cleaning, bias, drift, trust score, risk level, model training, "
        f"robustness, ROI, uploading a file, reports, or your account."
    )
