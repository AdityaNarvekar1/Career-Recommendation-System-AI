# config.py — Central configuration for the entire project
import os
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent
DATA_RAW        = BASE_DIR / "data" / "raw"
DATA_PROCESSED  = BASE_DIR / "data" / "processed"
MODELS_DIR      = BASE_DIR / "outputs" / "models"
REPORTS_DIR     = BASE_DIR / "outputs" / "reports"

for d in [DATA_RAW, DATA_PROCESSED, MODELS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── O*NET Raw File Names (place these in data/raw/) ─────────────────────────
ONET_FILES = {
    "occupation":      "Occupation Data.xlsx",
    "skills":          "Skills.xlsx",
    "abilities":       "Abilities.xlsx",
    "knowledge":       "Knowledge.xlsx",
    "interests":       "Interests.xlsx",
    "work_activities": "Work Activities.xlsx",
    "work_styles":     "Work Styles.xlsx",
    "job_zones":       "Job Zones.xlsx",
    "tasks":           "Task Statements.xlsx",
}

# ─── RIASEC Codes ─────────────────────────────────────────────────────────────
RIASEC_CODES = ["Realistic", "Investigative", "Artistic", "Social", "Enterprising", "Conventional"]

RIASEC_DESCRIPTIONS = {
    "Realistic":     "hands-on, mechanical, and technical skills",
    "Investigative": "analytical, scientific, and research skills",
    "Artistic":      "creative, expressive, and design skills",
    "Social":        "helping, teaching, and interpersonal skills",
    "Enterprising":  "leadership, persuasion, and business skills",
    "Conventional":  "organizing, data management, and detail-oriented skills",
}

RIASEC_CAREER_DOMAINS = {
    "Realistic":     "Engineering & Trades",
    "Investigative": "Science & Research",
    "Artistic":      "Arts & Design",
    "Social":        "Healthcare & Education",
    "Enterprising":  "Business & Management",
    "Conventional":  "Finance & Administration",
}

# ─── Feature Engineering ──────────────────────────────────────────────────────
TFIDF_MAX_FEATURES    = 100       # TF-IDF features from task statements
TFIDF_NGRAM_RANGE     = (1, 2)    # Unigrams + bigrams
KMEANS_MAX_K          = 15        # Max K for elbow method
KMEANS_OPTIMAL_K      = 8         # Will be overridden by elbow method
TOP_N_RECOMMENDATIONS = 10        # Default top-N careers
RIASEC_THRESHOLD      = 0.15      # Min score to be "dominant" RIASEC type

# ─── Model Random State ───────────────────────────────────────────────────────
RANDOM_STATE = 42

# ─── Processed File Names ─────────────────────────────────────────────────────
MASTER_FEATURES_FILE  = DATA_PROCESSED / "master_features.pkl"
MASTER_LABELS_FILE    = DATA_PROCESSED / "master_labels.pkl"
OCCUPATION_META_FILE  = DATA_PROCESSED / "occupation_meta.pkl"
TFIDF_MATRIX_FILE     = DATA_PROCESSED / "tfidf_matrix.pkl"
TFIDF_VECTORIZER_FILE = DATA_PROCESSED / "tfidf_vectorizer.pkl"
JOB_VECTORS_FILE      = DATA_PROCESSED / "job_vectors.pkl"
KMEANS_MODEL_FILE     = MODELS_DIR    / "kmeans_model.pkl"
BEST_MODEL_FILE       = MODELS_DIR    / "best_classifier.pkl"
SCALER_FILE           = DATA_PROCESSED / "scaler.pkl"
