# src/utils/feature_engineering.py — Build 3-group feature vectors
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import (
    TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, RANDOM_STATE,
    TFIDF_VECTORIZER_FILE, SCALER_FILE, DATA_PROCESSED
)
from src.utils.data_loader import extract_numeric_features


# =============================================================================
# MISSING VALUE STRATEGY
# O*NET scores range 0-7. Missing = "not applicable" for that occupation.
# We fill with 0, NOT median. Median would falsely give occupations a moderate
# score in skills they genuinely do not use.
# =============================================================================

def audit_missing_values(data: dict) -> pd.DataFrame:
    print("\n[MISSING VALUE AUDIT]")
    print("-" * 65)
    strategies = {
        "occupation":      "No action (0% missing)",
        "skills":          "Fill 0 - absent = not required",
        "abilities":       "Fill 0 - absent = not required",
        "knowledge":       "Fill 0 - domain not applicable",
        "interests":       "No action (0% missing)",
        "work_activities": "Fill 0 - activity not performed",
        "work_styles":     "No action (0% missing)",
        "job_zones":       "No action (0% missing)",
        "tasks":           "Drop rows; empty string for TF-IDF",
    }
    rows = []
    for key, df in data.items():
        total   = df.size
        missing = df.isnull().sum().sum()
        pct     = missing / total * 100
        strat   = strategies.get(key, "Fill 0")
        print(f"  {key:<22} {missing:>8,} missing ({pct:>5.2f}%)  {strat}")
        rows.append({"File": key, "Total": total, "Missing": missing,
                     "Pct": round(pct, 2), "Strategy": strat})
    print("-" * 65)
    return pd.DataFrame(rows)


# =============================================================================
# GROUP 1: Skills + Abilities + Knowledge + Work Styles
# =============================================================================

def build_numeric_features(data: dict) -> pd.DataFrame:
    print("\n  [Group 1] Skills + Abilities + Knowledge + Styles...")
    g = {}
    for key, prefix in [("skills","skill__"),("abilities","abil__"),
                         ("knowledge","know__"),("work_styles","style__")]:
        piv = extract_numeric_features(data[key]).fillna(0)
        piv.columns = [prefix + c for c in piv.columns]
        g[key] = piv

    numeric = (g["skills"]
               .join(g["abilities"],   how="outer")
               .join(g["knowledge"],   how="outer")
               .join(g["work_styles"], how="outer")
               .fillna(0))

    print(f"    -> Shape: {numeric.shape}")
    return numeric


# =============================================================================
# GROUP 2: Task Statements -> TF-IDF
# =============================================================================

def build_tfidf_features(data: dict, soc_index: pd.Index,
                          fit: bool = True, vectorizer=None) -> tuple:
    print("\n  [Group 2] TF-IDF from Task Statements...")
    tasks = data["tasks"].copy()
    tasks.columns = [c.strip() for c in tasks.columns]
    task_col = "Task" if "Task" in tasks.columns else tasks.columns[-1]

    task_text = (
        tasks.groupby("O*NET-SOC Code")[task_col]
        .apply(lambda x: " ".join(x.dropna().astype(str)))
        .reindex(soc_index, fill_value="")
    )

    if fit:
        vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
            stop_words="english",
            sublinear_tf=True
        )
        tfidf_matrix = vectorizer.fit_transform(task_text)
        joblib.dump(vectorizer, TFIDF_VECTORIZER_FILE)
    else:
        tfidf_matrix = vectorizer.transform(task_text)

    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        index=soc_index,
        columns=["tfidf__" + f for f in vectorizer.get_feature_names_out()]
    )
    print(f"    -> Shape: {tfidf_df.shape}")
    return tfidf_df, vectorizer


# =============================================================================
# GROUP 3: Work Activities (Importance scale)
# =============================================================================

def build_work_activity_features(data: dict) -> pd.DataFrame:
    print("\n  [Group 3] Work Activities...")
    wa = data["work_activities"].copy()
    wa["Data Value"] = pd.to_numeric(wa["Data Value"], errors="coerce")
    if "Scale Name" in wa.columns:
        wa = wa[wa["Scale Name"] == "Importance"]

    pivoted = wa.pivot_table(
        index="O*NET-SOC Code",
        columns="Element Name",
        values="Data Value",
        aggfunc="mean"
    ).fillna(0)

    pivoted.columns = ["wa__" + str(c).strip() for c in pivoted.columns]
    print(f"    -> Shape: {pivoted.shape}")
    return pivoted


# =============================================================================
# COMBINE ALL GROUPS + SCALE
# CRITICAL: Scaler fitted on NUMPY (no feature names) so inference
# also passes numpy -> zero sklearn feature-name warnings.
# =============================================================================

def build_full_feature_matrix(data: dict, fit_scaler: bool = True) -> tuple:
    print("\n[Building Full Feature Matrix]")

    group1 = build_numeric_features(data)
    group2, vectorizer = build_tfidf_features(data, soc_index=group1.index)
    group3 = build_work_activity_features(data)

    common_soc = (group1.index
                  .intersection(group2.index)
                  .intersection(group3.index))

    X = (group1.loc[common_soc]
         .join(group2.loc[common_soc])
         .join(group3.loc[common_soc]))

    print(f"\n  Combined: {X.shape}  "
          f"(G1={group1.shape[1]}, G2={group2.shape[1]}, G3={group3.shape[1]})")

    # Save feature column names before converting to numpy
    feature_cols = list(X.columns)
    joblib.dump(feature_cols, DATA_PROCESSED / "feature_cols.pkl")

    # FIT SCALER ON NUMPY - this is the root fix for sklearn feature-name warnings.
    # When scaler is fit on numpy it stores no feature names, so transform(numpy)
    # never warns. Previously scaler was fit on DataFrame -> stored names ->
    # warned when numpy was passed at inference time.
    X_np = X.values
    if fit_scaler:
        scaler = MinMaxScaler()
        X_scaled_np = scaler.fit_transform(X_np)
        joblib.dump(scaler, SCALER_FILE)
    else:
        scaler = joblib.load(SCALER_FILE)
        X_scaled_np = scaler.transform(X_np)

    # Rebuild as DataFrame only for downstream indexing (clustering, recommender)
    X_scaled = pd.DataFrame(X_scaled_np, index=common_soc, columns=feature_cols)

    print(f"  Feature matrix ready: {X_scaled.shape}")
    return X_scaled, scaler, vectorizer


# =============================================================================
# SKILL SYNONYM MAP
# Maps sidebar UI labels -> O*NET element keywords.
# Without this "Mathematics" never matches "skill__Mathematical Reasoning".
# =============================================================================

SKILL_SYNONYM_MAP = {
    "Mathematics":       ["mathematical", "mathematics", "statistics", "quantitative"],
    "Critical Thinking": ["critical thinking", "judgment", "decision making", "reasoning"],
    "Data Analysis":     ["data analysis", "information ordering", "inductive reasoning",
                          "deductive reasoning", "statistical"],
    "Research":          ["investigation", "research", "science", "analyzing data"],
    "Programming":       ["programming", "computers", "software", "technology", "computer"],
    "Engineering":       ["engineering", "mechanical", "physics", "technical"],
    "Science":           ["science", "biology", "chemistry", "physics", "life sciences"],
    "Technology":        ["technology", "computers", "equipment", "tools"],
    "Communication":     ["speaking", "writing", "active listening", "communication",
                          "oral comprehension", "oral expression", "written"],
    "Teaching":          ["instructing", "teaching", "training", "education", "learning"],
    "Leadership":        ["leadership", "management", "coordination", "directing",
                          "persuasion", "motivating"],
    "Teamwork":          ["cooperation", "social perceptiveness", "coordination",
                          "interpersonal", "working with others"],
    "Writing":           ["writing", "written expression", "documentation", "editing"],
    "Design":            ["design", "originality", "arts", "creativity", "visualization",
                          "fine arts"],
    "Arts":              ["arts", "fine arts", "music", "performing", "creative"],
    "Innovation":        ["originality", "innovation", "creative", "fluency of ideas"],
    "Management":        ["management", "administration", "planning", "organizing",
                          "enterprise", "personnel"],
    "Sales":             ["persuasion", "negotiation", "sales", "marketing", "customer"],
    "Finance":           ["economics", "accounting", "finance", "financial", "mathematics"],
    "Planning":          ["planning", "organizing", "scheduling", "time management",
                          "management of resources"],
}


# =============================================================================
# USER PROFILE VECTORIZATION
# =============================================================================

def vectorize_user_profile(
    user_skills: dict,
    user_interests: dict,
    all_feature_columns: list,
    scaler,
    riasec_codes: list
) -> tuple:
    """
    Convert user sidebar inputs -> scaled numpy-backed feature vector.

    Scaler was fitted on numpy so we pass numpy here -> no sklearn warnings.
    User skill score 1-5 mapped to O*NET range 0-7 linearly.
    Unmatched features stay 0 (no signal for that dimension).
    """
    n = len(all_feature_columns)
    col_index  = {col: i for i, col in enumerate(all_feature_columns)}
    user_vec   = np.zeros(n, dtype=np.float64)
    matched    = set()

    for skill_name, score in user_skills.items():
        normalized = (score / 5.0) * 7.0
        keywords   = SKILL_SYNONYM_MAP.get(skill_name, [skill_name.lower()])
        for col in all_feature_columns:
            col_lower = col.lower()
            for kw in keywords:
                if kw.lower() in col_lower:
                    user_vec[col_index[col]] = normalized
                    matched.add(col)
                    break

    print(f"  Skill matching: {len(matched)}/{n} features matched")

    # Transform as numpy - scaler fitted on numpy = no warnings
    user_scaled_np = scaler.transform(user_vec.reshape(1, -1))  # (1, n_features)

    # Wrap in DataFrame only for column-name access downstream
    user_scaled_df = pd.DataFrame(user_scaled_np, columns=all_feature_columns)

    # RIASEC from interest sliders - normalize to probability distribution
    total      = sum(user_interests.values()) or 1
    riasec_vec = {r: user_interests.get(r, 1) / total for r in riasec_codes}

    return user_scaled_df, riasec_vec