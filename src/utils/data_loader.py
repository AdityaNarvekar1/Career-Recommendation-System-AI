# src/utils/data_loader.py — Load and merge all O*NET files
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import DATA_RAW, ONET_FILES

def load_onet_file(key: str) -> pd.DataFrame:
    """Load a single O*NET file by key."""
    path = DATA_RAW / ONET_FILES[key]
    if not path.exists():
        raise FileNotFoundError(
            f"\n❌ Missing file: {path}"
            f"\n👉 Download '{ONET_FILES[key]}' from https://www.onetcenter.org/database.html"
            f"\n👉 Place it in: {DATA_RAW}"
        )
    print(f"  ✅ Loading {ONET_FILES[key]}...")
    return pd.read_excel(path)


def extract_numeric_features(df: pd.DataFrame, value_col: str = "Data Value") -> pd.DataFrame:
    """
    Pivot long-format O*NET data → wide format per occupation.
    Input:  O*NET file with [O*NET-SOC Code, Element Name, Data Value]
    Output: occupation × element matrix
    """
    # Keep only relevant columns
    cols_needed = ["O*NET-SOC Code", "Element Name", value_col]
    df = df[cols_needed].copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    # Pivot: rows = occupations, cols = element names
    pivoted = df.pivot_table(
        index="O*NET-SOC Code",
        columns="Element Name",
        values=value_col,
        aggfunc="mean"
    )
    pivoted.columns = [str(c).strip() for c in pivoted.columns]
    return pivoted


def load_all_data() -> dict:
    """
    Load all 9 O*NET files and return a dictionary of DataFrames.
    """
    print("\n📂 Loading O*NET Database Files...")
    data = {}

    data["occupation"]      = load_onet_file("occupation")
    data["skills"]          = load_onet_file("skills")
    data["abilities"]       = load_onet_file("abilities")
    data["knowledge"]       = load_onet_file("knowledge")
    data["interests"]       = load_onet_file("interests")
    data["work_activities"] = load_onet_file("work_activities")
    data["work_styles"]     = load_onet_file("work_styles")
    data["job_zones"]       = load_onet_file("job_zones")
    data["tasks"]           = load_onet_file("tasks")

    print(f"\n✅ All files loaded successfully!\n")
    return data


def build_occupation_meta(data: dict) -> pd.DataFrame:
    """
    Build occupation metadata table:
    SOC Code | Title | Description | Job Zone
    """
    occ = data["occupation"][["O*NET-SOC Code", "Title", "Description"]].copy()
    occ.columns = ["soc_code", "title", "description"]

    # Job Zone (education/experience level 1–5)
    jz = data["job_zones"][["O*NET-SOC Code", "Job Zone"]].copy()
    jz.columns = ["soc_code", "job_zone"]
    jz["job_zone"] = pd.to_numeric(jz["job_zone"], errors="coerce")

    meta = occ.merge(jz, on="soc_code", how="left")
    meta["job_zone"] = meta["job_zone"].fillna(3).astype(int)
    meta = meta.drop_duplicates("soc_code").set_index("soc_code")

    print(f"  📋 Occupation meta: {meta.shape[0]} occupations")
    return meta


def build_riasec_labels(data: dict) -> pd.DataFrame:
    """
    Build RIASEC multi-label target from Interests.xlsx.
    Returns: occupation × [Realistic, Investigative, Artistic, Social, Enterprising, Conventional]
    Scores are normalized to sum to 1 per occupation → probability distribution.
    """
    interests_df = data["interests"].copy()

    # Keep only top-level RIASEC (not sub-interest elements)
    riasec_names = ["Realistic", "Investigative", "Artistic", "Social", "Enterprising", "Conventional"]

    # Filter to RIASEC element names only
    riasec_df = interests_df[interests_df["Element Name"].isin(riasec_names)].copy()
    riasec_df["Data Value"] = pd.to_numeric(riasec_df["Data Value"], errors="coerce")

    # Pivot
    labels = riasec_df.pivot_table(
        index="O*NET-SOC Code",
        columns="Element Name",
        values="Data Value",
        aggfunc="mean"
    )[riasec_names]  # enforce column order

    # Normalize each row to sum to 1 (probability distribution)
    row_sums = labels.sum(axis=1)
    labels = labels.div(row_sums, axis=0)
    labels = labels.fillna(0)

    print(f"  🎯 RIASEC labels: {labels.shape[0]} occupations × 6 RIASEC types")
    return labels
