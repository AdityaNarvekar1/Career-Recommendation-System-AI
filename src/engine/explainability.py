# src/engine/explainability.py — SHAP-based Explainability Layer
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import joblib
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import RIASEC_CODES, REPORTS_DIR


def compute_shap_values(model, X_train: pd.DataFrame, X_explain: pd.DataFrame,
                         max_samples: int = 200):
    """
    Compute SHAP values using the best estimator.
    Uses TreeExplainer for tree-based models, KernelExplainer otherwise.
    Returns shap_values array.
    """
    print("\n🔍 Computing SHAP values...")

    # Get the first sub-estimator to detect model type
    try:
        base_estimator = model.estimators_[0]
    except AttributeError:
        base_estimator = model

    # Sample background data for efficiency
    background = shap.sample(X_train, min(max_samples, len(X_train)))

    try:
        explainer   = shap.TreeExplainer(base_estimator)
        shap_values = explainer.shap_values(X_explain)
        print("  ✅ Used TreeExplainer")
    except Exception:
        try:
            explainer   = shap.LinearExplainer(base_estimator, background)
            shap_values = explainer.shap_values(X_explain)
            print("  ✅ Used LinearExplainer")
        except Exception:
            explainer   = shap.KernelExplainer(base_estimator.predict, background)
            shap_values = explainer.shap_values(X_explain, nsamples=100)
            print("  ✅ Used KernelExplainer")

    return explainer, shap_values


def plot_shap_summary(shap_values, X_explain: pd.DataFrame,
                       riasec_idx: int = 1, top_n: int = 20):
    """
    Plot SHAP summary for a specific RIASEC dimension.
    riasec_idx: 0=Realistic, 1=Investigative, 2=Artistic, 3=Social, 4=Enterprising, 5=Conventional
    """
    riasec_name = RIASEC_CODES[riasec_idx]
    print(f"\n  📊 SHAP summary for: {riasec_name}")

    fig, ax = plt.subplots(figsize=(10, 7))

    if isinstance(shap_values, list):
        sv = shap_values[riasec_idx]
    elif shap_values.ndim == 3:
        sv = shap_values[:, :, riasec_idx]
    else:
        sv = shap_values

    # Get top features by mean |SHAP|
    mean_shap   = np.abs(sv).mean(axis=0)
    top_indices = np.argsort(mean_shap)[-top_n:]
    top_features = X_explain.columns[top_indices]

    shap.summary_plot(
        sv[:, top_indices],
        X_explain.iloc[:, top_indices],
        feature_names=[f.replace("skill__","").replace("abil__","").replace("know__","")
                        .replace("wa__","").replace("tfidf__","[text] ")
                        for f in top_features],
        show=False,
        plot_type="bar"
    )
    plt.title(f"Top Features for {riasec_name} Careers (SHAP)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / f"shap_{riasec_name.lower()}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾 Saved: shap_{riasec_name.lower()}.png")


def get_top_features_for_user(
    shap_values,
    X_user: pd.DataFrame,
    riasec_dist: dict,
    top_n: int = 5
) -> dict:
    """
    For a single user vector, extract top positive SHAP features per dominant RIASEC type.
    Returns: {"Investigative": [("feature_name", shap_val), ...], ...}
    """
    results = {}
    dominant_types = sorted(riasec_dist, key=riasec_dist.get, reverse=True)[:2]

    for i, riasec_name in enumerate(RIASEC_CODES):
        if riasec_name not in dominant_types:
            continue

        if isinstance(shap_values, list):
            sv_user = shap_values[i][0] if len(shap_values[i].shape) > 1 else shap_values[i]
        elif shap_values.ndim == 3:
            sv_user = shap_values[0, :, i]
        else:
            sv_user = shap_values[0]

        # Top positive SHAP values = features boosting this RIASEC type
        shap_series = pd.Series(sv_user, index=X_user.columns)
        top_positive = shap_series.nlargest(top_n)

        results[riasec_name] = [
            (
                col.replace("skill__","").replace("abil__","").replace("know__","")
                   .replace("wa__","").replace("tfidf__",""),
                round(float(val), 4)
            )
            for col, val in top_positive.items() if val > 0
        ]

    return results


def generate_text_explanation(
    riasec_dist: dict,
    shap_top_features: dict,
    career_title: str,
    match_score: float
) -> str:
    """
    Generate a natural language explanation using SHAP insights.
    """
    dominant = sorted(riasec_dist, key=riasec_dist.get, reverse=True)[:2]

    lines = [f"🎯 Why you match '{career_title}' ({match_score:.1f}% fit):"]
    lines.append(f"   • Your profile is {dominant[0]} ({riasec_dist[dominant[0]]*100:.0f}%) "
                 f"+ {dominant[1]} ({riasec_dist[dominant[1]]*100:.0f}%)")

    for d in dominant:
        if d in shap_top_features and shap_top_features[d]:
            feats = ", ".join([f[0] for f in shap_top_features[d][:3]])
            lines.append(f"   • Strong in {d} because of: {feats}")

    return "\n".join(lines)
