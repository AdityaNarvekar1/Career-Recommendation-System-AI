"""
generate_notebook.py
Run this once to create the full Jupyter Notebook:
    python generate_notebook.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    return nbf.v4.new_markdown_cell(text)

def code(text):
    return nbf.v4.new_code_cell(text)

# ════════════════════════════════════════════════════════════════════════════
cells.append(md("""# 🎯 AI-Powered Career Recommendation System
## Using O*NET Database | End-to-End Machine Learning Pipeline

**Pipeline:**
1. Data Loading & Merging
2. Exploratory Data Analysis (EDA)
3. Feature Engineering (3-group vectors)
4. Preprocessing
5. Multi-Label RIASEC Classification (11 models)
6. Model Comparison & Best Model Selection
7. K-Means Career Clustering (Elbow Method)
8. Hybrid Recommendation Engine
9. Skill Gap Analysis
10. SHAP Explainability
11. Career Transition Path
"""))

# ════════════════════════════════════════════════════════════════════════════
cells.append(md("## 0. Setup & Imports"))
cells.append(code("""\
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.append("..")   # so we can import from src/ and config.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Project modules
from config import *
from src.utils.data_loader import load_all_data, build_occupation_meta, build_riasec_labels
from src.utils.feature_engineering import (
    build_numeric_features, build_tfidf_features,
    build_work_activity_features, build_full_feature_matrix
)
from src.models.classifiers import run_full_model_comparison
from src.models.clustering import find_optimal_k, fit_kmeans, label_clusters, plot_clusters_2d
from src.engine.recommender import HybridCareerRecommender
from src.engine.explainability import (
    compute_shap_values, plot_shap_summary,
    get_top_features_for_user, generate_text_explanation
)

pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", "{:.4f}".format)
print("✅ All imports successful!")
print(f"📁 Data directory: {DATA_RAW}")
"""))

# ════════════════════════════════════════════════════════════════════════════
cells.append(md("---\n## 1. Data Loading & Merging"))
cells.append(code("""\
# ── Load all 9 O*NET files ──────────────────────────────────────────────────
data = load_all_data()

# ── Build occupation metadata ───────────────────────────────────────────────
occupation_meta = build_occupation_meta(data)
print(f"\\n📋 Occupation Meta:\\n{occupation_meta.head(3)}")
print(f"\\nShape: {occupation_meta.shape}")
"""))

cells.append(code("""\
# ── Build RIASEC multi-label targets ────────────────────────────────────────
riasec_labels = build_riasec_labels(data)
print(f"\\n🎯 RIASEC Labels (first 5 rows):")
print(riasec_labels.head())
print(f"\\nShape: {riasec_labels.shape}")
print(f"\\n📊 RIASEC Score Statistics:")
print(riasec_labels.describe().round(4))
"""))

# ════════════════════════════════════════════════════════════════════════════
cells.append(md("---\n## 2. Exploratory Data Analysis (EDA)"))

cells.append(code("""\
# ── 2.1 RIASEC Distribution Across All Occupations ─────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
colors = ["#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6","#1abc9c"]

for i, (riasec, color) in enumerate(zip(RIASEC_CODES, colors)):
    axes[i].hist(riasec_labels[riasec], bins=40, color=color, alpha=0.8, edgecolor="white")
    axes[i].set_title(f"{riasec} Distribution", fontweight="bold")
    axes[i].set_xlabel("Score (normalized)")
    axes[i].set_ylabel("Frequency")
    axes[i].axvline(riasec_labels[riasec].mean(), color="black", linestyle="--",
                    label=f"mean={riasec_labels[riasec].mean():.3f}")
    axes[i].legend(fontsize=9)
    axes[i].grid(True, alpha=0.3)

plt.suptitle("RIASEC Score Distributions Across All Occupations", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "eda_riasec_distributions.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ RIASEC distributions plotted")
"""))

cells.append(code("""\
# ── 2.2 RIASEC Correlation Heatmap ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
corr = riasec_labels.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt=".3f", cmap="coolwarm", center=0,
            mask=mask, ax=ax, linewidths=0.5,
            annot_kws={"size": 11, "weight": "bold"})
ax.set_title("RIASEC Inter-Correlation Heatmap", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "eda_riasec_correlation.png", dpi=150, bbox_inches="tight")
plt.show()
"""))

cells.append(code("""\
# ── 2.3 Dominant RIASEC per Occupation (Bar Chart) ─────────────────────────
dominant_type = riasec_labels.idxmax(axis=1)
counts = dominant_type.value_counts()

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(counts.index, counts.values,
              color=["#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6","#1abc9c"])
ax.set_title("Dominant RIASEC Type per Occupation", fontsize=13, fontweight="bold")
ax.set_xlabel("RIASEC Type")
ax.set_ylabel("Number of Occupations")
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            str(val), ha="center", va="bottom", fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "eda_dominant_riasec.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"\\n📊 Dominant RIASEC counts:\\n{counts}")
"""))

cells.append(code("""\
# ── 2.4 Job Zone Distribution ───────────────────────────────────────────────
jz_counts = occupation_meta["job_zone"].value_counts().sort_index()
jz_labels_map = {
    1: "Zone 1\\n(Little prep)",
    2: "Zone 2\\n(Some prep)",
    3: "Zone 3\\n(Medium prep)",
    4: "Zone 4\\n(Considerable prep)",
    5: "Zone 5\\n(Extensive prep)"
}

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar([jz_labels_map.get(k, str(k)) for k in jz_counts.index],
       jz_counts.values, color="#3498db", alpha=0.85, edgecolor="white")
ax.set_title("Occupation Distribution by Job Zone (Experience Level)", fontsize=13, fontweight="bold")
ax.set_ylabel("Number of Occupations")
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "eda_job_zones.png", dpi=150, bbox_inches="tight")
plt.show()
"""))

cells.append(code("""\
# ── 2.5 Missing Value Audit & Strategy ─────────────────────────────────────
from src.utils.feature_engineering import audit_missing_values
missing_audit = audit_missing_values(data)

print(\"\"\"
╔══════════════════════════════════════════════════════════════╗
║  IMPUTATION DECISION: Fill with 0 (not median)              ║
║                                                              ║
║  Reason: O*NET scores range 0-7. A missing score means the  ║
║  skill/ability/activity is absent or not applicable for      ║
║  that occupation — NOT that the value is unknown.            ║
║                                                              ║
║  Filling with MEDIAN would falsely give every occupation a   ║
║  'moderate' score in skills they never use (e.g. a fisherman ║
║  getting a median score in 'Computer Programming').          ║
║                                                              ║
║  Filling with 0 correctly says: 'this occupation does not    ║
║  require this skill' — preserving the semantic meaning.      ║
╚══════════════════════════════════════════════════════════════╝
\"\"\")
"""))

cells.append(code("""\
# ── 2.6 Top 20 Highest Paying Occupations by Job Zone ──────────────────────
top_jz5 = occupation_meta[occupation_meta["job_zone"] == 5]["title"].head(20)
print("🏆 Top 20 Zone-5 (Highest Expertise) Occupations:")
for i, title in enumerate(top_jz5, 1):
    print(f"  {i:2d}. {title}")
"""))

# ════════════════════════════════════════════════════════════════════════════
cells.append(md("---\n## 3. Feature Engineering (3-Group Vectors)"))

cells.append(code("""\
# ── Build full feature matrix ───────────────────────────────────────────────
X, scaler, tfidf_vectorizer = build_full_feature_matrix(data, fit_scaler=True)

# Align RIASEC labels with feature matrix
common_socs = X.index.intersection(riasec_labels.index)
X = X.loc[common_socs]
y = riasec_labels.loc[common_socs]

print(f"\\n✅ Final Dataset:")
print(f"   X (features): {X.shape}")
print(f"   y (RIASEC):   {y.shape}")
print(f"   Common SOCs:  {len(common_socs)}")

# Save for use by engine
joblib.dump(X, MASTER_FEATURES_FILE)
joblib.dump(y, MASTER_LABELS_FILE)
joblib.dump(occupation_meta, OCCUPATION_META_FILE)
print("\\n💾 Feature matrix saved.")
"""))

cells.append(code("""\
# ── Feature Group Summary ───────────────────────────────────────────────────
groups = {
    "Skills + Abilities + Knowledge + Styles":
        [c for c in X.columns if any(c.startswith(p) for p in ["skill__","abil__","know__","style__"])],
    "TF-IDF (Task Statements)":
        [c for c in X.columns if c.startswith("tfidf__")],
    "Work Activities":
        [c for c in X.columns if c.startswith("wa__")],
}
print("📊 Feature Group Breakdown:")
for name, cols in groups.items():
    print(f"  {name}: {len(cols)} features")
print(f"  TOTAL: {X.shape[1]} features")
"""))

cells.append(code("""\
# ── Top Skill Features by Variance ─────────────────────────────────────────
skill_cols = [c for c in X.columns if c.startswith("skill__")]
variances  = X[skill_cols].var().nlargest(15)
clean_names = [c.replace("skill__","") for c in variances.index]

fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(clean_names[::-1], variances.values[::-1], color="#3498db", alpha=0.85)
ax.set_title("Top 15 Skills by Variance (Most Discriminating)", fontsize=13, fontweight="bold")
ax.set_xlabel("Variance")
ax.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "eda_top_skill_variance.png", dpi=150, bbox_inches="tight")
plt.show()
"""))

# ════════════════════════════════════════════════════════════════════════════
cells.append(md("---\n## 4. Preprocessing & Train-Test Split"))

cells.append(code("""\
# ── Train / Test Split ──────────────────────────────────────────────────────
# IMPORTANT: Convert to numpy BEFORE splitting so models train without
# feature names. At inference time we also convert to numpy → no warnings.
feature_cols = list(X.columns)
joblib.dump(feature_cols, DATA_PROCESSED / "feature_cols.pkl")  # save for app

X_np = X.values   # numpy array — no feature names
y_np = y.values

X_train, X_test, y_train, y_test = train_test_split(
    X_np, y_np, test_size=0.2, random_state=RANDOM_STATE
)

# Keep DataFrame versions only for EDA / SHAP (not for model training)
X_train_df = pd.DataFrame(X_train, columns=feature_cols)
X_test_df  = pd.DataFrame(X_test,  columns=feature_cols)

print(f"✅ Train set: {X_train.shape}  (numpy array — no feature name warnings)")
print(f"✅ Test  set: {X_test.shape}")
print(f"\\n📊 RIASEC label statistics (train):")
print(pd.DataFrame(y_train, columns=RIASEC_CODES).describe().round(4))
"""))

# ════════════════════════════════════════════════════════════════════════════
cells.append(md("---\n## 5. Multi-Output Regression — All Models\n\n> **Why Regression, not Classification?**\n> Our RIASEC target is a *continuous probability distribution* per occupation (e.g. Investigative=0.42, Conventional=0.28...). Classifiers like Logistic Regression expect *discrete class labels* and will throw `Unknown label type: continuous`. The correct approach is **Multi-Output Regression** — predict all 6 RIASEC scores simultaneously as real numbers. We evaluate with R², RMSE, and Cosine Similarity between predicted and actual distributions."))

cells.append(code("""\
# ── Run Full Model Comparison ───────────────────────────────────────────────
leaderboard, trained_models, all_results, best_name, best_model = run_full_model_comparison(
    X_train, y_train, X_test, y_test
)

print("\\n🏆 FINAL LEADERBOARD:")
print(leaderboard.to_string())
"""))

# ════════════════════════════════════════════════════════════════════════════
cells.append(md("---\n## 6. Model Comparison & Visualization"))

cells.append(code("""\
# ── 6.1 R² Score Comparison Bar Chart ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Model Comparison: Multi-Label RIASEC Prediction", fontsize=14, fontweight="bold")

colors_bar = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(leaderboard)))

# R² Score
axes[0].barh(leaderboard["Model"][::-1], leaderboard["R² Score"][::-1], color=colors_bar)
axes[0].set_title("R² Score (higher = better)", fontweight="bold")
axes[0].set_xlabel("R² Score")
axes[0].axvline(0, color="black", linewidth=0.8)
axes[0].grid(True, alpha=0.3, axis="x")

# RMSE
colors_rmse = plt.cm.RdYlGn(np.linspace(0.9, 0.2, len(leaderboard)))
axes[1].barh(leaderboard["Model"][::-1], leaderboard["RMSE"][::-1], color=colors_rmse)
axes[1].set_title("RMSE (lower = better)", fontweight="bold")
axes[1].set_xlabel("RMSE")
axes[1].grid(True, alpha=0.3, axis="x")

# Cosine Similarity
axes[2].barh(leaderboard["Model"][::-1], leaderboard["Cosine Sim"][::-1], color=colors_bar)
axes[2].set_title("Cosine Similarity (higher = better)", fontweight="bold")
axes[2].set_xlabel("Cosine Similarity")
axes[2].grid(True, alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig(REPORTS_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
"""))

cells.append(code("""\
# ── 6.2 Per-RIASEC R² Heatmap for Best Model ────────────────────────────────
best_result = next(r for r in all_results if r["Model"] == best_name)
per_riasec  = pd.DataFrame([best_result["Per-RIASEC R²"]], index=[best_name])

fig, ax = plt.subplots(figsize=(10, 3))
sns.heatmap(per_riasec, annot=True, fmt=".4f", cmap="YlGn",
            ax=ax, linewidths=0.5, cbar_kws={"label": "R² Score"})
ax.set_title(f"Per-RIASEC R² Score — Best Model: {best_name}", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "best_model_per_riasec_r2.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"\\n🥇 Best Model: {best_name}")
print(f"   Overall R²:   {best_result['R² Score']}")
print(f"   RMSE:         {best_result['RMSE']}")
print(f"   Cosine Sim:   {best_result['Cosine Sim']}")
"""))

cells.append(code("""\
# ── 6.3 Prediction vs Actual scatter for best model ─────────────────────────
y_pred_best = best_model.predict(X_test)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for i, (riasec, ax) in enumerate(zip(RIASEC_CODES, axes)):
    ax.scatter(y_test[:, i], y_pred_best[:, i], alpha=0.3, s=10, c="#3498db")
    max_val = max(y_test[:, i].max(), y_pred_best[:, i].max())
    ax.plot([0, max_val], [0, max_val], "r--", linewidth=1.5, label="Perfect")
    ax.set_title(f"{riasec}", fontweight="bold")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle(f"Predicted vs Actual RIASEC Scores — {best_name}", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "pred_vs_actual.png", dpi=150, bbox_inches="tight")
plt.show()
"""))

# ════════════════════════════════════════════════════════════════════════════
cells.append(md("---\n## 7. K-Means Career Clustering"))

cells.append(code("""\
# ── 7.1 Elbow Method ────────────────────────────────────────────────────────
# Use work activities + skills for clustering
cluster_cols = [c for c in X.columns if c.startswith("wa__") or c.startswith("skill__")]
X_cluster    = X[cluster_cols].values

optimal_k, inertias, silhouettes, elbow_fig = find_optimal_k(X_cluster, max_k=KMEANS_MAX_K)
elbow_fig
"""))

cells.append(code("""\
# ── 7.2 Fit Final K-Means ────────────────────────────────────────────────────
km_model, cluster_series = fit_kmeans(X_cluster, k=optimal_k, soc_index=X.index)

# Label clusters semantically using RIASEC profiles
cluster_labels_dict, cluster_summaries = label_clusters(
    km_model, X_cluster, X.index, riasec_labels, occupation_meta
)
print(f"\\n✅ Cluster labels: {cluster_labels_dict}")
"""))

cells.append(code("""\
# ── 7.3 2D PCA Visualization ─────────────────────────────────────────────────
plot_clusters_2d(X_cluster, km_model.labels_, cluster_labels_dict,
                  title=f"Career Clusters (K={optimal_k}) — PCA 2D")
plt.figure(figsize=(1,1))
img = plt.imread(REPORTS_DIR / "cluster_visualization.png")
plt.figure(figsize=(12,8))
plt.imshow(img); plt.axis("off")
plt.show()
"""))

cells.append(code("""\
# ── 7.4 Cluster Size Distribution ────────────────────────────────────────────
cluster_counts = cluster_series.value_counts().sort_index()
cluster_names  = [cluster_labels_dict.get(i, f"Cluster {i}") for i in cluster_counts.index]

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(cluster_names, cluster_counts.values,
              color=plt.cm.Set3(np.linspace(0, 1, len(cluster_counts))))
ax.set_title(f"Career Cluster Sizes (K={optimal_k})", fontsize=13, fontweight="bold")
ax.set_ylabel("Number of Occupations")
plt.xticks(rotation=30, ha="right")
for bar, val in zip(bars, cluster_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            str(val), ha="center", fontweight="bold", fontsize=10)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "cluster_sizes.png", dpi=150, bbox_inches="tight")
plt.show()
"""))

# ════════════════════════════════════════════════════════════════════════════
cells.append(md("---\n## 8. SHAP Explainability"))

cells.append(code("""\
# ── Compute SHAP on test sample ──────────────────────────────────────────────
# Use 100 samples for speed — increase if needed
X_shap_sample = X_test_df.iloc[:100]

try:
    shap_explainer, shap_values = compute_shap_values(
        best_model, X_train_df, X_shap_sample, max_samples=100
    )
    # Plot for Investigative (index 1) — most common in data science careers
    plot_shap_summary(shap_values, X_shap_sample, riasec_idx=1, top_n=15)
    plot_shap_summary(shap_values, X_shap_sample, riasec_idx=0, top_n=15)
    print("\\n✅ SHAP plots saved to reports/")
    SHAP_AVAILABLE = True
except Exception as e:
    print(f"⚠️  SHAP computation skipped: {e}")
    print("    (SHAP works best with tree-based models. Proceeding without it.)")
    SHAP_AVAILABLE = False
    shap_explainer, shap_values = None, None
"""))

# ════════════════════════════════════════════════════════════════════════════
cells.append(md("---\n## 9. Hybrid Recommendation Engine — Live Demo"))

cells.append(code("""\
# ── Build the recommender ────────────────────────────────────────────────────
recommender = HybridCareerRecommender(
    best_model        = best_model,
    job_vectors       = X,
    riasec_labels     = riasec_labels,
    occupation_meta   = occupation_meta,
    cluster_labels    = cluster_labels_dict,
    cluster_assignments = cluster_series,
    feature_columns   = list(X.columns),
    scaler            = scaler,
)
print("✅ Recommender built and ready!")
"""))

cells.append(code("""\
# ── Demo: Define a User Profile ──────────────────────────────────────────────
# Simulating a user who is good at data/analytics and interested in research

USER_SKILLS = {
    "Programming":             4,
    "Mathematics":             5,
    "Critical Thinking":       5,
    "Data Analysis":           4,
    "Writing":                 3,
    "Communication":           3,
    "Problem Solving":         5,
    "Research":                4,
}

USER_INTERESTS = {
    "Investigative": 5,
    "Conventional":  3,
    "Realistic":     2,
    "Artistic":      1,
    "Social":        2,
    "Enterprising":  3,
}

# ── Vectorize user ─────────────────────────────────────────────────────────
from src.utils.feature_engineering import vectorize_user_profile

user_vector, user_riasec_norm = vectorize_user_profile(
    user_skills       = USER_SKILLS,
    user_interests    = USER_INTERESTS,
    all_feature_columns = list(X.columns),
    scaler            = scaler,
    riasec_codes      = RIASEC_CODES
)
print("✅ User profile vectorized!")
print(f"   User vector shape: {user_vector.shape}")
print(f"   User RIASEC (normalized): {user_riasec_norm}")
"""))

cells.append(code("""\
# ── Run Recommendation ───────────────────────────────────────────────────────
results = recommender.recommend(
    user_vector       = user_vector,
    user_riasec_input = USER_INTERESTS,
    top_n             = 10,
    current_career_soc= None    # Set a SOC code here for transition path
)

print(f"\\n{'='*65}")
print(f"🎯 YOUR RIASEC PROFILE")
print(f"{'='*65}")
for code, score in sorted(results['riasec_profile'].items(),
                            key=lambda x: x[1], reverse=True):
    bar = "█" * int(score * 30)
    print(f"  {code:15s} {score*100:5.1f}%  {bar}")

print(f"\\n{'='*65}")
print(f"🏆 TOP {len(results['recommendations'])} CAREER RECOMMENDATIONS")
print(f"{'='*65}")
for r in results["recommendations"]:
    print(f"\\n  #{r['rank']} {r['title']}")
    print(f"     Match Score:    {r['match_score']}%")
    print(f"     Career Domain:  {r['career_domain']}")
    print(f"     Job Zone:       {r['job_zone']}/5")
    if r['skill_gaps']:
        gaps = ", ".join([g[0] for g in r['skill_gaps'][:3]])
        print(f"     Skill Gaps:     {gaps}")
"""))

cells.append(code("""\
# ── Visualize Top Recommendations ───────────────────────────────────────────
recs   = results["recommendations"]
titles = [r["title"][:35] + "..." if len(r["title"]) > 35 else r["title"] for r in recs]
scores = [r["match_score"] for r in recs]
colors_rec = plt.cm.RdYlGn(np.linspace(0.4, 0.9, len(recs)))

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(titles[::-1], scores[::-1], color=colors_rec)
ax.set_title("Top Career Recommendations — Match Score (%)", fontsize=13, fontweight="bold")
ax.set_xlabel("Match Score (%)")
ax.set_xlim(0, 105)
for bar, score in zip(bars, scores[::-1]):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f"{score}%", va="center", fontweight="bold")
ax.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "top_recommendations.png", dpi=150, bbox_inches="tight")
plt.show()
"""))

# ════════════════════════════════════════════════════════════════════════════
cells.append(md("---\n## 10. Skill Gap Analysis — Deep Dive"))

cells.append(code("""\
# ── Skill Gap for Top 3 Careers ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Skill Gap Analysis — Top 3 Recommended Careers", fontsize=13, fontweight="bold")

for idx in range(min(3, len(recs))):
    r    = recs[idx]
    gaps = r["skill_gaps"][:8]
    if not gaps:
        axes[idx].text(0.5, 0.5, "No significant gaps!", ha="center", va="center")
        axes[idx].set_title(r["title"][:30])
        continue
    skills_g, scores_g = zip(*gaps)
    axes[idx].barh(list(skills_g)[::-1], list(scores_g)[::-1], color="#e74c3c", alpha=0.85)
    axes[idx].set_title(r["title"][:30] + f"\\n({r['match_score']}% match)", fontweight="bold")
    axes[idx].set_xlabel("Gap Score")
    axes[idx].grid(True, alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig(REPORTS_DIR / "skill_gap_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
"""))

# ════════════════════════════════════════════════════════════════════════════
cells.append(md("---\n## 11. Career Transition Path — Bonus Feature"))

cells.append(code("""\
# ── Career Transition Demo ───────────────────────────────────────────────────
# Pick top recommended career as target
top_career_soc = recs[0]["soc_code"]

# Find a "current" career — e.g., pick a Zone-2 occupation as starting point
zone2_socs = occupation_meta[occupation_meta["job_zone"] == 2].index
zone2_socs = zone2_socs.intersection(X.index)
current_soc = zone2_socs[0] if len(zone2_socs) > 0 else None

if current_soc:
    transition = recommender.career_transition_path(current_soc, top_career_soc, user_vector)
    print(f"\\n{'='*65}")
    print(f"🛤️  CAREER TRANSITION PATH")
    print(f"{'='*65}")
    print(f"  FROM:  {transition['current']}")
    print(f"  TO:    {transition['target']}")
    print(f"  Feasibility: {transition['feasibility']}")
    print(f"\\n  📚 Skills to Learn:")
    for skill in transition['skills_to_learn']:
        print(f"     • {skill}")
    if transition["intermediate_roles"]:
        print(f"\\n  🪜 Intermediate Stepping-Stone Roles:")
        for role in transition["intermediate_roles"]:
            print(f"     → {role}")
    print(f"\\n  📈 Job Zone Change: {transition['job_zone_change']:+d} levels")
else:
    print("⚠️  No Zone-2 occupations found in dataset.")
"""))

# ════════════════════════════════════════════════════════════════════════════
cells.append(md("---\n## 12. Final Conclusion & Model Summary"))

cells.append(code("""\
print(f"\\n{'='*65}")
print(f"🏆 FINAL CONCLUSION")
print(f"{'='*65}")
print(f"\\n  Best Model:     {best_name}")
best_r = next(r for r in all_results if r["Model"] == best_name)
print(f"  R² Score:       {best_r['R² Score']:.4f}")
print(f"  RMSE:           {best_r['RMSE']:.4f}")
print(f"  Cosine Sim:     {best_r['Cosine Sim']:.4f}")

print(f"\\n  📊 Model Rankings:")
print(leaderboard.to_string())

print(f"\\n  📁 Deliverables:")
print(f"     ✅ Jupyter Notebook   — Full EDA + ML Pipeline")
print(f"     ✅ Modular Backend    — src/ modules")
print(f"     ✅ Streamlit UI       — streamlit_app/")
print(f"     ✅ Saved Models       — outputs/models/")
print(f"     ✅ Reports & Plots    — outputs/reports/")

print(f"\\n  🎯 System Capabilities:")
print(f"     ✅ Multi-Label RIASEC Prediction")
print(f"     ✅ Hybrid Recommendation (Classifier + Cosine Similarity)")
print(f"     ✅ Skill Gap Analysis")
print(f"     ✅ SHAP Explainability")
print(f"     ✅ Career Transition Path")
print(f"     ✅ K-Means Career Domain Clustering (K={optimal_k})")
"""))

# ════════════════════════════════════════════════════════════════════════════
nb.cells = cells

# Save notebook
output_path = r"D:\adity\career_recommendation_system\notebooks\career_recommendation_system.ipynb"
with open(output_path, "w",encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"✅ Notebook created: {output_path}")