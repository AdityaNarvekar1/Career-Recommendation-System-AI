# 🎯 AI-Powered Career Recommendation System
### Built on O*NET Database | Full ML Pipeline | Streamlit UI

---

## 📁 Project Structure

```
career_recommendation_system/
│
├── config.py                          ← Central config (all paths & constants)
├── requirements.txt                   ← All dependencies
├── generate_notebook.py               ← Run once to generate Jupyter Notebook
│
├── data/
│   ├── raw/                           ← ⬅️  PUT YOUR O*NET FILES HERE
│   └── processed/                     ← Auto-generated (pickle files)
│
├── notebooks/
│   └── career_recommendation_system.ipynb   ← Full EDA + ML Pipeline
│
├── src/
│   ├── utils/
│   │   ├── data_loader.py             ← Load & merge 9 O*NET files
│   │   └── feature_engineering.py    ← 3-group feature vectors
│   ├── models/
│   │   ├── classifiers.py             ← All 11 ML models
│   │   └── clustering.py             ← K-Means + Elbow Method
│   └── engine/
│       ├── recommender.py             ← Hybrid Recommendation Engine
│       └── explainability.py         ← SHAP Explainability Layer
│
├── streamlit_app/
│   └── app.py                         ← Streamlit Web App
│
└── outputs/
    ├── models/                        ← Saved trained models (.pkl)
    └── reports/                       ← All plots & figures (.png)
```

---

## ⚙️ Setup Instructions

### Step 1 — Install Dependencies
```bash
cd career_recommendation_system
pip install -r requirements.txt
```

### Step 2 — Download O*NET Files
Go to: https://www.onetcenter.org/database.html

Download these 9 files and place them in `data/raw/`:

| File Name | Section |
|-----------|---------|
| `Occupation Data.xlsx` | Occupation Titles |
| `Skills.xlsx` | Knowledge, Skills, Abilities |
| `Abilities.xlsx` | Knowledge, Skills, Abilities |
| `Knowledge.xlsx` | Knowledge, Skills, Abilities |
| `Interests.xlsx` | Interests |
| `Work Activities.xlsx` | Work Activities |
| `Work Styles.xlsx` | Work Styles |
| `Job Zones.xlsx` | Education, Experience, Training |
| `Task Statements.xlsx` | Tasks |

### Step 3 — Run the Jupyter Notebook (Train Models)
```bash
jupyter notebook notebooks/career_recommendation_system.ipynb
```
Run all cells top to bottom. This will:
- Load and merge all 9 O*NET files
- Run EDA + feature engineering
- Train all 11 ML models and compare them
- Run K-Means clustering (Elbow Method)
- Save trained models to `outputs/models/`
- Save all plots to `outputs/reports/`

### Step 4 — Launch Streamlit App
```bash
streamlit run streamlit_app/app.py
```

---

## 🤖 ML Pipeline Summary

| Stage | Details |
|-------|---------|
| **Target** | RIASEC multi-label distribution (6 scores, sum to 1) |
| **Features** | Group 1: Skills+Abilities+Knowledge+Styles (numeric) |
| | Group 2: Task Statements → TF-IDF (100 features) |
| | Group 3: Work Activities (weighted by importance) |
| **Preprocessing** | MinMax scaling, median imputation, TF-IDF sublinear |
| **Models** | Ridge, Logistic(OvR), Decision Tree, Extra Trees, Random Forest, MLP, Bagging, AdaBoost, Gradient Boosting, XGBoost, LightGBM |
| **Recommendation** | Hybrid: Classifier → RIASEC Filter → Cosine Similarity Ranking |
| **Clustering** | K-Means with Elbow + Silhouette (auto-selects K) |
| **Explainability** | SHAP values → Natural language explanations |
| **Bonus** | Career Transition Path with feasibility score |

---

## 📊 Key Outputs

- `outputs/reports/eda_riasec_distributions.png` — RIASEC score distributions
- `outputs/reports/model_comparison.png` — All 11 models compared
- `outputs/reports/kmeans_elbow.png` — Elbow + Silhouette method
- `outputs/reports/cluster_visualization.png` — PCA 2D cluster plot
- `outputs/reports/shap_investigative.png` — SHAP feature importance
- `outputs/models/best_classifier.pkl` — Best trained model
- `outputs/models/kmeans_model.pkl` — Trained K-Means model
