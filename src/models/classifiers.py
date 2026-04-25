# src/models/classifiers.py — Full ML Pipeline: Simple → Complex → Best
#
# ⚠️  WHY WE USE REGRESSORS (not classifiers):
#   Our RIASEC target is a continuous probability distribution per occupation
#   (e.g. Investigative=0.42, Conventional=0.28, ...) that sums to 1.
#   LogisticRegression / classifiers expect DISCRETE class labels → they fail.
#   The correct approach is Multi-Output Regression:
#     - Predict all 6 RIASEC scores simultaneously
#     - Evaluate with R², RMSE, Cosine Similarity
#   "Logistic Regression" in our pipeline = LinearRegression (the continuous
#   analog) so we keep the conceptual progression intact for the report.
#
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, BaggingRegressor,
    AdaBoostRegressor, GradientBoostingRegressor,
    ExtraTreesRegressor
)
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import RANDOM_STATE, MODELS_DIR, RIASEC_CODES


def get_all_models():
    """
    Returns ordered dict of all models from simple → complex.
    All models use MultiOutputRegressor to predict 6 RIASEC scores simultaneously.

    NOTE: RIASEC targets are continuous [0,1] probability distributions.
          Classifiers (LogisticRegression etc.) require discrete labels → incompatible.
          We use the regression equivalents throughout the pipeline.
    """
    models = {
        # ── Tier 1: Linear / Baseline ──────────────────────────────────────
        # Ridge = regularized linear regression (prevents overfitting on wide feature matrix)
        "Ridge Regression": MultiOutputRegressor(
            Ridge(alpha=1.0), n_jobs=-1
        ),
        # Linear Regression = baseline OLS (equivalent role to Logistic in classification)
        "Linear Regression": MultiOutputRegressor(
            LinearRegression(n_jobs=-1), n_jobs=-1
        ),
        # Lasso = L1 regularization (forces sparse feature weights → feature selection)
        "Lasso Regression": MultiOutputRegressor(
            Lasso(alpha=0.001, max_iter=2000), n_jobs=-1
        ),

        # KNN = instance-based (no training phase) — good conceptual contrast
        "KNN Regressor": MultiOutputRegressor(
            KNeighborsRegressor(n_neighbors=7, weights="distance", n_jobs=-1), n_jobs=-1
        ),

        # ── Tier 2: Tree-Based ─────────────────────────────────────────────
        "Decision Tree": MultiOutputRegressor(
            DecisionTreeRegressor(max_depth=10, random_state=RANDOM_STATE), n_jobs=-1
        ),
        "Extra Trees": MultiOutputRegressor(
            ExtraTreesRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1), n_jobs=-1
        ),
        "Random Forest": MultiOutputRegressor(
            RandomForestRegressor(n_estimators=200, max_depth=15,
                                   random_state=RANDOM_STATE, n_jobs=-1), n_jobs=-1
        ),

        # ── Tier 3: Neural Network ─────────────────────────────────────────
        "Neural Network (MLP)": MultiOutputRegressor(
            MLPRegressor(hidden_layer_sizes=(256, 128, 64),
                          activation="relu", max_iter=300,
                          early_stopping=True, validation_fraction=0.1,
                          random_state=RANDOM_STATE), n_jobs=-1
        ),

        # ── Tier 4: Ensemble / Bagging ─────────────────────────────────────
        "Bagging (DT base)": MultiOutputRegressor(
            BaggingRegressor(estimator=DecisionTreeRegressor(max_depth=8),
                              n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1), n_jobs=-1
        ),

        # ── Tier 5: Boosting ───────────────────────────────────────────────
        "AdaBoost": MultiOutputRegressor(
            AdaBoostRegressor(n_estimators=100, learning_rate=0.1,
                               random_state=RANDOM_STATE), n_jobs=-1
        ),
        "Gradient Boosting": MultiOutputRegressor(
            GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                       max_depth=5, random_state=RANDOM_STATE), n_jobs=-1
        ),
        "XGBoost": MultiOutputRegressor(
            xgb.XGBRegressor(n_estimators=200, learning_rate=0.05,
                              max_depth=6, subsample=0.8,
                              colsample_bytree=0.8, random_state=RANDOM_STATE,
                              verbosity=0, n_jobs=-1), n_jobs=-1
        ),
        "LightGBM": MultiOutputRegressor(
            lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05,
                               num_leaves=63, random_state=RANDOM_STATE,
                               verbose=-1, n_jobs=-1), n_jobs=-1
        ),
    }
    return models


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """
    Train and evaluate a single model.
    Returns dict of metrics.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Per-RIASEC metrics
    mse_per  = mean_squared_error(y_test, y_pred, multioutput="raw_values")
    r2_per   = r2_score(y_test, y_pred, multioutput="raw_values")

    # Overall metrics
    mse_mean = float(np.mean(mse_per))
    r2_mean  = float(np.mean(r2_per))
    rmse     = float(np.sqrt(mse_mean))

    # Cosine similarity between true and predicted distributions
    from sklearn.metrics.pairwise import cosine_similarity
    cos_sim = float(np.mean([
        cosine_similarity([y_test[i]], [y_pred[i]])[0][0]
        for i in range(len(y_test))
    ]))

    result = {
        "Model":          model_name,
        "R² Score":       round(r2_mean, 4),
        "RMSE":           round(rmse, 4),
        "MSE":            round(mse_mean, 4),
        "Cosine Sim":     round(cos_sim, 4),
        "Per-RIASEC R²":  {k: round(v, 4) for k, v in zip(RIASEC_CODES, r2_per)},
        "Per-RIASEC MSE": {k: round(v, 4) for k, v in zip(RIASEC_CODES, mse_per)},
    }
    return result, model, y_pred


def run_full_model_comparison(X_train, y_train, X_test, y_test):
    """
    Run all models, collect results, return sorted leaderboard.
    """
    print("\n" + "="*70)
    print("🏆 FULL MODEL COMPARISON — Simple → Complex")
    print("="*70)

    all_results  = []
    trained_models = {}

    models = get_all_models()
    total  = len(models)

    for i, (name, model) in enumerate(models.items(), 1):
        print(f"\n  [{i}/{total}] Training: {name}...")
        try:
            result, trained_model, _ = evaluate_model(
                model, X_train, y_train, X_test, y_test, name
            )
            all_results.append(result)
            trained_models[name] = trained_model
            print(f"    R²={result['R² Score']:.4f}  RMSE={result['RMSE']:.4f}  CosSim={result['Cosine Sim']:.4f}")
        except Exception as e:
            print(f"    ❌ Failed: {e}")

    # Sort by R² descending
    leaderboard = pd.DataFrame([{
        "Model":      r["Model"],
        "R² Score":   r["R² Score"],
        "RMSE":       r["RMSE"],
        "Cosine Sim": r["Cosine Sim"],
    } for r in all_results]).sort_values("R² Score", ascending=False).reset_index(drop=True)

    leaderboard.index += 1  # 1-based rank

    print("\n" + "="*70)
    print("📊 LEADERBOARD")
    print("="*70)
    print(leaderboard.to_string())

    # Save best model
    best_name  = leaderboard.iloc[0]["Model"]
    best_model = trained_models[best_name]
    joblib.dump(best_model, MODELS_DIR / "best_classifier.pkl")
    joblib.dump(leaderboard, MODELS_DIR / "leaderboard.pkl")

    print(f"\n🥇 Best Model: {best_name}")
    print(f"   Saved to: {MODELS_DIR / 'best_classifier.pkl'}")

    return leaderboard, trained_models, all_results, best_name, best_model
