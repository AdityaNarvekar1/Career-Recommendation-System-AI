# src/models/clustering.py — K-Means with Elbow Method + Smart Cluster Labels
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import joblib
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import KMEANS_MAX_K, RANDOM_STATE, KMEANS_MODEL_FILE, REPORTS_DIR, RIASEC_CODES


def find_optimal_k(X: np.ndarray, max_k: int = KMEANS_MAX_K) -> tuple:
    """
    Elbow Method + Silhouette Score to find optimal K.
    Returns (optimal_k, inertias, silhouette_scores, fig)
    """
    print("\n🔍 Finding Optimal K via Elbow Method + Silhouette Score...")
    inertias    = []
    silhouettes = []
    k_range     = range(2, max_k + 1)

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)
        score = silhouette_score(X, km.labels_, sample_size=min(1000, len(X)))
        silhouettes.append(score)
        print(f"  K={k:2d}  Inertia={km.inertia_:.1f}  Silhouette={score:.4f}")

    # Auto-detect elbow via second derivative
    diffs1 = np.diff(inertias)
    diffs2 = np.diff(diffs1)
    elbow_idx = np.argmax(diffs2) + 2  # +2 because k starts at 2
    optimal_k_elbow = list(k_range)[elbow_idx]

    # Also pick best silhouette
    optimal_k_silhouette = list(k_range)[np.argmax(silhouettes)]

    # Choose the one with better silhouette
    optimal_k = optimal_k_silhouette
    print(f"\n  ✅ Elbow suggests K={optimal_k_elbow}, Silhouette suggests K={optimal_k_silhouette}")
    print(f"  🏆 Selected K={optimal_k} (best silhouette)")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("K-Means: Elbow Method & Silhouette Score", fontsize=14, fontweight="bold")

    axes[0].plot(list(k_range), inertias, "bo-", linewidth=2, markersize=8)
    axes[0].axvline(x=optimal_k, color="red", linestyle="--", label=f"Optimal K={optimal_k}")
    axes[0].set_title("Elbow Method (Inertia)")
    axes[0].set_xlabel("Number of Clusters (K)")
    axes[0].set_ylabel("Inertia")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(list(k_range), silhouettes, "go-", linewidth=2, markersize=8)
    axes[1].axvline(x=optimal_k, color="red", linestyle="--", label=f"Optimal K={optimal_k}")
    axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("Number of Clusters (K)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "kmeans_elbow.png", dpi=150, bbox_inches="tight")
    plt.close()

    return optimal_k, inertias, silhouettes, fig


def fit_kmeans(X: np.ndarray, k: int, soc_index: pd.Index) -> tuple:
    """
    Fit final K-Means with optimal K.
    Returns (kmeans_model, cluster_labels_series)
    """
    print(f"\n🔧 Fitting K-Means with K={k}...")
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10, max_iter=300)
    km.fit(X)
    joblib.dump(km, KMEANS_MODEL_FILE)

    cluster_series = pd.Series(km.labels_, index=soc_index, name="cluster")
    return km, cluster_series


def label_clusters(
    km_model: KMeans,
    X: np.ndarray,
    soc_index: pd.Index,
    riasec_labels: pd.DataFrame,
    occupation_meta: pd.DataFrame
) -> dict:
    """
    Auto-label each cluster using dominant RIASEC profile of occupations in it.
    Returns dict: {cluster_id: label_string}
    """
    cluster_assignments = pd.Series(km_model.labels_, index=soc_index, name="cluster")

    # Map RIASEC → domain label
    riasec_to_domain = {
        "Realistic":     "Engineering & Trades",
        "Investigative": "Science & Research",
        "Artistic":      "Arts & Design",
        "Social":        "Healthcare & Education",
        "Enterprising":  "Business & Management",
        "Conventional":  "Finance & Administration",
    }

    cluster_labels = {}
    cluster_summaries = {}

    for cid in sorted(cluster_assignments.unique()):
        socs_in_cluster = cluster_assignments[cluster_assignments == cid].index
        socs_in_riasec  = socs_in_cluster.intersection(riasec_labels.index)

        if len(socs_in_riasec) == 0:
            cluster_labels[cid]    = f"Career Domain {cid}"
            cluster_summaries[cid] = {}
            continue

        # Mean RIASEC scores for occupations in this cluster
        mean_riasec = riasec_labels.loc[socs_in_riasec].mean()
        dominant    = mean_riasec.idxmax()
        label       = riasec_to_domain.get(dominant, f"Domain {cid}")

        cluster_labels[cid]    = label
        cluster_summaries[cid] = mean_riasec.round(3).to_dict()

        # Sample occupations
        sample_titles = []
        if occupation_meta is not None:
            sample_socs   = socs_in_cluster[:3]
            sample_titles = occupation_meta.loc[
                occupation_meta.index.intersection(sample_socs), "title"
            ].tolist()

        print(f"  Cluster {cid:2d}: {label:30s}  "
              f"(dominant={dominant}, n={len(socs_in_cluster)}, "
              f"sample={sample_titles[:2]})")

    return cluster_labels, cluster_summaries


def plot_clusters_2d(X: np.ndarray, labels: np.ndarray,
                     cluster_names: dict, title: str = "Career Clusters") -> None:
    """PCA 2D visualization of clusters."""
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca.fit_transform(X)

    n_clusters = len(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

    fig, ax = plt.subplots(figsize=(12, 8))
    for cid in sorted(set(labels)):
        mask = labels == cid
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=[colors[cid]], label=cluster_names.get(cid, f"Cluster {cid}"),
            alpha=0.6, s=30, edgecolors="none"
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "cluster_visualization.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Cluster plot saved.")
