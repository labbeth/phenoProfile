import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    davies_bouldin_score,
)
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# Config
# ============================================================

ROOT = Path(".")
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "output"
PLOTS_DIR = OUT_DIR / "plots"

EMB_PATH = OUT_DIR / "patient_embeddings_all_methods.npz"
DIAG_PATH = DATA_DIR / "diagnosis_synthetic.csv"   # patient_id, diagnosis

N_CLUSTERS = 5
DO_TSNE = True        # Set to False to disable visualization
TSNE_PERPLEXITY = 30
RANDOM_SEED = 42

# ============================================================
# Helpers
# ============================================================

def load_diagnosis_map(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    if "patient_id" not in df.columns or "diagnosis" not in df.columns:
        raise ValueError("Diagnosis CSV must contain patient_id and diagnosis columns.")
    return dict(zip(df["patient_id"].astype(str), df["diagnosis"].astype(str)))


def compute_tsne(Z, seed=RANDOM_SEED, perplexity=30.0):
    tsne = TSNE(
        n_components=2,
        random_state=seed,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(Z)


def evaluate_embedding(Z, labels_true, n_clusters=5):
    """
    Run clustering + compute evaluation metrics.
    Returns a dict.
    """
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED)
    labels_pred = km.fit_predict(Z)

    out = {}

    # Supervised metrics (need ground truth)
    out["ARI"] = adjusted_rand_score(labels_true, labels_pred)
    out["NMI"] = normalized_mutual_info_score(labels_true, labels_pred)

    # Unsupervised metrics
    out["Silhouette"] = silhouette_score(Z, labels_pred)
    out["DaviesBouldin"] = davies_bouldin_score(Z, labels_pred)

    return out, labels_pred


# ============================================================
# Main
# ============================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading embeddings from NPZ:", EMB_PATH)
    data = np.load(EMB_PATH, allow_pickle=True)

    patient_ids = data.get("patient_ids")
    if patient_ids is None:
        raise ValueError("The NPZ file must contain `patient_ids`.")

    patient_ids = patient_ids.astype(str).tolist()
    print(f"Loaded {len(patient_ids)} patients.")

    print("Loading diagnosis ground truth from:", DIAG_PATH)
    diag_map = load_diagnosis_map(DIAG_PATH)

    # Align diagnosis to embeddings
    y_true = []
    missing = 0
    for pid in patient_ids:
        if pid in diag_map:
            y_true.append(diag_map[pid])
        else:
            y_true.append("UNKNOWN")
            missing += 1

    if missing > 0:
        print(f"WARNING: {missing} patients have no diagnosis label. "
              "They will be included under label 'UNKNOWN'.")
    y_true = np.array(y_true)

    # Prepare evaluation
    embedding_keys = [
        "frechet_unw",
        "einstein_unw",
        "frechet_ic",
        "einstein_ic",
        "svd",
        "nmf",
        "autoencoder",
    ]

    results = []

    for key in embedding_keys:
        if key not in data:
            print(f"[SKIP] {key} not found in NPZ.")
            continue

        print(f"\nEvaluating embedding: {key}")
        Z = data[key]

        # Normalization for Euclidean clustering (optional)
        Z = Z.astype(np.float32)
        Z_mean = Z.mean(axis=0, keepdims=True)
        Z_std = Z.std(axis=0, keepdims=True) + 1e-6
        Z_norm = (Z - Z_mean) / Z_std

        # Compute metrics
        metrics, labels_pred = evaluate_embedding(Z_norm, y_true, n_clusters=N_CLUSTERS)

        # Save metrics
        row = {"embedding": key}
        row.update(metrics)
        results.append(row)

        # t-SNE visualization
        if DO_TSNE:
            print(f"  Computing t-SNE for: {key}")
            Z_tsne = compute_tsne(Z_norm, perplexity=TSNE_PERPLEXITY)

            plt.figure(figsize=(7, 6))
            sns.scatterplot(
                x=Z_tsne[:, 0],
                y=Z_tsne[:, 1],
                hue=y_true,
                palette="tab20",
                s=10,
                linewidth=0,
            )
            plt.title(f"{key} – t-SNE (colored by diagnosis)")
            plt.tight_layout()
            fname = PLOTS_DIR / f"tsne_{key}.png"
            plt.savefig(fname, dpi=200)
            plt.close()
            print(f"  Saved t-SNE plot to: {fname}")

    # Save evaluation results
    df_res = pd.DataFrame(results)
    out_csv = OUT_DIR / "evaluation_results.csv"
    df_res.to_csv(out_csv, index=False)
    print("\nSaved evaluation metrics to:", out_csv)

    print("\nDone.")


if __name__ == "__main__":
    main()
