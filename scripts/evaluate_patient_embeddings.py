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

# Full-population clustering: number of clusters (can be ≠ #diagnoses)
N_CLUSTERS_FULL = 15

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


def evaluate_embedding_subset(Z, labels_true, n_clusters: int):
    """
    Run clustering + compute evaluation metrics on a subset
    where all labels_true are meaningful (no UNKNOWN).
    Returns:
      metrics: dict
      labels_pred: cluster assignments
    """
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED)
    labels_pred = km.fit_predict(Z)

    out = {}
    out["ARI"] = adjusted_rand_score(labels_true, labels_pred)
    out["NMI"] = normalized_mutual_info_score(labels_true, labels_pred)
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

    # Align diagnosis to embeddings, with cleaning of missing tokens
    MISSING_TOKENS = {"", "UNKNOWN", "unknown", "NaN", "nan", "NA", "N/A", "None"}

    y_true = []
    missing = 0
    for pid in patient_ids:
        raw = diag_map.get(pid, "")
        diag = str(raw).strip()

        if diag in MISSING_TOKENS:
            y_true.append("UNKNOWN")
            missing += 1
        else:
            y_true.append(diag)

    if missing > 0:
        print(
            f"INFO: {missing} patients have no diagnosis label "
            "(marked as 'UNKNOWN')."
        )
    y_true = np.array(y_true)

    # Mask for diagnosed-only subset
    diagnosed_mask = (y_true != "UNKNOWN")
    n_diag = diagnosed_mask.sum()
    unique_diag = np.unique(y_true[diagnosed_mask])
    print(f"Diagnosed patients: {n_diag} (unique diagnoses: {len(unique_diag)})")

    if n_diag < 10 or len(unique_diag) < 2:
        print("WARNING: Very few diagnosed patients or too few diagnosis classes. "
              "Diagnosed-only evaluation may be unstable.")

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

        print(f"\n==============================")
        print(f"Evaluating embedding: {key}")
        print(f"==============================")

        Z = data[key].astype(np.float32)

        # Normalization for Euclidean clustering (optional but standard)
        Z_mean = Z.mean(axis=0, keepdims=True)
        Z_std = Z.std(axis=0, keepdims=True) + 1e-6
        Z_norm = (Z - Z_mean) / Z_std

        # --------------------------------------------------------
        # 1) Diagnosed-only evaluation (Step 1)
        # --------------------------------------------------------
        if n_diag >= 2 and len(unique_diag) >= 2:
            Z_diag = Z_norm[diagnosed_mask]
            y_diag = y_true[diagnosed_mask]
            n_clusters_diag = len(np.unique(y_diag))

            print(f"  [Diag-only] Using n_clusters = #diagnoses = {n_clusters_diag}")

            metrics_diag, labels_diag = evaluate_embedding_subset(
                Z_diag, y_diag, n_clusters=n_clusters_diag
            )

            row_diag = {
                "embedding": key,
                "mode": "diag_only",
                "n_clusters": n_clusters_diag,
            }
            row_diag.update(metrics_diag)
            results.append(row_diag)

            # t-SNE for diagnosed-only patients, colored by diagnosis
            if DO_TSNE:
                print("  [Diag-only] Computing t-SNE...")
                Z_tsne_diag = compute_tsne(Z_diag, perplexity=TSNE_PERPLEXITY)

                plt.figure(figsize=(7, 6))
                sns.scatterplot(
                    x=Z_tsne_diag[:, 0],
                    y=Z_tsne_diag[:, 1],
                    hue=y_diag,
                    palette="tab20",
                    s=10,
                    linewidth=0,
                )
                plt.title(f"{key} – t-SNE (diagnosed only, colored by diagnosis)")
                plt.tight_layout()
                fname = PLOTS_DIR / f"tsne_{key}_diag_only.png"
                plt.savefig(fname, dpi=200)
                plt.close()
                print(f"  [Diag-only] Saved t-SNE plot to: {fname}")
        else:
            print("  [Diag-only] Skipped (not enough diagnosed data).")

        # --------------------------------------------------------
        # 2) Full-pop clustering (Step 2)
        # --------------------------------------------------------
        print(f"  [Full-pop] Clustering all patients with n_clusters = {N_CLUSTERS_FULL}")
        km_full = KMeans(n_clusters=N_CLUSTERS_FULL, random_state=RANDOM_SEED)
        labels_full = km_full.fit_predict(Z_norm)

        # Unsupervised metrics on full population
        sil_full = silhouette_score(Z_norm, labels_full)
        db_full = davies_bouldin_score(Z_norm, labels_full)

        # ARI/NMI computed only over diagnosed patients,
        # but using cluster labels from full-pop clustering
        if n_diag >= 2 and len(unique_diag) >= 2:
            y_diag = y_true[diagnosed_mask]
            labels_full_diag = labels_full[diagnosed_mask]

            ari_full = adjusted_rand_score(y_diag, labels_full_diag)
            nmi_full = normalized_mutual_info_score(y_diag, labels_full_diag)
        else:
            ari_full = np.nan
            nmi_full = np.nan

        row_full = {
            "embedding": key,
            "mode": "full_pop",
            "n_clusters": N_CLUSTERS_FULL,
            "ARI": ari_full,
            "NMI": nmi_full,
            "Silhouette": sil_full,
            "DaviesBouldin": db_full,
        }
        results.append(row_full)

        # t-SNE for full population, colored by cluster ID
        if DO_TSNE:
            print("  [Full-pop] Computing t-SNE...")
            Z_tsne_full = compute_tsne(Z_norm, perplexity=TSNE_PERPLEXITY)

            plt.figure(figsize=(7, 6))
            sns.scatterplot(
                x=Z_tsne_full[:, 0],
                y=Z_tsne_full[:, 1],
                hue=labels_full,
                palette="tab20",
                s=10,
                linewidth=0,
                legend=False,
            )
            plt.title(f"{key} – t-SNE (full population, colored by cluster)")
            plt.tight_layout()
            fname = PLOTS_DIR / f"tsne_{key}_full_clusters.png"
            plt.savefig(fname, dpi=200)
            plt.close()
            print(f"  [Full-pop] Saved t-SNE plot to: {fname}")

        # --------------------------------------------------------
        # 3) Diagnosis overlay on full-pop manifold (third plot)
        # --------------------------------------------------------
        if DO_TSNE and n_diag >= 2 and len(unique_diag) >= 2:
            print("  [Full-pop] Plotting diagnosis overlay on population manifold...")

            # All patients in light grey (background)
            plt.figure(figsize=(7, 6))
            plt.scatter(
                Z_tsne_full[:, 0],
                Z_tsne_full[:, 1],
                c="lightgrey",
                s=6,
                alpha=0.4,
                linewidths=0,
            )

            # Overlay diagnosed patients with colors by diagnosis
            Z_tsne_diag = Z_tsne_full[diagnosed_mask]
            y_diag = y_true[diagnosed_mask]

            sns.scatterplot(
                x=Z_tsne_diag[:, 0],
                y=Z_tsne_diag[:, 1],
                hue=y_diag,
                palette="tab20",
                s=18,
                linewidth=0,
                alpha=0.9,
            )

            plt.title(f"{key} – t-SNE (population with diagnosed overlay)")
            plt.tight_layout()
            fname = PLOTS_DIR / f"tsne_{key}_pop_with_diagnosis.png"
            plt.savefig(fname, dpi=200)
            plt.close()
            print(f"  [Full-pop] Saved diagnosis overlay plot to: {fname}")


    # Save evaluation results
    df_res = pd.DataFrame(results)
    out_csv = OUT_DIR / "evaluation_results.csv"
    df_res.to_csv(out_csv, index=False)
    print("\nSaved evaluation metrics to:", out_csv)

    print("\nDone.")


if __name__ == "__main__":
    main()
