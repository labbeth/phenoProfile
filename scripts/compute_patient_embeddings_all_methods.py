import pickle
from pathlib import Path

import csv
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import TruncatedSVD, NMF
from scipy.spatial.distance import pdist

from hierarchy_transformers import HierarchyTransformer


# ============================================================
# Config
# ============================================================

EMB_DIR = Path(".")
DATA_DIR = EMB_DIR / "data"

HPO_EMB_PATH = DATA_DIR / "hpo_embeddings.npy"
META_PATH = DATA_DIR / "embeddings_metadata.pkl"
PATIENT_CSV_PATH = DATA_DIR / "binary_matrix.csv"     # patient × HPO binary matrix
OUTPUT_EMB_PATH = EMB_DIR / "output" / "patient_embeddings_all_methods.npz"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Classical DR dims
SVD_DIM = 64
NMF_DIM = 64

# How many singular values to analyze for spectrum
SVD_SPECTRUM_MAX_DIM = 200

RANDOM_SEED = 42

# Fréchet mean optimisation
FRECHET_MAX_ITER = 100
FRECHET_LR = 0.1
FRECHET_TOL = 1e-5


# ============================================================
# Data loading
# ============================================================

def load_embeddings_and_metadata(hpo_emb_path, meta_path):
    hpo_embeddings = np.load(hpo_emb_path)
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)
    return hpo_embeddings, metadata


def load_and_align_patient_matrix(csv_path: Path, metadata) -> tuple[np.ndarray, list[str] | None]:
    """
    Load patient x phenotype matrix from CSV and align columns to the
    HPO order used in hpo_embeddings (metadata["hpo_ids"]).

    CSV assumptions:
      - One row per patient.
      - One column per HPO term (header = HPO ID).
      - Optionally, first column = patient_id (string or 'patient_id' / 'id').

    Returns:
      patient_matrix: (n_patients, n_hpo_total) in the same order as metadata["hpo_ids"]
      patient_ids: list of patient IDs or None
    """
    df = pd.read_csv(csv_path)

    # Detect patient ID column
    patient_ids = None
    first_col = df.columns[0]
    if first_col.lower() in ("patient_id", "id") or not np.issubdtype(df.dtypes[0], np.number):
        patient_ids = df.iloc[:, 0].astype(str).tolist()
        df = df.iloc[:, 1:]  # drop ID column

    # Remaining columns are assumed to be HPO IDs
    csv_hpo_ids = list(df.columns)

    # HPO IDs used in embeddings
    emb_hpo_ids = metadata["hpo_ids"]  # list of length n_hpo_total
    emb_index = {hpo_id: j for j, hpo_id in enumerate(emb_hpo_ids)}

    # Find intersection between CSV HPOs and embedding HPOs
    matched_cols = [h for h in csv_hpo_ids if h in emb_index]
    unmatched_cols = [h for h in csv_hpo_ids if h not in emb_index]

    print(f"\nCSV HPO columns: {len(csv_hpo_ids)}")
    print(f"Matched HPO columns with embeddings: {len(matched_cols)}")
    print(f"Unmatched HPO columns (ignored): {len(unmatched_cols)}")
    if unmatched_cols:
        print("  Example unmatched HPO IDs:", unmatched_cols[:10])

    # Extract only matched columns
    df_matched = df[matched_cols]
    patient_matrix_small = df_matched.to_numpy(dtype=int)   # (n_patients, n_matched)
    n_patients = patient_matrix_small.shape[0]
    n_hpo_total = len(emb_hpo_ids)

    # Build full matrix aligned with embedding order
    patient_matrix_full = np.zeros((n_patients, n_hpo_total), dtype=int)
    for k, hpo_id in enumerate(matched_cols):
        j = emb_index[hpo_id]  # column index in embedding space
        patient_matrix_full[:, j] = patient_matrix_small[:, k]

    return patient_matrix_full, patient_ids


# ============================================================
# IC weights
# ============================================================

def compute_ic_weights(patient_matrix: np.ndarray) -> np.ndarray:
    """
    Compute Information Content (IC) per phenotype based on patient frequency.

    patient_matrix: (n_patients, n_phenotypes), 0/1
    Returns: ic (n_phenotypes,) with IC_j = -log p_j, p_j = freq of phenotype j.
    Phenotypes never seen in any patient get IC = 0.
    """
    n_patients, n_phenos = patient_matrix.shape
    counts = patient_matrix.sum(axis=0)            # (n_phenos,)
    p = counts / n_patients                        # empirical probability

    ic = np.zeros(n_phenos, dtype=np.float32)
    mask = counts > 0
    ic[mask] = -np.log(p[mask] + 1e-9)             # avoid log(0)
    return ic


# ============================================================
# Hyperbolic aggregation: Fréchet mean & Einstein midpoint
# ============================================================

def frechet_mean_manual(
    ball,
    xs: torch.Tensor,                 # (n_points, dim)
    weights: torch.Tensor | None = None,     # (n_points,)
    max_iter: int = FRECHET_MAX_ITER,
    lr: float = FRECHET_LR,
    tol: float = FRECHET_TOL,
) -> torch.Tensor:
    """
    Fréchet (Karcher) mean on the Poincaré ball using HT's manifold ops only.

    Minimize F(m) = sum_i w_i * d(m, x_i)^2 via gradient descent:
      grad F(m) ∝ sum_i w_i * log_m(x_i)
    """
    device = xs.device
    n, dim = xs.shape

    if n == 0:
        raise ValueError("frechet_mean_manual: xs is empty.")
    if n == 1:
        return xs[0]

    if weights is None:
        w = torch.ones(n, device=device, dtype=xs.dtype) / n
    else:
        w = weights.to(device=device, dtype=xs.dtype)
        w = w / (w.sum() + 1e-9)

    # Initialize at origin
    m = ball.origin(dim, device=device, dtype=xs.dtype).squeeze(0)

    for _ in range(max_iter):
        m_expanded = m.unsqueeze(0).expand_as(xs)      # (n, dim)

        # log_m(x_i) in tangent space at m
        logs = ball.logmap(m_expanded, xs)            # (n, dim)

        # gradient (up to const factor): sum_i w_i * log_m(x_i)
        grad = (w.unsqueeze(1) * logs).sum(dim=0)     # (dim,)

        # step size in tangent space
        step = lr * grad

        # norm of step in tangent metric
        step_norm = ball.norm(m.unsqueeze(0), step.unsqueeze(0)).item()
        if step_norm < tol:
            break

        # update: m ← exp_m(step)
        m = ball.expmap(m, step)

    return m


def einstein_midpoint_manual(
    ball,
    xs: torch.Tensor,                 # (n_points, dim)
    weights: torch.Tensor | None = None,     # (n_points,)
) -> torch.Tensor:
    """
    Einstein midpoint / relativistic barycenter in Poincaré coordinates,
    via lambda_x (conformal factor).
    """
    device = xs.device
    n, dim = xs.shape

    if n == 0:
        raise ValueError("einstein_midpoint_manual: xs is empty.")
    if n == 1:
        return xs[0]

    if weights is None:
        w = torch.ones(n, device=device, dtype=xs.dtype) / n
    else:
        w = weights.to(device=device, dtype=xs.dtype)
        w = w / (w.sum() + 1e-9)

    # lambda_x: (n, 1) – conformal factor
    lam = ball.lambda_x(xs, dim=-1, keepdim=True)   # λ(x) = 2/(1 - c||x||^2)
    gamma = lam / 2.0                               # γ(x) = λ(x)/2

    w = w.view(-1, 1)                               # (n, 1)
    gw = gamma * w                                  # (n, 1)

    num = (gw * xs).sum(dim=0)                      # (dim,)
    denom = gw.sum(dim=0) + 1e-9                    # scalar in a 1D tensor
    m = num / denom

    # project back to the ball to be safe
    m = ball.projx(m)

    return m


def compute_patient_embeddings(
    manifold,
    patient_matrix: np.ndarray,         # (n_patients, n_phenotypes) 0/1
    pheno_embeddings: torch.Tensor,     # (n_phenotypes, dim)
    method: str = "frechet",
    weights_matrix: np.ndarray | None = None,  # optional (n_patients, n_phenotypes)
    max_iter: int = FRECHET_MAX_ITER,
) -> torch.Tensor:
    """
    Aggregate phenotype embeddings into per-patient embeddings.

    method: "frechet" or "einstein"
    """
    n_patients, n_phenotypes = patient_matrix.shape
    dim = pheno_embeddings.shape[1]
    device = pheno_embeddings.device
    dtype = pheno_embeddings.dtype

    patient_embs = torch.zeros(n_patients, dim, device=device, dtype=dtype)

    for i in range(n_patients):
        mask = patient_matrix[i].astype(bool)
        idx = np.where(mask)[0]

        if idx.size == 0:
            # patient with no phenotypes: put at origin
            patient_embs[i] = manifold.origin(dim, device=device, dtype=dtype).squeeze(0)
            continue

        xs = pheno_embeddings[idx]            # (k_i, dim)

        w_i = None
        if weights_matrix is not None:
            w_i = torch.from_numpy(weights_matrix[i, idx]).to(device=device, dtype=dtype)

        if method == "frechet":
            patient_embs[i] = frechet_mean_manual(
                ball=manifold,
                xs=xs,
                weights=w_i,
                max_iter=max_iter,
            )
        elif method == "einstein":
            patient_embs[i] = einstein_midpoint_manual(
                ball=manifold,
                xs=xs,
                weights=w_i,
            )
        else:
            raise ValueError(f"Unknown method: {method}")

    return patient_embs


# ============================================================
# Classical DR: SVD & NMF
# ============================================================

def compute_svd_embeddings(patient_matrix: np.ndarray, dim: int, seed: int) -> np.ndarray:
    svd = TruncatedSVD(n_components=dim, random_state=seed)
    Z = svd.fit_transform(patient_matrix)  # (n_patients, dim)
    return Z


def compute_nmf_embeddings(patient_matrix: np.ndarray, dim: int, seed: int) -> np.ndarray:
    nmf = NMF(n_components=dim, random_state=seed, init="nndsvda", max_iter=200)
    Z = nmf.fit_transform(patient_matrix)  # (n_patients, dim)
    return Z


# ============================================================
# Statistics
# ============================================================
def radial_stats(manifold, Z: torch.Tensor):
    """Return mean, std, min, max of hyperbolic radius (dist to origin)."""
    with torch.no_grad():
        r = manifold.dist0(Z).cpu().numpy()
    return dict(mean=float(r.mean()), std=float(r.std()),
                min=float(r.min()), max=float(r.max()))


def vector_correlation(a: np.ndarray, b: np.ndarray):
    """Pearson correlation between 2 vectors."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.ndim > 1:
        a = a.ravel()
    if b.ndim > 1:
        b = b.ravel()
    if a.size != b.size or a.size == 0:
        return np.nan
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def write_stats_csv(path: Path, rows: list[dict]):
    """
    Write a list of statistics dictionaries to CSV.
    Expected keys in each row:
      - metric
      - value
      - description
      - interpretation
    """
    if not rows:
        print("No statistics to save.")
        return
    keys = ["metric", "value", "description", "interpretation"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved statistics CSV: {path}")


def pairwise_distance_correlation(Z1: np.ndarray,
                                  Z2: np.ndarray,
                                  n_samples: int = 3000,
                                  seed: int = 42) -> float:
    """
    Correlation between pairwise Euclidean distance structures of two embeddings.

    We:
      - subsample up to n_samples patients
      - compute pdist() (condensed distance vector) for each embedding
      - compute Pearson correlation between the two distance vectors
    """
    Z1 = np.asarray(Z1)
    Z2 = np.asarray(Z2)
    n = Z1.shape[0]
    if Z2.shape[0] != n or n == 0:
        return np.nan

    m = min(n_samples, n)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=m, replace=False)

    A = Z1[idx]
    B = Z2[idx]

    d1 = pdist(A, metric="euclidean")
    d2 = pdist(B, metric="euclidean")

    if d1.std() == 0 or d2.std() == 0:
        return np.nan
    return float(np.corrcoef(d1, d2)[0, 1])


def analyze_svd_spectrum(
    patient_matrix: np.ndarray,
    max_components: int,
    seed: int,
) -> dict:
    """
    Fit a TruncatedSVD with up to `max_components` components and
    analyze the singular value spectrum.

    Returns a dict with:
      - k_90: dimension for >=90% explained variance
      - k_95: dimension for >=95% explained variance
      - elbow_k: index of largest singular value drop (1-based)
    """
    from sklearn.decomposition import TruncatedSVD

    n_features = patient_matrix.shape[1]
    n_components = min(max_components, n_features - 1)

    print(f"\nAnalyzing SVD spectrum with n_components={n_components}...")
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    svd.fit(patient_matrix)

    singular_values = svd.singular_values_
    var_ratio = svd.explained_variance_ratio_
    cum_var = np.cumsum(var_ratio)

    # Print top singular values
    print("\nTop 20 SVD singular values:")
    for i, sv in enumerate(singular_values[:20]):
        print(f"  σ[{i+1:02d}] = {sv:.4f}")

    # Variance thresholds
    if np.any(cum_var >= 0.90):
        k_90 = int(np.argmax(cum_var >= 0.90) + 1)
    else:
        k_90 = n_components

    if np.any(cum_var >= 0.95):
        k_95 = int(np.argmax(cum_var >= 0.95) + 1)
    else:
        k_95 = n_components

    print("\nCumulative variance:")
    print(f"  90% variance at k = {k_90}")
    print(f"  95% variance at k = {k_95}")

    # Elbow via largest drop σ_i - σ_{i+1}
    if len(singular_values) > 1:
        diffs = singular_values[:-1] - singular_values[1:]
        elbow_k = int(np.argmax(diffs) + 1)
    else:
        elbow_k = 1

    print(f"\nElbow point (largest drop): k = {elbow_k}")

    return {
        "k_90": k_90,
        "k_95": k_95,
        "elbow_k": elbow_k,
    }


# ============================================================
# Main
# ============================================================

def main():
    # 1. Load HPO embeddings and metadata
    print(f"Loading HPO embeddings from: {HPO_EMB_PATH}")
    hpo_embeddings_np, metadata = load_embeddings_and_metadata(HPO_EMB_PATH, META_PATH)
    n_hpo, dim = hpo_embeddings_np.shape
    print(f"HPO embeddings shape: {hpo_embeddings_np.shape}")

    model_path = DATA_DIR / "HiT-all-MiniLM-L12-v2-hpo-hpo_datasets_syn_multi_random" / "final"

    # 2. Load HiT model to get the manifold
    print(f"Loading HierarchyTransformer model from: {model_path}")
    model = HierarchyTransformer.from_pretrained(model_path)
    model.to(DEVICE)
    model.eval()
    manifold = model.manifold
    print("Manifold:", manifold)

    # 3. Convert HPO embeddings to torch tensor on the same device
    hpo_embeddings = torch.from_numpy(hpo_embeddings_np).to(DEVICE).to(torch.float32)

    # 4. Load patient matrix from CSV and align to HPO order
    print(f"\nLoading patient matrix from: {PATIENT_CSV_PATH}")
    patient_matrix, patient_ids = load_and_align_patient_matrix(PATIENT_CSV_PATH, metadata)
    n_patients, n_phenos_mat = patient_matrix.shape
    print(f"Aligned patient matrix shape: {patient_matrix.shape}")

    if n_phenos_mat != n_hpo:
        raise ValueError(
            f"Number of phenotypes in matrix ({n_phenos_mat}) does not match "
            f"HPO embeddings ({n_hpo})."
        )

    # 5. Compute IC weights
    print("\nComputing IC weights...")
    ic = compute_ic_weights(patient_matrix)      # (n_phenotypes,)
    weights_matrix_ic = patient_matrix * ic[np.newaxis, :]   # (n_patients, n_phenotypes)

    # 6. Hyperbolic embeddings
    print("\nComputing HYPERBOLIC patient embeddings...")

    print("  Fréchet UNWEIGHTED...")
    frechet_unw = compute_patient_embeddings(
        manifold=manifold,
        patient_matrix=patient_matrix,
        pheno_embeddings=hpo_embeddings,
        method="frechet",
        weights_matrix=None,
        max_iter=FRECHET_MAX_ITER,
    )

    print("  Einstein UNWEIGHTED...")
    einstein_unw = compute_patient_embeddings(
        manifold=manifold,
        patient_matrix=patient_matrix,
        pheno_embeddings=hpo_embeddings,
        method="einstein",
        weights_matrix=None,
    )

    print("  Fréchet IC-WEIGHTED...")
    frechet_ic = compute_patient_embeddings(
        manifold=manifold,
        patient_matrix=patient_matrix,
        pheno_embeddings=hpo_embeddings,
        method="frechet",
        weights_matrix=weights_matrix_ic,
        max_iter=FRECHET_MAX_ITER,
    )

    print("  Einstein IC-WEIGHTED...")
    einstein_ic = compute_patient_embeddings(
        manifold=manifold,
        patient_matrix=patient_matrix,
        pheno_embeddings=hpo_embeddings,
        method="einstein",
        weights_matrix=weights_matrix_ic,
    )

    # 7. Classical embeddings (SVD, NMF)
    print("\nComputing classical (SVD, NMF) embeddings...")
    Z_svd = compute_svd_embeddings(patient_matrix, dim=SVD_DIM, seed=RANDOM_SEED)
    Z_nmf = compute_nmf_embeddings(patient_matrix, dim=NMF_DIM, seed=RANDOM_SEED)

    # 8. Save everything
    print(f"\nSaving all embeddings to: {OUTPUT_EMB_PATH}")
    np.savez_compressed(
        OUTPUT_EMB_PATH,
        patient_ids=np.array(patient_ids) if patient_ids is not None else np.array([]),
        hpo_ids=np.array(metadata["hpo_ids"]),
        ic=ic,
        svd=Z_svd,
        nmf=Z_nmf,
        frechet_unw=frechet_unw.cpu().numpy(),
        einstein_unw=einstein_unw.cpu().numpy(),
        frechet_ic=frechet_ic.cpu().numpy(),
        einstein_ic=einstein_ic.cpu().numpy(),
    )

    print("\n=== Computing statistics for all 6 embedding methods ===")

    stats_rows: list[dict] = []

    # --------------------------------------------------------
    # 1) Hyperbolic radial statistics
    # --------------------------------------------------------
    frechet_unw_cpu = frechet_unw.cpu()
    einstein_unw_cpu = einstein_unw.cpu()
    frechet_ic_cpu = frechet_ic.cpu()
    einstein_ic_cpu = einstein_ic.cpu()

    rad_frechet_unw = radial_stats(manifold, frechet_unw_cpu)
    rad_einstein_unw = radial_stats(manifold, einstein_unw_cpu)
    rad_frechet_ic = radial_stats(manifold, frechet_ic_cpu)
    rad_einstein_ic = radial_stats(manifold, einstein_ic_cpu)

    print("\nHyperbolic radial statistics:")
    print(f"  Frechet UNW : {rad_frechet_unw}")
    print(f"  Einstein UNW: {rad_einstein_unw}")
    print(f"  Frechet ICW : {rad_frechet_ic}")
    print(f"  Einstein ICW: {rad_einstein_ic}")

    stats_rows.extend([
        {
            "metric": "radial_mean_Frechet_UNW",
            "value": rad_frechet_unw["mean"],
            "description": "Mean hyperbolic distance to origin for Fréchet mean (unweighted)",
            "interpretation": "Higher values indicate more specific or extreme patient profiles in hyperbolic space.",
        },
        {
            "metric": "radial_mean_Einstein_UNW",
            "value": rad_einstein_unw["mean"],
            "description": "Mean hyperbolic distance to origin for Einstein midpoint (unweighted)",
            "interpretation": "Slightly larger than Fréchet UNW typically, reflecting a more outward barycenter.",
        },
        {
            "metric": "radial_mean_Frechet_ICW",
            "value": rad_frechet_ic["mean"],
            "description": "Mean hyperbolic distance to origin for Fréchet mean with IC weights",
            "interpretation": "IC weighting pushes patients outward according to phenotype specificity.",
        },
        {
            "metric": "radial_mean_Einstein_ICW",
            "value": rad_einstein_ic["mean"],
            "description": "Mean hyperbolic distance to origin for Einstein midpoint with IC weights",
            "interpretation": "Combines Einstein barycenter with IC; typically the most radial of the four.",
        },
    ])

    # For radial correlations
    rFU = manifold.dist0(frechet_unw_cpu).cpu().numpy()
    rEU = manifold.dist0(einstein_unw_cpu).cpu().numpy()
    rFI = manifold.dist0(frechet_ic_cpu).cpu().numpy()
    rEI = manifold.dist0(einstein_ic_cpu).cpu().numpy()

    corr_FU_FI = vector_correlation(rFU, rFI)
    corr_EU_EI = vector_correlation(rEU, rEI)

    print("\nRadial correlations (UNW vs ICW):")
    print(f"  Corr(Frechet UNW, Frechet ICW)  = {corr_FU_FI:.6f}")
    print(f"  Corr(Einstein UNW, Einstein ICW) = {corr_EU_EI:.6f}")

    stats_rows.extend([
        {
            "metric": "corr_Frechet_UNW_vs_ICW",
            "value": corr_FU_FI,
            "description": "Correlation between hyperbolic radii of Fréchet UNW and Fréchet ICW embeddings",
            "interpretation": "Close to 1 means IC weighting preserves the relative ordering of patients by specificity.",
        },
        {
            "metric": "corr_Einstein_UNW_vs_ICW",
            "value": corr_EU_EI,
            "description": "Correlation between hyperbolic radii of Einstein UNW and Einstein ICW embeddings",
            "interpretation": "High correlation indicates that IC mainly rescales specificity rather than changing rank.",
        },
    ])

    # --------------------------------------------------------
    # 2) L2 embedding differences between hyperbolic variants
    # --------------------------------------------------------
    l2_FU_EU = float((frechet_unw_cpu - einstein_unw_cpu).norm(dim=1).mean())
    l2_FI_EI = float((frechet_ic_cpu - einstein_ic_cpu).norm(dim=1).mean())
    l2_FU_FI = float((frechet_unw_cpu - frechet_ic_cpu).norm(dim=1).mean())
    l2_EU_EI = float((einstein_unw_cpu - einstein_ic_cpu).norm(dim=1).mean())

    print("\nMean L2 distances (hyperbolic embeddings in ambient space):")
    print(f"  Frechet UNW vs Einstein UNW   = {l2_FU_EU:.6f}")
    print(f"  Frechet ICW vs Einstein ICW   = {l2_FI_EI:.6f}")
    print(f"  Frechet UNW vs Frechet ICW    = {l2_FU_FI:.6f}")
    print(f"  Einstein UNW vs Einstein ICW  = {l2_EU_EI:.6f}")

    stats_rows.extend([
        {
            "metric": "l2_Frechet_UNW_vs_Einstein_UNW",
            "value": l2_FU_EU,
            "description": "Mean L2 difference between Fréchet and Einstein (unweighted) embeddings",
            "interpretation": "Small value indicates Einstein midpoint closely approximates the Fréchet mean.",
        },
        {
            "metric": "l2_Frechet_ICW_vs_Einstein_ICW",
            "value": l2_FI_EI,
            "description": "Mean L2 difference between Fréchet and Einstein (IC-weighted) embeddings",
            "interpretation": "Again measures how close the two barycenters are under IC weighting.",
        },
        {
            "metric": "l2_Frechet_UNW_vs_Frechet_ICW",
            "value": l2_FU_FI,
            "description": "Mean L2 difference between Fréchet embeddings with and without IC weighting",
            "interpretation": "Quantifies how much IC weighting changes patient positions for Fréchet.",
        },
        {
            "metric": "l2_Einstein_UNW_vs_Einstein_ICW",
            "value": l2_EU_EI,
            "description": "Mean L2 difference between Einstein embeddings with and without IC weighting",
            "interpretation": "Shows how strongly IC affects Einstein-based patient embeddings.",
        },
    ])

    # --------------------------------------------------------
    # 3) Cross-method comparisons: hyperbolic vs SVD/NMF
    # --------------------------------------------------------
    Z_svd_np = Z_svd  # already numpy
    Z_nmf_np = Z_nmf  # already numpy
    Z_FU_np = frechet_unw_cpu.numpy()
    Z_EU_np = einstein_unw_cpu.numpy()

    # 3a. Pairwise distance correlations (Euclidean distances in embedding space)
    print("\nPairwise distance correlations (Euclidean geometry):")
    corr_D_FU_SVD = pairwise_distance_correlation(Z_FU_np, Z_svd_np, n_samples=3000, seed=RANDOM_SEED)
    corr_D_EU_SVD = pairwise_distance_correlation(Z_EU_np, Z_svd_np, n_samples=3000, seed=RANDOM_SEED)
    corr_D_FU_NMF = pairwise_distance_correlation(Z_FU_np, Z_nmf_np, n_samples=3000, seed=RANDOM_SEED)
    corr_D_EU_NMF = pairwise_distance_correlation(Z_EU_np, Z_nmf_np, n_samples=3000, seed=RANDOM_SEED)

    print(f"  Corr(pairwise D: Frechet UNW vs SVD)   = {corr_D_FU_SVD:.4f}")
    print(f"  Corr(pairwise D: Einstein UNW vs SVD)  = {corr_D_EU_SVD:.4f}")
    print(f"  Corr(pairwise D: Frechet UNW vs NMF)   = {corr_D_FU_NMF:.4f}")
    print(f"  Corr(pairwise D: Einstein UNW vs NMF)  = {corr_D_EU_NMF:.4f}")

    stats_rows.extend([
        {
            "metric": "corr_pairwise_FrechetUNW_vs_SVD",
            "value": corr_D_FU_SVD,
            "description": "Correlation between pairwise Euclidean distances of Fréchet UNW and SVD embeddings",
            "interpretation": "Measures how similarly hyperbolic Fréchet and SVD organize patient distances.",
        },
        {
            "metric": "corr_pairwise_EinsteinUNW_vs_SVD",
            "value": corr_D_EU_SVD,
            "description": "Correlation between pairwise Euclidean distances of Einstein UNW and SVD embeddings",
            "interpretation": "High values indicate similar patient geometry between Einstein and SVD.",
        },
        {
            "metric": "corr_pairwise_FrechetUNW_vs_NMF",
            "value": corr_D_FU_NMF,
            "description": "Correlation between pairwise Euclidean distances of Fréchet UNW and NMF embeddings",
            "interpretation": "Shows how close Fréchet hyperbolic geometry is to NMF topic geometry.",
        },
        {
            "metric": "corr_pairwise_EinsteinUNW_vs_NMF",
            "value": corr_D_EU_NMF,
            "description": "Correlation between pairwise Euclidean distances of Einstein UNW and NMF embeddings",
            "interpretation": "Shows whether Einstein embedding and NMF induce similar global patient distances.",
        },
    ])

    # 3b. Norm correlations: hyperbolic radius vs Euclidean norms
    norm_svd = np.linalg.norm(Z_svd_np, axis=1)
    norm_nmf = np.linalg.norm(Z_nmf_np, axis=1)

    corr_radFU_normSVD = vector_correlation(rFU, norm_svd)
    corr_radFU_normNMF = vector_correlation(rFU, norm_nmf)

    print("\nCorrelation of 'complexity' measures (radius vs Euclidean norms):")
    print(f"  Corr(radius Frechet UNW, ||SVD||) = {corr_radFU_normSVD:.4f}")
    print(f"  Corr(radius Frechet UNW, ||NMF||) = {corr_radFU_normNMF:.4f}")

    stats_rows.extend([
        {
            "metric": "corr_radius_FrechetUNW_vs_norm_SVD",
            "value": corr_radFU_normSVD,
            "description": "Correlation between hyperbolic radius (Fréchet UNW) and Euclidean norm of SVD embeddings",
            "interpretation": "Indicates whether SVD norm tracks clinical specificity encoded in hyperbolic radius.",
        },
        {
            "metric": "corr_radius_FrechetUNW_vs_norm_NMF",
            "value": corr_radFU_normNMF,
            "description": "Correlation between hyperbolic radius (Fréchet UNW) and Euclidean norm of NMF embeddings",
            "interpretation": "Shows if NMF activation magnitude reflects phenotype specificity.",
        },
    ])

    # --------------------------------------------------------
    # 4) SVD spectrum analysis for AE latent dimension selection
    # --------------------------------------------------------
    svd_spec = analyze_svd_spectrum(
        patient_matrix=patient_matrix.astype(float),
        max_components=SVD_SPECTRUM_MAX_DIM,
        seed=RANDOM_SEED,
    )

    stats_rows.extend([
        {
            "metric": "svd_k_90",
            "value": svd_spec["k_90"],
            "description": "Latent dimension where TruncatedSVD reaches >=90% explained variance",
            "interpretation": "Suggested lower bound for AE latent dimension (captures most variance).",
        },
        {
            "metric": "svd_k_95",
            "value": svd_spec["k_95"],
            "description": "Latent dimension where TruncatedSVD reaches >=95% explained variance",
            "interpretation": "Suggested upper bound for AE latent dimension (more conservative, more detailed).",
        },
        {
            "metric": "svd_elbow_k",
            "value": svd_spec["elbow_k"],
            "description": "Elbow point in singular value decay (largest drop between consecutive singular values)",
            "interpretation": "Smallest meaningful AE latent dimension; components beyond this may be mostly noise.",
        },
    ])


    # --------------------------------------------------------
    # 5) Save all stats to CSV
    # --------------------------------------------------------
    stats_csv_path = EMB_DIR / "output" / "patient_embedding_stats.csv"
    write_stats_csv(stats_csv_path, stats_rows)

    print("\nAll statistics saved.\nDone.")


if __name__ == "__main__":
    main()
