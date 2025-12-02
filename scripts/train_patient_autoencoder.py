"""
Train a simple autoencoder on the binary patient x HPO matrix
and add the resulting patient embeddings into the existing
patient_embeddings_all_methods.npz file.

Requirements:
  - The compute_patient_embeddings_all_methods.py script has already
    been run, producing:
      output/patient_embeddings_all_methods.npz

This script will:
  1) Load metadata + patient x HPO matrix (aligned to metadata['hpo_ids'])
  2) Train an autoencoder on this matrix
  3) Load patient_embeddings_all_methods.npz
  4) Add 'autoencoder' and 'autoencoder_latent_dim' to it
  5) Save back to the same NPZ file
"""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split


# ============================================================
# Config
# ============================================================

ROOT_DIR = Path(".")
DATA_DIR = ROOT_DIR / "data"
OUT_DIR = ROOT_DIR / "output"

PATIENT_CSV_PATH = DATA_DIR / "binary_matrix.csv"
META_PATH = DATA_DIR / "embeddings_metadata.pkl"

ALL_EMB_PATH = OUT_DIR / "patient_embeddings_all_methods.npz"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LATENT_DIM = 64          # AE latent dimension (compare with SVD/NMF)
HIDDEN_DIM = 1024
BATCH_SIZE = 256
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.1
RANDOM_SEED = 42


# ============================================================
# Utils: loading and alignment
# ============================================================

def load_metadata(meta_path: Path):
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)
    return metadata


def load_and_align_patient_matrix(csv_path: Path, metadata) -> tuple[np.ndarray, list[str] | None]:
    """
    Load patient x phenotype matrix from CSV and align columns to the
    HPO order used in embeddings (metadata["hpo_ids"]).

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
        # First column is a patient identifier
        patient_ids = df.iloc[:, 0].astype(str).tolist()
        df = df.iloc[:, 1:]  # drop ID column

    # Remaining columns are assumed to be HPO IDs
    csv_hpo_ids = list(df.columns)

    # HPO IDs used in embeddings
    emb_hpo_ids = metadata["hpo_ids"]  # list of length n_hpo_total
    emb_index = {hpo_id: j for j, hpo_id in enumerate(emb_hpo_ids)}

    # Intersection between CSV HPOs and embedding HPOs
    matched_cols = [h for h in csv_hpo_ids if h in emb_index]
    unmatched_cols = [h for h in csv_hpo_ids if h not in emb_index]

    print(f"\nCSV HPO columns: {len(csv_hpo_ids)}")
    print(f"Matched HPO columns with embeddings: {len(matched_cols)}")
    print(f"Unmatched HPO columns (ignored): {len(unmatched_cols)}")
    if unmatched_cols:
        print("  Example unmatched HPO IDs:", unmatched_cols[:10])

    # Extract only matched columns
    df_matched = df[matched_cols]
    patient_matrix_small = df_matched.to_numpy(dtype=np.float32)   # (n_patients, n_matched)
    n_patients = patient_matrix_small.shape[0]
    n_hpo_total = len(emb_hpo_ids)

    # Build full matrix aligned with embedding order
    M = np.zeros((n_patients, n_hpo_total), dtype=np.float32)
    for k, hpo_id in enumerate(matched_cols):
        j = emb_index[hpo_id]  # column index in embedding space
        M[:, j] = patient_matrix_small[:, k]

    return M, patient_ids


# ============================================================
# Dataset
# ============================================================

class PatientPhenotypeDataset(Dataset):
    def __init__(self, matrix: np.ndarray):
        """
        matrix: (n_patients, n_features) float32
        """
        self.X = torch.from_numpy(matrix)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        return x, x   # input = target (autoencoder)


# ============================================================
# Autoencoder model
# ============================================================

class PatientAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),   # inputs are 0/1 (or in [0,1])
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


# ============================================================
# Training loop
# ============================================================

def train_autoencoder(model, train_loader, val_loader, device, num_epochs, lr):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()  # suitable for binary-like matrices

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            recon, _ = model(batch_x)
            loss = criterion(recon, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                recon, _ = model(batch_x)
                loss = criterion(recon, batch_y)
                val_loss += loss.item() * batch_x.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch:3d}/{num_epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        # Track best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"Best validation loss: {best_val_loss:.4f}")
    return model


# ============================================================
# Main
# ============================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load metadata & patient matrix (aligned)
    print(f"Loading metadata from: {META_PATH}")
    metadata = load_metadata(META_PATH)

    print(f"Loading patient matrix from: {PATIENT_CSV_PATH}")
    patient_matrix, patient_ids = load_and_align_patient_matrix(PATIENT_CSV_PATH, metadata)
    n_patients, n_features = patient_matrix.shape
    print(f"Aligned patient matrix shape: {patient_matrix.shape}")
    print(f"Number of patients: {n_patients}, number of phenotypes: {n_features}")

    # 2. Build Dataset & split into train/val
    dataset = PatientPhenotypeDataset(patient_matrix)

    n_val = int(VAL_SPLIT * len(dataset))
    n_train = len(dataset) - n_val

    torch.manual_seed(RANDOM_SEED)
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 3. Define and train autoencoder
    print("\nBuilding autoencoder...")
    model = PatientAutoencoder(
        input_dim=n_features,
        latent_dim=LATENT_DIM,
        hidden_dim=HIDDEN_DIM,
    )

    print(f"Training autoencoder on {DEVICE}...")
    model = train_autoencoder(
        model,
        train_loader,
        val_loader,
        device=DEVICE,
        num_epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
    )

    # 4. Extract latent embeddings for all patients
    print("\nComputing latent embeddings for all patients...")
    model.eval()
    all_latent = []
    full_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    with torch.no_grad():
        for batch_x, _ in full_loader:
            batch_x = batch_x.to(DEVICE)
            _, z = model(batch_x)
            all_latent.append(z.cpu().numpy())

    Z_ae = np.vstack(all_latent)  # (n_patients, LATENT_DIM)
    print("Autoencoder embedding shape:", Z_ae.shape)

    # 5. Load existing embeddings file and merge
    print(f"\nLoading existing embeddings from: {ALL_EMB_PATH}")
    data = np.load(ALL_EMB_PATH, allow_pickle=True)
    existing = {k: data[k] for k in data.files}

    # 5a. Check patient_ids consistency if present
    existing_patient_ids = existing.get("patient_ids", None)
    if existing_patient_ids is not None and existing_patient_ids.size > 0:
        existing_patient_ids = existing_patient_ids.astype(str).tolist()
        if patient_ids is not None:
            if len(existing_patient_ids) != len(patient_ids):
                raise ValueError(
                    f"Mismatch in number of patients between NPZ ({len(existing_patient_ids)}) "
                    f"and AE input ({len(patient_ids)})."
                )
            if existing_patient_ids != [str(p) for p in patient_ids]:
                print("WARNING: patient_ids differ between NPZ and AE input.")
                print("         Assuming same order, but IDs are not identical.")
        else:
            print("NOTE: Existing NPZ has patient_ids, but AE script CSV has no explicit IDs.")
    else:
        # NPZ has no patient_ids stored
        if patient_ids is not None:
            print("NOTE: Existing NPZ has no patient_ids, but AE script CSV does. "
                  "They will NOT be stored/updated here.")
        else:
            print("NOTE: No patient_ids present in either source or target.")

    # 5b. Attach AE embeddings into the same file
    existing["autoencoder"] = Z_ae
    existing["autoencoder_latent_dim"] = np.array([LATENT_DIM], dtype=np.int32)

    # 6. Save back to the same NPZ
    print(f"\nSaving updated embeddings (with autoencoder) to: {ALL_EMB_PATH}")
    np.savez_compressed(ALL_EMB_PATH, **existing)

    print("Done.")


if __name__ == "__main__":
    main()
