"""
Generate a synthetic diagnosis file aligned with patient_embeddings_all_methods.npz.

- patient_id format: "ann_clean_reportX"
- X ranges between 2 and 20491
- But only some subset appears in actual patient_ids (from NPZ)
- Generate random diagnoses (categorical)
- Include missing diagnoses for some patients (simulate real-world gaps)

Output:
    data/diagnosis_synthetic.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
import random

# Paths
ROOT = Path(".")
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "output"

NPZ_PATH = OUT_DIR / "patient_embeddings_all_methods.npz"
OUT_CSV_PATH = DATA_DIR / "diagnosis_synthetic.csv"

# Config
NUM_DIAGNOSES = 15                # total synthetic diagnostic categories
MISSING_RATE = 0.95               # 95% of patients will be missing diagnosis
RANDOM_SEED = 42

def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Load patient IDs from NPZ
    print("Loading patient embeddings from:", NPZ_PATH)
    data = np.load(NPZ_PATH, allow_pickle=True)

    if "patient_ids" not in data:
        raise ValueError("NPZ file must contain a `patient_ids` field.")

    patient_ids = data["patient_ids"].astype(str).tolist()
    print(f"Found {len(patient_ids)} patient IDs.")

    # Create mapping: patient_id → synthetic diagnosis
    diagnoses = [f"Diagnosis_{i}" for i in range(1, NUM_DIAGNOSES + 1)]

    rows = []
    for pid in patient_ids:

        # Convert patient ID into the requested format:
        # Use X extracted from ann_clean_reportX
        # If the ID doesn't match that format, generate one
        if pid.startswith("ann_clean_report"):
            patient_id_formatted = pid
        else:
            # if numeric ID, convert to ann_clean_reportX format
            try:
                x = int(pid)
                patient_id_formatted = f"ann_clean_report{x}"
            except:
                # fallback
                patient_id_formatted = pid

        # Randomly skip some diagnoses to create missing values
        if random.random() < MISSING_RATE:
            diag = ""   # missing
        else:
            diag = random.choice(diagnoses)

        rows.append((patient_id_formatted, diag))

    # Save DataFrame
    df = pd.DataFrame(rows, columns=["patient_id", "diagnosis"])
    OUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV_PATH, index=False)

    print(f"Synthetic diagnosis file written to: {OUT_CSV_PATH}")
    print(df.head())


if __name__ == "__main__":
    main()
