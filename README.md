# Pheno Profile

This project provides a complete pipeline for learning dense patient embeddings from a binary patient × HPO phenotype matrix, using multiple embedding methods:

* Hyperbolic embeddings (Fréchet mean, Einstein midpoint)

* Linear factorization (Truncated SVD, NMF)

* Non-linear autoencoder

* Optional: hybrid methods (planned)

The goal is to compare how different mathematical representations capture phenotypic similarity across patients, and how well these representations align with ground-truth diagnoses.

## 🌳 Project structure

```
project/
│
├── data/
│   ├── binary_matrix.csv              # Patient × phenotype binary matrix
│   ├── diagnosis_synthetic.csv        # Synthetic patient diagnoses (generated)
│   ├── hpo_embeddings.npy             # Precomputed HPO hyperbolic embeddings
│   ├── embeddings_metadata.pkl        # Contains metadata, including ordered HPO IDs
│   └── [...]                          # Other data files
│
├── output/
│   ├── patient_embeddings_all_methods.npz   # Final consolidated embedding file
│   ├── patient_embedding_stats.csv          # Stats describing embeddings
│   ├── evaluation_results.csv               # Cluster evaluation metrics
│   ├── plots/
│   │   ├── tsne_frechet_unw.png
│   │   ├── tsne_einstein_unw.png
│   │   └── ... (t-SNE visualizations)
│   └── [...]
│
├── scripts/
│   ├── compute_patient_embeddings_all_methods.py   # Generates all embeddings
│   ├── evaluate_patient_embeddings.py               # Evaluates clustering vs diagnosis
│   ├── train_patient_autoencoder.py                 # Trains AE + stores embeddings
│   ├── generate_synthetic_diagnosis_file.py         # Creates fake diagnosis file
│   └── [...]
│
└── README.md
```

## 🧩 Core Concepts

### 1. Patient × Phenotype Matrix

The project starts from a CSV where:

* Rows = patients

* Columns = HPO codes

* Values = 1/0 (phenotype present/absent)

The matrix is automatically aligned with the HPO embeddings using metadata["hpo_ids"] (the first column of the CSV should be IDs)


### 2. HPO Hyperbolic Embeddings

Using a pretrained HierarchyTransformers hyperbolic model, each phenotype has a dense embedding in the Poincaré ball. These embeddings encode:

* hierarchical depth

* semantic similarity

* taxonomic relationships

### 3. Patient Embeddings (6 methods)

For each patient, we derive a dense embedding using:

#### Data-driven approaches

Linear and non-linear methods using only the binary Patient × Phenotype matrix.

* Truncated SVD
* NMF (Non-negative Matrix Factorization)
* Autoencoder

#### Knowledge-based approaches

Non-linear based on HPO hyperbolic embeddings applied to the binary Patient × Phenotype matrix. 

Unweighted methods treat each phenotype equally, while IC (Information-Content) methods weight each phenotype based on its relative discriminative importance within all the patients.

* Fréchet mean (unweighted)
* Einstein midpoint (unweighted)
* Fréchet mean (IC-weighted)
* Einstein midpoint (IC-weighted)

All resulting embeddings are stored in a single file:
```text
output/patient_embeddings_all_methods.npz
```

## 🧪 Evaluation Workflow

The evaluation script compares each embedding method using:

### 1. Clustering performance

Using KMeans (k = number of diagnosis classes):

* Adjusted Rand Index (ARI)
* Normalized Mutual Information (NMI)

These compare unsupervised clusters to the (synthetic or real) diagnosis labels.

### 2. Intrinsic cluster quality

Independent of diagnoses:

* Silhouette score
* Davies–Bouldin index

These assess how “clusterable” the embedding space is.

### 3. Visual inspection

Optional t-SNE projections for each embedding:

```
output/plots/tsne_<method>.png
```

## 🛠 How to Run the Pipeline