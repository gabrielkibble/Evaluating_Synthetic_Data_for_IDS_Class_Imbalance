# Synthetic Traffic Augmentation for Minority Class Detection in Network IDS

**Repository:** [https://github.com/gabrielkibble/Dissertation](https://github.com/gabrielkibble/Dissertation)

This repository contains the implementation and evaluation code for a supervised Markov chain pipeline that generates synthetic network traffic to improve minority class detection in intrusion detection systems.

## Repository Structure

```
DBN/dbn-based-nids/
├── models/                          # DBN, MLP, RBM model implementations
├── preprocessing/
│   ├── cicids2017.py                # CICIDS2017 preprocessing + synthetic injection
│   └── unsw_nb15.py                 # UNSW-NB15 preprocessing + synthetic injection
├── utils/                           # Training, testing, visualisation utilities
├── configs/                         # Model configuration files (JSON)
├── notebooks/                       # Exploratory analysis notebooks
├── logs/                            # Training logs
│
├── main.py                          # Main training/evaluation entry point
├── working_no_prints.py             # Synthetic generation pipeline (both datasets)
├── synthetic_cicids_standard.csv    # Generated: Markov T=1.0 (CICIDS2017)
├── synthetic_cicids_zeroday.csv     # Generated: Markov T=1.3 (CICIDS2017)
├── synthetic_cicids_T{1.4-3.0}.csv  # Generated: temperature sweep variants
├── synthetic_unsw_standard.csv      # Generated: Markov T=1.0 (UNSW-NB15)
│
├── smote_comparison.py              # SMOTE augmentation (CICIDS2017)
├── smote_comparison_unsw.py         # SMOTE augmentation (UNSW-NB15)
├── visualise_synthetic.py           # t-SNE/PCA real vs synthetic plots
├── visualise_smote_comparison.py    # Three-way comparison visualisations
├── run_significance_test.py         # Paired t-test (MLP, 5 seeds)
├── run_significance_test_dbn.py     # Paired t-test (DBN, 5 seeds)
├── sha256_verification.py           # Privacy verification (both datasets)
│
├── balanced_cicids2017.csv          # Balanced subset for DBSCAN clustering
├── balanced_NB15_iot.csv            # Balanced subset for DBSCAN clustering
├── requirements.txt                 # Python dependencies
└── Makefile                         # Build shortcuts
```

```
IDS2 Bret/IntrusionDetectionSystem/Python/
├── Preprocess.py                    # Autoencoder data preprocessing
├── Main.py                          # Autoencoder training and evaluation
├── run_synthetic_experiment.py      # Synthetic threshold calibration experiment
├── IDS.py                           # Autoencoder model definition
├── DataEncoding.py                  # Feature encoding utilities
├── Hyper.py                         # Hyperparameter configuration
├── Tuning.py                        # Hyperparameter tuning
└── trained.h5                       # Trained autoencoder weights

Kitsune-py/                          # Kitsune (excluded - see dissertation)
```

## Datasets

This project uses two publicly available benchmark datasets. Download and place them as follows:

### CICIDS2017
- Download from: https://www.unb.ca/cic/datasets/ids-2017.html
- Place all 8 CSV files in `DBN/dbn-based-nids/data/raw/`

### UNSW-NB15
- Download from: https://research.unsw.edu.au/projects/unsw-nb15-dataset
- Place the 4 CSV files in `DBN/dbn-based-nids/data/raw_unsw/`

## Setup

```bash
cd DBN/dbn-based-nids
python3 -m venv dbn_ids
source dbn_ids/bin/activate
pip install -r requirements.txt
```

## Reproduction

All commands assume you are in `DBN/dbn-based-nids/` with the virtual environment activated.

### 1. CICIDS2017 Baseline

```bash
# Create symlink
ln -s processed_cicids data/processed

# Preprocess (no synthetic data)
rm -f data/synthetic_cicids_standard.csv
python preprocessing/cicids2017.py

# Train baseline models
python main.py --config ./configs/multilayerPerceptron.json
python main.py --config ./configs/deepBeliefNetwork.json

# Save baseline for later comparison
cp data/processed_cicids/train/train_features.pkl data/processed_cicids/train/train_features_baseline.pkl
cp data/processed_cicids/train/train_labels.pkl data/processed_cicids/train/train_labels_baseline.pkl
```

### 2. Generate Synthetic Data (CICIDS2017)

```bash
# Generates synthetic_cicids_standard.csv (T=1.0), synthetic_cicids_zeroday.csv (T=1.3),
# and temperature sweep variants (T=1.4 to T=3.0)
python working_no_prints.py
```

### 3. Train with Markov T=1.0

```bash
cp synthetic_cicids_standard.csv data/synthetic_cicids_standard.csv
python preprocessing/cicids2017.py
python main.py --config ./configs/multilayerPerceptron.json
python main.py --config ./configs/deepBeliefNetwork.json
```

### 4. Train with Markov T=1.3

```bash
cp synthetic_cicids_zeroday.csv data/synthetic_cicids_standard.csv
python preprocessing/cicids2017.py
python main.py --config ./configs/multilayerPerceptron.json
python main.py --config ./configs/deepBeliefNetwork.json
```

### 5. Train with SMOTE

```bash
python smote_comparison.py
python main.py --config ./configs/multilayerPerceptron.json
python main.py --config ./configs/deepBeliefNetwork.json
```

### 6. Temperature Sweep (T=1.4 to T=3.0)

```bash
for T in 1.4 1.6 1.8 2.0 2.5 3.0; do
    cp data/processed_cicids/train/train_features_baseline.pkl data/processed_cicids/train/train_features.pkl
    cp data/processed_cicids/train/train_labels_baseline.pkl data/processed_cicids/train/train_labels.pkl
    cp synthetic_cicids_T${T}.csv data/synthetic_cicids_standard.csv
    python preprocessing/cicids2017.py
    python main.py --config ./configs/multilayerPerceptron.json
done
```

### 7. Statistical Significance Testing

```bash
# MLP: 5 seeds, baseline vs Markov T=1.0
python run_significance_test.py

# DBN: 5 seeds, baseline vs Markov T=1.0
python run_significance_test_dbn.py
```

### 8. UNSW-NB15

```bash
# Switch symlink
rm data/processed
ln -s processed_unsw data/processed

# Generate synthetic UNSW data
python working_no_prints.py

# Baseline (no synthetic data)
rm -f data/synthetic_unsw_standard.csv
python preprocessing/unsw_nb15.py
python main.py --config ./configs/deepBeliefNetwork.json
python main.py --config ./configs/multilayerPerceptron.json

# With synthetic augmentation
cp synthetic_unsw_standard.csv data/synthetic_unsw_standard.csv
python preprocessing/unsw_nb15.py
python main.py --config ./configs/deepBeliefNetwork.json
python main.py --config ./configs/multilayerPerceptron.json

# SMOTE comparison
python smote_comparison_unsw.py
```

### 9. Autoencoder

```bash
cd "IDS2 Bret/IntrusionDetectionSystem/Python"
source ../../ids2_env/bin/activate

python Preprocess.py
python Main.py
python run_synthetic_experiment.py
```

### 10. Visualisations

```bash
cd DBN/dbn-based-nids/
python visualise_synthetic.py
python visualise_smote_comparison.py
```

### 11. SHA-256 Privacy Verification

```bash
python sha256_verification.py
# Output saved to sha256_verification.txt
```

## GPU Usage

If the default GPU is occupied, specify an alternative:

```bash
CUDA_VISIBLE_DEVICES=1 python main.py --config ./configs/deepBeliefNetwork.json
```

## Key Results

| Dataset | Model | Baseline | Markov T=1.3 | SMOTE |
|---------|-------|----------|--------------|-------|
| CICIDS2017 | MLP | 0.66 | **0.89** | 0.87 |
| CICIDS2017 | DBN | 0.50 | 0.76 | **0.78** |
| UNSW-NB15 | MLP | 0.27 | 0.27 | 0.30 |
| UNSW-NB15 | DBN | 0.25 | 0.11 | **0.32** |

Statistical significance confirmed via paired t-tests (MLP: p=0.0016, DBN: p=0.021).