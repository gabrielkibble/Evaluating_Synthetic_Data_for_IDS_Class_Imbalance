"""
Visualise REAL vs MARKOV CHAIN vs SMOTE synthetic data for CICIDS2017.
Three-way comparison showing how each augmentation method relates to real data.

Run from: ~/Dissertation/DBN/dbn-based-nids/
Usage: dbn_ids/bin/python visualise_smote_comparison.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from imblearn.over_sampling import SMOTE
import glob
import os
import time

# ============================================================
# CONFIGURATION
# ============================================================

MARKOV_SYNTHETIC_CSV = "synthetic_cicids_standard.csv"
RAW_DATA_DIR = "./data/raw/"
OUTPUT_DIR = "./images/"

FEATURES = [
    "flow_duration",
    "total_fwd_packets", "total_backward_packets",
    "total_length_of_fwd_packets", "total_length_of_bwd_packets",
    "fwd_packets_s", "bwd_packets_s",
    "fwd_packet_length_mean", "bwd_packet_length_mean",
    "fwd_iat_std", "bwd_iat_std",
    "fwd_iat_mean", "bwd_iat_mean"
]

ATTACK_GROUP = {
    'BENIGN': 'Benign', 'PortScan': 'PortScan', 'DDoS': 'DoS/DDoS',
    'DoS Hulk': 'DoS/DDoS', 'DoS GoldenEye': 'DoS/DDoS',
    'DoS slowloris': 'DoS/DDoS', 'DoS Slowhttptest': 'DoS/DDoS',
    'Heartbleed': 'DoS/DDoS', 'FTP-Patator': 'Brute Force',
    'SSH-Patator': 'Brute Force', 'Bot': 'Botnet ARES',
    'Web Attack – Brute Force': 'Web Attack',
    'Web Attack – Sql Injection': 'Web Attack',
    'Web Attack – XSS': 'Web Attack',
    'Web Attack  Brute Force': 'Web Attack',
    'Web Attack  Sql Injection': 'Web Attack',
    'Web Attack  XSS': 'Web Attack',
}

COLOURS = {
    'Benign': '#2196F3',
    'Botnet ARES': '#F44336',
    'Brute Force': '#FF9800',
    'DoS/DDoS': '#9C27B0',
    'PortScan': '#4CAF50',
    'Web Attack': '#E91E63',
}

# Sample sizes — smaller for three-way comparison to keep plots readable
SAMPLE_PER_CATEGORY = 500

os.makedirs(OUTPUT_DIR, exist_ok=True)


def clean_column_name(col):
    return col.strip().replace('/', '_').replace(' ', '_').lower()


def load_real_data():
    """Load and process real CICIDS2017 data."""
    print("Loading real CICIDS2017 data...")
    files = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "*.csv")))
    files = [f for f in files if 'synthetic' not in os.path.basename(f).lower()]

    dfs = []
    for f in files:
        print(f"  Reading {os.path.basename(f)}...")
        df = pd.read_csv(f, encoding='cp1252', low_memory=False)
        df.columns = [clean_column_name(c) for c in df.columns]
        dfs.append(df)

    df_real = pd.concat(dfs, ignore_index=True)
    df_real['category'] = df_real['label'].map(
        lambda x: ATTACK_GROUP.get(str(x).strip(), 'Other')
    )
    df_real = df_real[df_real['category'] != 'Other']

    for col in FEATURES:
        df_real[col] = pd.to_numeric(df_real[col], errors='coerce')
    df_real = df_real.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)

    return df_real


def load_markov_synthetic():
    """Load Markov chain synthetic data."""
    print(f"Loading Markov chain synthetic data...")
    df = pd.read_csv(MARKOV_SYNTHETIC_CSV, low_memory=False)
    df.columns = [clean_column_name(c) for c in df.columns]

    GROUPED_LABELS = set(COLOURS.keys())
    df['category'] = df['label'].map(
        lambda x: str(x).strip() if str(x).strip() in GROUPED_LABELS
        else ATTACK_GROUP.get(str(x).strip(), 'Other')
    )
    df = df[df['category'] != 'Other']

    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)

    return df


def generate_smote_synthetic(df_real):
    """Generate SMOTE synthetic data for minority classes only."""
    print("Generating SMOTE synthetic data...")

    # Use only the minority classes that SMOTE would typically target
    target_classes = ['Botnet ARES', 'Brute Force', 'Web Attack']
    benign_sample = df_real[df_real['category'] == 'Benign'].sample(
        n=min(50000, len(df_real[df_real['category'] == 'Benign'])),
        random_state=42
    )

    attack_subsets = [df_real[df_real['category'] == cat] for cat in target_classes]
    df_subset = pd.concat([benign_sample] + attack_subsets, ignore_index=True)

    X = df_subset[FEATURES].values.astype(np.float32)
    y = df_subset['category'].values

    # Target counts (matching the Markov chain volumes approximately)
    target_counts = {
        'Benign': len(benign_sample),  # Keep as is
        'Botnet ARES': 20000,
        'Brute Force': 20000,
        'Web Attack': 20000,
    }

    start = time.time()
    smote = SMOTE(sampling_strategy=target_counts, random_state=42, k_neighbors=3)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print(f"SMOTE took {time.time() - start:.1f}s")

    # Extract only the synthesised samples (not the originals)
    # We identify originals by comparing to the input
    synthetic_mask = np.ones(len(X_resampled), dtype=bool)
    synthetic_mask[:len(X)] = False  # First len(X) are the original real samples

    df_smote = pd.DataFrame(X_resampled[synthetic_mask], columns=FEATURES)
    df_smote['category'] = y_resampled[synthetic_mask]

    return df_smote


def sample_by_category(df, n_per_category):
    """Sample up to n rows per category."""
    sampled = []
    for cat in df['category'].unique():
        subset = df[df['category'] == cat]
        n = min(len(subset), n_per_category)
        sampled.append(subset.sample(n=n, random_state=42))
    return pd.concat(sampled, ignore_index=True)


def plot_three_way_tsne(X_real, X_markov, X_smote, labels_real, labels_markov, labels_smote,
                        filename="tsne_three_way_comparison.pdf"):
    """Three-panel t-SNE: real, Markov, SMOTE."""
    print("Computing three-way t-SNE (may take a few minutes)...")

    scaler = StandardScaler()
    X_combined = np.vstack([X_real, X_markov, X_smote])
    X_scaled = scaler.fit_transform(X_combined)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000, verbose=1)
    X_tsne = tsne.fit_transform(X_scaled)

    n_real = len(X_real)
    n_markov = len(X_markov)
    X_tsne_real = X_tsne[:n_real]
    X_tsne_markov = X_tsne[n_real:n_real + n_markov]
    X_tsne_smote = X_tsne[n_real + n_markov:]

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    # Plot 1: Real
    ax = axes[0]
    for cat in sorted(COLOURS.keys()):
        mask = labels_real == cat
        if mask.sum() > 0:
            ax.scatter(X_tsne_real[mask, 0], X_tsne_real[mask, 1],
                      c=COLOURS[cat], label=cat, alpha=0.5, s=10, edgecolors='none')
    ax.set_title('Real Traffic', fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(fontsize=8, markerscale=2, loc='upper right')

    # Plot 2: Markov chain synthetic
    ax = axes[1]
    for cat in sorted(COLOURS.keys()):
        mask = labels_markov == cat
        if mask.sum() > 0:
            ax.scatter(X_tsne_markov[mask, 0], X_tsne_markov[mask, 1],
                      c=COLOURS[cat], label=cat, alpha=0.5, s=10, marker='x')
    ax.set_title('Markov Chain Synthetic', fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(fontsize=8, markerscale=2, loc='upper right')

    # Plot 3: SMOTE synthetic
    ax = axes[2]
    for cat in sorted(COLOURS.keys()):
        mask = labels_smote == cat
        if mask.sum() > 0:
            ax.scatter(X_tsne_smote[mask, 0], X_tsne_smote[mask, 1],
                      c=COLOURS[cat], label=cat, alpha=0.5, s=10, marker='+')
    ax.set_title('SMOTE Synthetic', fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(fontsize=8, markerscale=2, loc='upper right')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved {path}")
    plt.close()


def plot_per_class_three_way(X_real, X_markov, X_smote, labels_real, labels_markov, labels_smote,
                             filename="per_class_three_way_pca.pdf"):
    """Per-class PCA with real (circles), Markov (×), SMOTE (+)."""
    print("Computing per-class three-way PCA comparison...")

    scaler = StandardScaler()
    X_combined = np.vstack([X_real, X_markov, X_smote])
    X_scaled = scaler.fit_transform(X_combined)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    n_real = len(X_real)
    n_markov = len(X_markov)
    X_pca_real = X_pca[:n_real]
    X_pca_markov = X_pca[n_real:n_real + n_markov]
    X_pca_smote = X_pca[n_real + n_markov:]

    # Only plot classes that were augmented by both methods
    augmented_classes = ['Botnet ARES', 'Brute Force', 'Web Attack']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, cat in enumerate(augmented_classes):
        ax = axes[i]

        # Plot benign background
        mask_benign = labels_real == 'Benign'
        if mask_benign.sum() > 0:
            ax.scatter(X_pca_real[mask_benign, 0], X_pca_real[mask_benign, 1],
                      c='#DDDDDD', alpha=0.2, s=5, edgecolors='none', label='Benign (real)')

        # Plot real attack
        mask_r = labels_real == cat
        if mask_r.sum() > 0:
            ax.scatter(X_pca_real[mask_r, 0], X_pca_real[mask_r, 1],
                      c=COLOURS[cat], alpha=0.6, s=25, edgecolors='none',
                      label=f'{cat} (real)')

        # Plot Markov synthetic
        mask_m = labels_markov == cat
        if mask_m.sum() > 0:
            ax.scatter(X_pca_markov[mask_m, 0], X_pca_markov[mask_m, 1],
                      c=COLOURS[cat], alpha=0.5, s=30, marker='x',
                      label=f'{cat} (Markov)')

        # Plot SMOTE synthetic — black edge to distinguish
        mask_s = labels_smote == cat
        if mask_s.sum() > 0:
            ax.scatter(X_pca_smote[mask_s, 0], X_pca_smote[mask_s, 1],
                      facecolors='none', edgecolors=COLOURS[cat], alpha=0.7, s=35,
                      marker='o', linewidth=1.2,
                      label=f'{cat} (SMOTE)')

        ax.set_title(cat, fontsize=12, fontweight='bold')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.legend(fontsize=8, markerscale=1.2, loc='upper right')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved {path}")
    plt.close()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SMOTE vs MARKOV CHAIN SYNTHETIC DATA COMPARISON")
    print("=" * 60)

    df_real = load_real_data()
    df_markov = load_markov_synthetic()
    df_smote = generate_smote_synthetic(df_real)

    print(f"\nReal: {len(df_real):,} rows")
    print(f"Markov: {len(df_markov):,} rows")
    print(f"SMOTE: {len(df_smote):,} rows")

    # Sample for visualisation
    df_real_s = sample_by_category(df_real, SAMPLE_PER_CATEGORY)
    df_markov_s = sample_by_category(df_markov, SAMPLE_PER_CATEGORY)
    df_smote_s = sample_by_category(df_smote, SAMPLE_PER_CATEGORY)

    X_real = df_real_s[FEATURES].values
    X_markov = df_markov_s[FEATURES].values
    X_smote = df_smote_s[FEATURES].values

    labels_real = df_real_s['category'].values
    labels_markov = df_markov_s['category'].values
    labels_smote = df_smote_s['category'].values

    print(f"\nSampled: real={len(X_real)}, markov={len(X_markov)}, smote={len(X_smote)}")

    plot_three_way_tsne(X_real, X_markov, X_smote, labels_real, labels_markov, labels_smote)
    plot_per_class_three_way(X_real, X_markov, X_smote, labels_real, labels_markov, labels_smote)

    print("\n" + "=" * 60)
    print("DONE")
    print(f"Saved to {OUTPUT_DIR}")
    print("=" * 60)