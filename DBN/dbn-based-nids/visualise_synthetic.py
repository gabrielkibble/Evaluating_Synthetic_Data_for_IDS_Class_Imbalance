"""
Visualise real vs synthetic CICIDS2017 traffic using PCA and t-SNE.
Shows that synthetic data overlaps with but extends the real distribution.

Run from: ~/Dissertation/DBN/dbn-based-nids/
Usage: dbn_ids/bin/python visualise_synthetic.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import glob
import os

# ============================================================
# CONFIGURATION
# ============================================================

SYNTHETIC_CSV = "synthetic_cicids_standard.csv"
RAW_DATA_DIR = "./data/raw/"
OUTPUT_DIR = "./images/"

# Same features used in generation pipeline
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

# Colours for each category
COLOURS = {
    'Benign': '#2196F3',
    'Botnet ARES': '#F44336',
    'Brute Force': '#FF9800',
    'DoS/DDoS': '#9C27B0',
    'PortScan': '#4CAF50',
    'Web Attack': '#E91E63',
}

# Sample sizes for visualisation (too many points = unreadable plot)
SAMPLE_REAL = 5000
SAMPLE_SYNTH = 5000

os.makedirs(OUTPUT_DIR, exist_ok=True)


def clean_column_name(col):
    return col.strip().replace('/', '_').replace(' ', '_').lower()


def load_real_data():
    """Load and sample real CICIDS2017 data."""
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

    # Group labels
    df_real['category'] = df_real['label'].map(
        lambda x: ATTACK_GROUP.get(str(x).strip(), 'Other')
    )
    df_real = df_real[df_real['category'] != 'Other']
    df_real = df_real[df_real['category'] != 'Infiltration']

    # Force numeric
    for col in FEATURES:
        df_real[col] = pd.to_numeric(df_real[col], errors='coerce')
    df_real = df_real.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)

    return df_real


def load_synthetic_data():
    """Load synthetic CICIDS2017 data."""
    print(f"Loading synthetic data from {SYNTHETIC_CSV}...")
    df_synth = pd.read_csv(SYNTHETIC_CSV, low_memory=False)
    df_synth.columns = [clean_column_name(c) for c in df_synth.columns]

    # Map labels to grouped categories
    GROUPED_LABELS = set(COLOURS.keys())
    df_synth['category'] = df_synth['label'].map(
        lambda x: str(x).strip() if str(x).strip() in GROUPED_LABELS
        else ATTACK_GROUP.get(str(x).strip(), 'Other')
    )
    df_synth = df_synth[df_synth['category'] != 'Other']

    for col in FEATURES:
        df_synth[col] = pd.to_numeric(df_synth[col], errors='coerce')
    df_synth = df_synth.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)

    return df_synth


def sample_by_category(df, n_per_category):
    """Sample up to n rows per category for balanced visualisation."""
    sampled = []
    for cat in df['category'].unique():
        subset = df[df['category'] == cat]
        n = min(len(subset), n_per_category)
        sampled.append(subset.sample(n=n, random_state=42))
    return pd.concat(sampled, ignore_index=True)


def plot_pca(X_real, X_synth, labels_real, labels_synth, filename="pca_real_vs_synthetic.pdf"):
    """PCA visualisation: real vs synthetic, side by side."""
    print("Computing PCA...")

    scaler = StandardScaler()
    X_combined = np.vstack([X_real, X_synth])
    X_scaled = scaler.fit_transform(X_combined)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    X_pca_real = X_pca[:len(X_real)]
    X_pca_synth = X_pca[len(X_real):]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Plot 1: Real data only
    ax = axes[0]
    for cat in sorted(COLOURS.keys()):
        mask = labels_real == cat
        if mask.sum() > 0:
            ax.scatter(X_pca_real[mask, 0], X_pca_real[mask, 1],
                      c=COLOURS[cat], label=cat, alpha=0.4, s=8, edgecolors='none')
    ax.set_title('Real Traffic', fontsize=14, fontweight='bold')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax.legend(fontsize=7, markerscale=2, loc='upper right')

    # Plot 2: Synthetic data only
    ax = axes[1]
    for cat in sorted(COLOURS.keys()):
        mask = labels_synth == cat
        if mask.sum() > 0:
            ax.scatter(X_pca_synth[mask, 0], X_pca_synth[mask, 1],
                      c=COLOURS[cat], label=cat, alpha=0.4, s=8, marker='x')
    ax.set_title('Synthetic Traffic', fontsize=14, fontweight='bold')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax.legend(fontsize=7, markerscale=2, loc='upper right')

    # Plot 3: Overlay
    ax = axes[2]
    for cat in sorted(COLOURS.keys()):
        mask_r = labels_real == cat
        mask_s = labels_synth == cat
        if mask_r.sum() > 0:
            ax.scatter(X_pca_real[mask_r, 0], X_pca_real[mask_r, 1],
                      c=COLOURS[cat], alpha=0.3, s=8, edgecolors='none', label=f'{cat} (real)')
        if mask_s.sum() > 0:
            ax.scatter(X_pca_synth[mask_s, 0], X_pca_synth[mask_s, 1],
                      c=COLOURS[cat], alpha=0.5, s=12, marker='x', label=f'{cat} (synth)')
    ax.set_title('Overlay', fontsize=14, fontweight='bold')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax.legend(fontsize=6, markerscale=2, loc='upper right', ncol=2)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved PCA plot to {path}")
    plt.close()


def plot_tsne(X_real, X_synth, labels_real, labels_synth, filename="tsne_real_vs_synthetic.pdf"):
    """t-SNE visualisation: real vs synthetic, side by side."""
    print("Computing t-SNE (this may take a few minutes)...")

    scaler = StandardScaler()
    X_combined = np.vstack([X_real, X_synth])
    X_scaled = scaler.fit_transform(X_combined)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000, verbose=1)
    X_tsne = tsne.fit_transform(X_scaled)

    X_tsne_real = X_tsne[:len(X_real)]
    X_tsne_synth = X_tsne[len(X_real):]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Plot 1: Real data only
    ax = axes[0]
    for cat in sorted(COLOURS.keys()):
        mask = labels_real == cat
        if mask.sum() > 0:
            ax.scatter(X_tsne_real[mask, 0], X_tsne_real[mask, 1],
                      c=COLOURS[cat], label=cat, alpha=0.4, s=8, edgecolors='none')
    ax.set_title('Real Traffic', fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(fontsize=7, markerscale=2, loc='upper right')

    # Plot 2: Synthetic data only
    ax = axes[1]
    for cat in sorted(COLOURS.keys()):
        mask = labels_synth == cat
        if mask.sum() > 0:
            ax.scatter(X_tsne_synth[mask, 0], X_tsne_synth[mask, 1],
                      c=COLOURS[cat], label=cat, alpha=0.4, s=8, marker='x')
    ax.set_title('Synthetic Traffic', fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(fontsize=7, markerscale=2, loc='upper right')

    # Plot 3: Overlay
    ax = axes[2]
    for cat in sorted(COLOURS.keys()):
        mask_r = labels_real == cat
        mask_s = labels_synth == cat
        if mask_r.sum() > 0:
            ax.scatter(X_tsne_real[mask_r, 0], X_tsne_real[mask_r, 1],
                      c=COLOURS[cat], alpha=0.3, s=8, edgecolors='none', label=f'{cat} (real)')
        if mask_s.sum() > 0:
            ax.scatter(X_tsne_synth[mask_s, 0], X_tsne_synth[mask_s, 1],
                      c=COLOURS[cat], alpha=0.5, s=12, marker='x', label=f'{cat} (synth)')
    ax.set_title('Overlay', fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(fontsize=6, markerscale=2, loc='upper right', ncol=2)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved t-SNE plot to {path}")
    plt.close()


def plot_per_class_comparison(X_real, X_synth, labels_real, labels_synth,
                               filename="per_class_pca.pdf"):
    """Per-class PCA overlay — one subplot per attack category."""
    print("Computing per-class PCA comparison...")

    scaler = StandardScaler()
    X_combined = np.vstack([X_real, X_synth])
    X_scaled = scaler.fit_transform(X_combined)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    X_pca_real = X_pca[:len(X_real)]
    X_pca_synth = X_pca[len(X_real):]

    categories = sorted([c for c in COLOURS.keys() if c != 'Benign'])
    n_cats = len(categories)
    cols = 3
    rows = (n_cats + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = axes.flatten()

    for i, cat in enumerate(categories):
        ax = axes[i]

        # Plot benign background in grey
        mask_benign_r = labels_real == 'Benign'
        if mask_benign_r.sum() > 0:
            ax.scatter(X_pca_real[mask_benign_r, 0], X_pca_real[mask_benign_r, 1],
                      c='#CCCCCC', alpha=0.15, s=4, edgecolors='none', label='Benign (real)')

        # Plot real attack
        mask_r = labels_real == cat
        if mask_r.sum() > 0:
            ax.scatter(X_pca_real[mask_r, 0], X_pca_real[mask_r, 1],
                      c=COLOURS[cat], alpha=0.5, s=15, edgecolors='none', label=f'{cat} (real)')

        # Plot synthetic attack
        mask_s = labels_synth == cat
        if mask_s.sum() > 0:
            ax.scatter(X_pca_synth[mask_s, 0], X_pca_synth[mask_s, 1],
                      c=COLOURS[cat], alpha=0.6, s=20, marker='x', label=f'{cat} (synth)')

        ax.set_title(cat, fontsize=12, fontweight='bold')
        ax.set_xlabel(f'PC1')
        ax.set_ylabel(f'PC2')
        ax.legend(fontsize=7, markerscale=1.5)

    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved per-class PCA plot to {path}")
    plt.close()


def plot_feature_distributions(df_real, df_synth, filename="feature_distributions.pdf"):
    """Compare feature distributions between real and synthetic data for target classes."""
    print("Plotting feature distributions...")

    target_classes = ['Brute Force', 'Web Attack', 'Botnet ARES']
    plot_features = ['flow_duration', 'total_fwd_packets', 'fwd_packet_length_mean',
                     'fwd_iat_mean', 'bwd_iat_std', 'fwd_packets_s']

    fig, axes = plt.subplots(len(target_classes), len(plot_features),
                             figsize=(4 * len(plot_features), 4 * len(target_classes)))

    for i, cat in enumerate(target_classes):
        real_subset = df_real[df_real['category'] == cat]
        synth_subset = df_synth[df_synth['category'] == cat]

        for j, feat in enumerate(plot_features):
            ax = axes[i][j]

            if len(real_subset) > 0 and feat in real_subset.columns:
                real_vals = real_subset[feat].dropna()
                # Clip to 99th percentile to avoid extreme outliers
                clip_val = real_vals.quantile(0.99) if len(real_vals) > 0 else 1
                real_vals = real_vals.clip(upper=clip_val)
                ax.hist(real_vals, bins=50, alpha=0.5, color=COLOURS[cat],
                       label='Real', density=True)

            if len(synth_subset) > 0 and feat in synth_subset.columns:
                synth_vals = synth_subset[feat].dropna()
                synth_vals = synth_vals.clip(upper=clip_val)
                ax.hist(synth_vals, bins=50, alpha=0.5, color='black',
                       label='Synthetic', density=True)

            if i == 0:
                ax.set_title(feat.replace('_', ' ').title(), fontsize=9)
            if j == 0:
                ax.set_ylabel(cat, fontsize=10, fontweight='bold')
            ax.legend(fontsize=6)
            ax.tick_params(labelsize=7)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved feature distributions to {path}")
    plt.close()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SYNTHETIC DATA QUALITY VISUALISATION")
    print("=" * 60)

    # Load data
    df_real = load_real_data()
    df_synth = load_synthetic_data()

    print(f"\nReal data: {len(df_real):,} rows")
    print(f"Real categories: {df_real['category'].value_counts().to_string()}")
    print(f"\nSynthetic data: {len(df_synth):,} rows")
    print(f"Synthetic categories: {df_synth['category'].value_counts().to_string()}")

    # Sample for visualisation
    n_per_cat = SAMPLE_REAL // len(COLOURS)
    df_real_sampled = sample_by_category(df_real, n_per_cat)
    df_synth_sampled = sample_by_category(df_synth, n_per_cat)

    X_real = df_real_sampled[FEATURES].values
    X_synth = df_synth_sampled[FEATURES].values
    labels_real = df_real_sampled['category'].values
    labels_synth = df_synth_sampled['category'].values

    print(f"\nSampled real: {len(X_real)} rows")
    print(f"Sampled synthetic: {len(X_synth)} rows")

    # Generate all plots
    plot_pca(X_real, X_synth, labels_real, labels_synth)
    plot_tsne(X_real, X_synth, labels_real, labels_synth)
    plot_per_class_comparison(X_real, X_synth, labels_real, labels_synth)
    plot_feature_distributions(df_real, df_synth)

    print("\n" + "=" * 60)
    print("ALL VISUALISATIONS COMPLETE")
    print(f"Saved to {OUTPUT_DIR}")
    print("=" * 60)