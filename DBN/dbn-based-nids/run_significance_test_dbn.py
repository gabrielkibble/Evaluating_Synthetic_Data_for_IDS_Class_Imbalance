"""
Statistical Significance Testing — CICIDS2017 MLP
==================================================

Runs MLP baseline and Markov T=1.0 augmented conditions
across 5 random seeds, then performs a paired t-test.

Usage:
    1. Make sure ./data/processed points to CICIDS2017 data
    2. Run: dbn_ids/bin/python run_significance_test.py

Run from: ~/Dissertation/DBN/dbn-based-nids/
"""

import subprocess
import re
import numpy as np
from scipy import stats
import json
import os
import shutil
import time

# ============================================================
# CONFIGURATION
# ============================================================

SEEDS = [42, 123, 456, 789, 1024]
CONFIG_PATH = "./configs/deepBeliefNetwork.json"
PYTHON = "dbn_ids/bin/python"

# Paths to baseline and augmented training data
PROCESSED_DIR = "./data/processed_cicids"
TRAIN_FEATURES = os.path.join(PROCESSED_DIR, "train/train_features.pkl")
TRAIN_LABELS = os.path.join(PROCESSED_DIR, "train/train_labels.pkl")
BASELINE_FEATURES = os.path.join(PROCESSED_DIR, "train/train_features_baseline.pkl")
BASELINE_LABELS = os.path.join(PROCESSED_DIR, "train/train_labels_baseline.pkl")


def set_seed_in_config(config_path, seed, output_path):
    """Load config, set seed, save to temp file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    config['seed'] = seed
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=4)


def extract_macro_f1(output_text):
    """Extract test set macro avg F1 from model output."""
    lines = output_text.split('\n')
    # Find the LAST "macro avg" line (test set report)
    macro_f1 = None
    for line in lines:
        if 'macro avg' in line:
            parts = line.split()
            # macro avg  precision  recall  f1-score  support
            # Find the f1-score (3rd number after "macro avg")
            numbers = re.findall(r'\d+\.\d+', line)
            if len(numbers) >= 3:
                macro_f1 = float(numbers[2])  # f1-score is the 3rd number
    return macro_f1


def swap_training_data(use_baseline):
    """Swap training data between baseline and augmented."""
    if use_baseline:
        # Copy baseline data into active training slots
        if os.path.exists(BASELINE_FEATURES):
            shutil.copy2(BASELINE_FEATURES, TRAIN_FEATURES)
            shutil.copy2(BASELINE_LABELS, TRAIN_LABELS)
            print("  Swapped to BASELINE training data")
        else:
            print("  WARNING: No baseline backup found!")
    else:
        # Need to re-run preprocessing with synthetic injection
        # Or if you have the augmented data saved separately:
        augmented_features = os.path.join(PROCESSED_DIR, "train/train_features_augmented.pkl")
        augmented_labels = os.path.join(PROCESSED_DIR, "train/train_labels_augmented.pkl")
        if os.path.exists(augmented_features):
            shutil.copy2(augmented_features, TRAIN_FEATURES)
            shutil.copy2(augmented_labels, TRAIN_LABELS)
            print("  Swapped to AUGMENTED training data")
        else:
            print("  WARNING: No augmented backup found!")
            print("  Run preprocessing with synthetic injection first,")
            print("  then copy the resulting train_features.pkl to train_features_augmented.pkl")


def run_experiment(config_path, seed, label):
    """Run a single training run and extract macro F1."""
    temp_config = f"./configs/temp_seed_{seed}.json"
    set_seed_in_config(config_path, seed, temp_config)

    print(f"\n  Running {label} with seed {seed}...")
    start = time.time()

    result = subprocess.run(
        [PYTHON, 'main.py', '--config', temp_config],
        capture_output=True, text=True
    )

    elapsed = time.time() - start
    output = result.stdout + result.stderr

    macro_f1 = extract_macro_f1(output)

    # Clean up temp config
    if os.path.exists(temp_config):
        os.remove(temp_config)

    if macro_f1 is not None:
        print(f"  Seed {seed}: Macro F1 = {macro_f1:.4f} ({elapsed:.0f}s)")
    else:
        print(f"  Seed {seed}: FAILED to extract Macro F1")
        print("  Last 20 lines of output:")
        for line in output.split('\n')[-20:]:
            print(f"    {line}")

    return macro_f1


def main():
    print("=" * 60)
    print("STATISTICAL SIGNIFICANCE TEST")
    print("CICIDS2017 MLP: Baseline vs Markov T=1.0")
    print(f"Seeds: {SEEDS}")
    print("=" * 60)

    # ============================================================
    # Phase 1: Run baseline (no augmentation) across 5 seeds
    # ============================================================
    print("\n\n--- PHASE 1: BASELINE (no augmentation) ---\n")
    swap_training_data(use_baseline=True)

    baseline_results = []
    for seed in SEEDS:
        f1 = run_experiment(CONFIG_PATH, seed, "Baseline")
        if f1 is not None:
            baseline_results.append(f1)

    # ============================================================
    # Phase 2: Run augmented (Markov T=1.0) across 5 seeds
    # ============================================================
    print("\n\n--- PHASE 2: MARKOV T=1.0 (augmented) ---\n")
    swap_training_data(use_baseline=False)

    augmented_results = []
    for seed in SEEDS:
        f1 = run_experiment(CONFIG_PATH, seed, "Markov T=1.0")
        if f1 is not None:
            augmented_results.append(f1)

    # ============================================================
    # Phase 3: Statistical analysis
    # ============================================================
    print("\n\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\nBaseline F1 scores:  {baseline_results}")
    print(f"Augmented F1 scores: {augmented_results}")

    if len(baseline_results) == len(SEEDS) and len(augmented_results) == len(SEEDS):
        baseline_arr = np.array(baseline_results)
        augmented_arr = np.array(augmented_results)

        print(f"\nBaseline:  mean={baseline_arr.mean():.4f}, std={baseline_arr.std():.4f}")
        print(f"Augmented: mean={augmented_arr.mean():.4f}, std={augmented_arr.std():.4f}")
        print(f"Mean improvement: +{(augmented_arr.mean() - baseline_arr.mean()):.4f}")

        t_stat, p_value = stats.ttest_rel(augmented_arr, baseline_arr)
        print(f"\nPaired t-test:")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value:     {p_value:.6f}")
        print(f"  Significant at p<0.05: {p_value < 0.05}")
        print(f"  Significant at p<0.01: {p_value < 0.01}")

        # Effect size (Cohen's d for paired samples)
        diff = augmented_arr - baseline_arr
        cohens_d = diff.mean() / diff.std()
        print(f"  Cohen's d:   {cohens_d:.4f}")
    else:
        print("\nERROR: Not all runs completed successfully.")
        print(f"  Baseline: {len(baseline_results)}/{len(SEEDS)} completed")
        print(f"  Augmented: {len(augmented_results)}/{len(SEEDS)} completed")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()