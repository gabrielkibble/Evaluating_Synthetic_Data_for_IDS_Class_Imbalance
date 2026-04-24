"""
SMOTE Comparison Experiment for CICIDS2017
==========================================

Applies SMOTE oversampling to the CICIDS2017 training set to produce the
same class distribution as the Markov chain synthetic augmentation.
This enables a direct comparison: same target counts, same test set,
same model configs — only the augmentation method differs.

Usage:
    1. First run the baseline preprocessing (no synthetic injection):
       dbn_ids/bin/python preprocessing/cicids2017.py
       (with synthetic CSV removed/renamed so injection is skipped)

    2. Run this script to create SMOTE-augmented training data:
       dbn_ids/bin/python smote_comparison.py

    3. Train models:
       dbn_ids/bin/python main.py --config ./configs/deepBeliefNetwork.json
       dbn_ids/bin/python main.py --config ./configs/multilayerPerceptron.json

    4. Compare results against Markov chain synthetic augmentation.

Run from: ~/Dissertation/DBN/dbn-based-nids/
"""

import pandas as pd
import numpy as np
import os
import time

from imblearn.over_sampling import SMOTE
from collections import Counter

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "./data"
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# These are the class counts from the Markov chain synthetic run
# (from the preprocessing output when synthetic data was injected)
# We match these exactly so the comparison is fair
TARGET_DISTRIBUTION = {
    0: 621402,   # Benign (undersampled from ~810k)
    1: 13113,    # Botnet ARES (oversampled from ~1,174)
    2: 95438,    # Brute Force (oversampled from ~6,545)
    3: 172847,   # DoS/DDoS (kept roughly same)
    4: 95339,    # PortScan (kept roughly same)
    5: 88583,    # Web Attack (oversampled from ~1,308)
}

LABEL_NAMES = {
    0: "Benign",
    1: "Botnet ARES",
    2: "Brute Force",
    3: "DoS/DDoS",
    4: "PortScan",
    5: "Web Attack",
}

# Total should match: 1,086,722
TOTAL_TARGET = sum(TARGET_DISTRIBUTION.values())


def main():
    print("=" * 60)
    print("SMOTE COMPARISON EXPERIMENT")
    print("=" * 60)

    # ============================================================
    # 1. Load baseline training data (no synthetic injection)
    # ============================================================
    print("\n[1/4] Loading baseline training data...")

    X_train = pd.read_pickle(os.path.join(PROCESSED_DIR, "train/train_features.pkl"))
    y_train = pd.read_pickle(os.path.join(PROCESSED_DIR, "train/train_labels.pkl"))

    print(f"Training set shape: {X_train.shape}")
    print(f"\nBaseline class distribution:")
    for label_id, count in sorted(Counter(y_train['label'].values).items()):
        name = LABEL_NAMES.get(label_id, f"Unknown({label_id})")
        print(f"  {name}: {count:,}")

    # ============================================================
    # 2. Determine SMOTE targets
    # ============================================================
    print("\n[2/4] Computing SMOTE targets...")

    current_counts = Counter(y_train['label'].values)

    # SMOTE can only oversample — it can't reduce majority class
    # So we first apply SMOTE to bring minority classes up,
    # then undersample Benign to match target

    # For SMOTE: set each class to max(current, target)
    # Classes that are already above target stay as-is during SMOTE
    smote_targets = {}
    for label_id, target_count in TARGET_DISTRIBUTION.items():
        current = current_counts.get(label_id, 0)
        if target_count > current:
            smote_targets[label_id] = target_count
            print(f"  {LABEL_NAMES[label_id]}: {current:,} -> {target_count:,} (SMOTE)")
        else:
            smote_targets[label_id] = current  # Keep current count for SMOTE phase
            print(f"  {LABEL_NAMES[label_id]}: {current:,} (no SMOTE needed)")

    # ============================================================
    # 3. Apply SMOTE
    # ============================================================
    print("\n[3/4] Applying SMOTE (this may take a few minutes)...")

    X_array = X_train.values.astype(np.float32)
    y_array = y_train['label'].values

    # Replace any NaN/inf
    X_array = np.nan_to_num(X_array, nan=0.0, posinf=0.0, neginf=0.0)

    start_time = time.time()

    smote = SMOTE(
        sampling_strategy=smote_targets,
        random_state=42,
    )

    X_resampled, y_resampled = smote.fit_resample(X_array, y_array)

    print(f"SMOTE completed in {time.time() - start_time:.1f}s")
    print(f"Resampled shape: {X_resampled.shape}")

    print(f"\nPost-SMOTE distribution:")
    for label_id, count in sorted(Counter(y_resampled).items()):
        name = LABEL_NAMES.get(label_id, f"Unknown({label_id})")
        print(f"  {name}: {count:,}")

    # ============================================================
    # 4. Undersample Benign to match target total
    # ============================================================
    print("\n[4/4] Undersampling Benign to maintain constant training size...")

    benign_label = 0
    benign_mask = y_resampled == benign_label
    non_benign_mask = ~benign_mask

    n_non_benign = non_benign_mask.sum()
    n_benign_target = TARGET_DISTRIBUTION[benign_label]

    # Get indices
    benign_indices = np.where(benign_mask)[0]
    non_benign_indices = np.where(non_benign_mask)[0]

    # Randomly sample benign to target count
    rng = np.random.RandomState(42)
    benign_keep = rng.choice(benign_indices, size=n_benign_target, replace=False)

    # Combine and sort
    keep_indices = np.sort(np.concatenate([benign_keep, non_benign_indices]))

    X_final = X_resampled[keep_indices]
    y_final = y_resampled[keep_indices]

    # Shuffle
    shuffle_idx = rng.permutation(len(X_final))
    X_final = X_final[shuffle_idx]
    y_final = y_final[shuffle_idx]

    print(f"\nFinal training set size: {len(X_final):,} (target: {TOTAL_TARGET:,})")
    print(f"\nFinal class distribution:")
    for label_id, count in sorted(Counter(y_final).items()):
        name = LABEL_NAMES.get(label_id, f"Unknown({label_id})")
        target = TARGET_DISTRIBUTION[label_id]
        match = "✓" if count == target else "✗"
        print(f"  {name}: {count:,} (target: {target:,}) {match}")

    # ============================================================
    # 5. Save (overwrite training data only — val/test untouched)
    # ============================================================
    print("\nSaving SMOTE-augmented training data...")

    # Back up originals
    backup_feat = os.path.join(PROCESSED_DIR, "train/train_features_baseline.pkl")
    backup_label = os.path.join(PROCESSED_DIR, "train/train_labels_baseline.pkl")

    if not os.path.exists(backup_feat):
        print("  Backing up baseline training data...")
        X_train.to_pickle(backup_feat)
        y_train.to_pickle(backup_label)
    else:
        print("  Backup already exists, skipping.")

    # Save SMOTE versions
    X_final_df = pd.DataFrame(X_final, columns=X_train.columns)
    y_final_df = pd.DataFrame(y_final, columns=['label'])

    X_final_df.to_pickle(os.path.join(PROCESSED_DIR, "train/train_features.pkl"))
    y_final_df.to_pickle(os.path.join(PROCESSED_DIR, "train/train_labels.pkl"))

    print("  Saved SMOTE-augmented training data.")
    print("\n" + "=" * 60)
    print("DONE. Now run:")
    print("  dbn_ids/bin/python main.py --config ./configs/deepBeliefNetwork.json")
    print("  dbn_ids/bin/python main.py --config ./configs/multilayerPerceptron.json")
    print("")
    print("To restore baseline, run:")
    print(f"  cp {backup_feat} {os.path.join(PROCESSED_DIR, 'train/train_features.pkl')}")
    print(f"  cp {backup_label} {os.path.join(PROCESSED_DIR, 'train/train_labels.pkl')}")
    print("=" * 60)


if __name__ == "__main__":
    main()