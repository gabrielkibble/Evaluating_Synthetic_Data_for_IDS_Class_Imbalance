"""
SMOTE Comparison Experiment for UNSW-NB15
==========================================

Applies SMOTE oversampling to the UNSW-NB15 training set to produce the
same class distribution as the Markov chain synthetic augmentation.

Usage:
    1. Make sure ./data/processed points to UNSW (or that your preprocessing
       has been run fresh without synthetic injection):
       mv synthetic_unsw_standard.csv synthetic_unsw_standard.csv.bak
       dbn_ids/bin/python preprocessing/unsw_nb15.py

    2. Run this script:
       dbn_ids/bin/python smote_comparison_unsw.py

    3. Train models:
       dbn_ids/bin/python main.py --config ./configs/deepBeliefNetwork_unsw.json
       dbn_ids/bin/python main.py --config ./configs/multilayerPerceptron_unsw.json

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
PROCESSED_DIR = os.path.join(DATA_DIR, "processed_unsw")

# Target distribution matching the Markov chain synthetic UNSW run
# From generation targets: Analysis (20k), Backdoors (20k), Shellcode (20k)
# Plus undersample Normal proportionally
# Original UNSW training distribution (after preprocessing):
#   Normal: 547,041
#   Generic: 16,851
#   Exploits: 9,803
#   Fuzzers: 5,223
#   DoS: 4,384
#   Reconnaissance: 3,055
#   Analysis: 1,834
#   Backdoors: 1,622
#   Shellcode: 1,314
#
# After Markov chain injection (with 20k synth per target, undersampling Normal):
#   Normal: ~487,000 (undersampled by ~60k to keep total constant)
#   Generic: 16,851
#   Exploits: 9,803
#   Fuzzers: 5,223
#   DoS: 4,384
#   Reconnaissance: 3,055
#   Analysis: 20,000+1,834 → ~21,834
#   Backdoors: 20,000+1,622 → ~21,622
#   Shellcode: 20,000+1,314 → ~21,314

TARGET_DISTRIBUTION = {
    0: 21834,    # Analysis (oversampled from 1,834)
    1: 21622,    # Backdoors (oversampled from 1,622)
    2: 4384,     # DoS (unchanged)
    3: 9803,     # Exploits (unchanged)
    4: 5223,     # Fuzzers (unchanged)
    5: 16851,    # Generic (unchanged)
    6: 487241,   # Normal (undersampled to keep total constant)
    7: 3055,     # Reconnaissance (unchanged)
    8: 21314,    # Shellcode (oversampled from 1,314)
}

LABEL_NAMES = {
    0: "Analysis",
    1: "Backdoors",
    2: "DoS",
    3: "Exploits",
    4: "Fuzzers",
    5: "Generic",
    6: "Normal",
    7: "Reconnaissance",
    8: "Shellcode",
}

# Total should roughly match: ~591,127
TOTAL_TARGET = sum(TARGET_DISTRIBUTION.values())


def main():
    print("=" * 60)
    print("SMOTE COMPARISON EXPERIMENT — UNSW-NB15")
    print("=" * 60)

    # ============================================================
    # 1. Load baseline training data
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
    # 2. Determine SMOTE targets (only oversample minorities)
    # ============================================================
    print("\n[2/4] Computing SMOTE targets...")

    current_counts = Counter(y_train['label'].values)

    # SMOTE can only oversample, so we set each class to max(current, target)
    # Classes that need undersampling (Normal) are handled after SMOTE
    smote_targets = {}
    for label_id, target_count in TARGET_DISTRIBUTION.items():
        current = current_counts.get(label_id, 0)
        if target_count > current:
            smote_targets[label_id] = target_count
            print(f"  {LABEL_NAMES[label_id]}: {current:,} -> {target_count:,} (SMOTE)")
        else:
            smote_targets[label_id] = current
            print(f"  {LABEL_NAMES[label_id]}: {current:,} (no SMOTE needed)")

    # ============================================================
    # 3. Apply SMOTE
    # ============================================================
    print("\n[3/4] Applying SMOTE (this may take a while)...")

    X_array = X_train.values.astype(np.float32)
    y_array = y_train['label'].values

    X_array = np.nan_to_num(X_array, nan=0.0, posinf=0.0, neginf=0.0)

    start_time = time.time()

    smote = SMOTE(
        sampling_strategy=smote_targets,
        random_state=42,
        k_neighbors=5  # Smaller for speed, still effective
    )

    X_resampled, y_resampled = smote.fit_resample(X_array, y_array)

    smote_time = time.time() - start_time
    print(f"SMOTE completed in {smote_time:.1f}s")
    print(f"Resampled shape: {X_resampled.shape}")

    print(f"\nPost-SMOTE distribution:")
    for label_id, count in sorted(Counter(y_resampled).items()):
        name = LABEL_NAMES.get(label_id, f"Unknown({label_id})")
        print(f"  {name}: {count:,}")

    # ============================================================
    # 4. Undersample Normal to match target total
    # ============================================================
    print("\n[4/4] Undersampling Normal to maintain constant training size...")

    normal_label = 6
    normal_mask = y_resampled == normal_label
    non_normal_mask = ~normal_mask

    normal_indices = np.where(normal_mask)[0]
    non_normal_indices = np.where(non_normal_mask)[0]

    n_normal_target = TARGET_DISTRIBUTION[normal_label]

    rng = np.random.RandomState(42)
    normal_keep = rng.choice(normal_indices, size=n_normal_target, replace=False)

    keep_indices = np.sort(np.concatenate([normal_keep, non_normal_indices]))

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
    # 5. Save with backup
    # ============================================================
    print("\nSaving SMOTE-augmented training data...")

    backup_feat = os.path.join(PROCESSED_DIR, "train/train_features_baseline.pkl")
    backup_label = os.path.join(PROCESSED_DIR, "train/train_labels_baseline.pkl")

    if not os.path.exists(backup_feat):
        print("  Backing up baseline training data...")
        X_train.to_pickle(backup_feat)
        y_train.to_pickle(backup_label)
    else:
        print("  Backup already exists, skipping.")

    X_final_df = pd.DataFrame(X_final, columns=X_train.columns)
    y_final_df = pd.DataFrame(y_final, columns=['label'])

    X_final_df.to_pickle(os.path.join(PROCESSED_DIR, "train/train_features.pkl"))
    y_final_df.to_pickle(os.path.join(PROCESSED_DIR, "train/train_labels.pkl"))

    print("  Saved SMOTE-augmented training data.")
    print(f"\n=== GENERATION TIME: {smote_time:.1f}s ({smote_time/60:.1f} minutes) ===")
    print("\n" + "=" * 60)
    print("DONE. Now run:")
    print("  # Make sure ./data/processed points to UNSW first:")
    print("  rm ./data/processed")
    print("  ln -s processed_unsw ./data/processed")
    print("")
    print("  dbn_ids/bin/python main.py --config ./configs/deepBeliefNetwork_unsw.json")
    print("  dbn_ids/bin/python main.py --config ./configs/multilayerPerceptron_unsw.json")
    print("")
    print("To restore baseline, run:")
    print(f"  cp {backup_feat} {os.path.join(PROCESSED_DIR, 'train/train_features.pkl')}")
    print(f"  cp {backup_label} {os.path.join(PROCESSED_DIR, 'train/train_labels.pkl')}")
    print("=" * 60)


if __name__ == "__main__":
    main()