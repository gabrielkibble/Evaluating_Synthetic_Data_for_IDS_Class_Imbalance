# ==========
# Imports
# ==========
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import time
import warnings
from scipy.special import softmax
import glob
import os
from tqdm import tqdm


# ============
# File Paths
# ============

FULL_SOURCE_FILES = [
    "UNSW-NB15_1.csv",
    "UNSW-NB15_2.csv",
    "UNSW-NB15_3.csv",
    "UNSW-NB15_4.csv"
]

BALANCED_FILE = "balanced_NB15_iot.csv"
IMBALANCED_FILES = FULL_SOURCE_FILES

RAW_COLUMN_NAMES = [
    "srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes",
    "dbytes", "sttl", "dttl", "sloss", "dloss", "service", "sload", "dload",
    "spkts", "dpkts", "swin", "dwin", "stcpb", "dtcpb", "smean", "dmean",
    "trans_depth", "res_bdy_len", "sjit", "djit", "stime", "ltime", "sinpkt",
    "dinpkt", "tcprtt", "synack", "ackdat", "is_sm_ips_ports", "ct_state_ttl",
    "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd", "ct_srv_src",
    "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm",
    "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat", "Label"
]


# ========================
# Toggles and Parameters
# ========================

RECREATE_BALANCED_DATA = False
RECREATE_SEQUENCES = True

TARGET_SAMPLES_PER_CLASS = 300
SEQ_LENGTH = 20
STRIDE = 10
TEMPERATURE = 1.2

# ========================
# Generation Targets
# ========================
# Current UNSW-NB15 distribution:
#   Normal:         2,218,764  (87.3%) - NO generation needed
#   Generic:          215,481  ( 8.5%) - NO generation needed
#   Exploits:          44,525  ( 1.8%) - NO generation needed
#   Fuzzers:           24,246  ( 1.0%) - NO generation needed
#   DoS:               16,353  ( 0.6%) - NO generation needed
#   Reconnaissance:    13,987  ( 0.6%) - NO generation needed
#   Analysis:           2,677  ( 0.1%) - TARGET: ~20,000
#   Backdoors:          2,329  ( 0.09%) - TARGET: ~20,000 (merged Backdoor + Backdoors)
#   Shellcode:          1,511  ( 0.06%) - TARGET: ~20,000
#   Worms:                174  - DROPPED (too few)
#
# Adds ~60k synthetic rows, bringing minority classes in line with DoS/Recon

GENERATION_TARGETS = {
    "Analysis":  20_000,
    "Backdoors": 20_000,
    "Shellcode": 20_000,
}

FEATURES = [
    "dur",
    "spkts", "dpkts",
    "sbytes", "dbytes",
    "sload", "dload",
    "smean", "dmean",
    "sjit", "djit",
    "sinpkt", "dinpkt"
]


# ============================
# Building Balanced Dataset
# ============================

def create_balanced_dataset(input_path_list, output_path, target_samples):
    print("\n--- Creating Balanced Dataset from UNSW-NB15 Files ---")

    data_buckets = {
        "Normal": [], "Generic": [], "Exploits": [], "Fuzzers": [],
        "DoS": [], "Reconnaissance": [], "Analysis": [],
        "Backdoors": [], "Shellcode": []
    }

    chunk_size = 100000
    all_done = False

    for file_path in input_path_list:
        if all_done:
            break
        print(f"Scanning file: {file_path}...")

        try:
            for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False,
                                     names=RAW_COLUMN_NAMES, header=None):
                if 'attack_cat' in chunk.columns:
                    chunk.rename(columns={'attack_cat': 'category'}, inplace=True)

                if 'category' not in chunk.columns:
                    continue

                # Clean labels
                chunk['category'] = chunk.apply(
                    lambda row: 'Normal' if pd.isna(row['category']) and row['Label'] == 0
                    else str(row['category']).strip(), axis=1
                )
                # Merge Backdoor -> Backdoors
                chunk['category'] = chunk['category'].replace('Backdoor', 'Backdoors')

                for cat in data_buckets.keys():
                    current_count = sum(len(df) for df in data_buckets[cat])
                    if current_count < target_samples:
                        subset = chunk[chunk["category"] == cat]
                        if len(subset) > 0:
                            needed = target_samples - current_count
                            data_buckets[cat].append(subset.head(needed))

                total_counts = {k: sum(len(df) for df in v) for k, v in data_buckets.items()}
                print(f"Status: {total_counts}")

                if all(c >= target_samples for c in total_counts.values()):
                    print("All buckets full!")
                    all_done = True
                    break
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

    print("Combining extracted data...")
    frames = [pd.concat(bl) for bl in data_buckets.values() if bl]
    if not frames:
        print("No data found!")
        return

    balanced_df = pd.concat(frames).sample(frac=1, random_state=42).reset_index(drop=True)
    print("\nFinal Balanced Distribution:")
    print(balanced_df["category"].value_counts())
    balanced_df.to_csv(output_path, index=False)
    print(f"Saved to '{output_path}'")


if RECREATE_BALANCED_DATA:
    create_balanced_dataset(FULL_SOURCE_FILES, BALANCED_FILE, TARGET_SAMPLES_PER_CLASS)
else:
    print(f"\nSkipping data creation. Using existing '{BALANCED_FILE}'...")


# ================================================
# STEP 1 - LEARNING CLUSTERS OFF BALANCED DATA
# ================================================

print("\n--- PHASE 1: Learning Vocabulary (Balanced Data) ---")

df_bal = pd.read_csv(BALANCED_FILE)

# Merge Backdoor -> Backdoors in balanced data too
if 'category' in df_bal.columns:
    df_bal['category'] = df_bal['category'].replace('Backdoor', 'Backdoors')

X_bal = df_bal[FEATURES].replace([np.inf, -np.inf], np.nan).dropna()

scaler = StandardScaler()
X_bal_scaled = scaler.fit_transform(X_bal)

pca = PCA(n_components=5, random_state=42)
X_bal_pca = pca.fit_transform(X_bal_scaled)

dbscan = DBSCAN(eps=0.1, min_samples=5, n_jobs=-1)
bal_clusters = dbscan.fit_predict(X_bal_pca)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_bal_pca, bal_clusters)

print(f"Vocabulary Learned: {len(np.unique(bal_clusters))} Clusters Found.")
print(pd.Series(bal_clusters).value_counts())

stats = X_bal.copy()
stats["cluster"] = bal_clusters
stats["category"] = df_bal.loc[X_bal.index, "category"]


def get_majority_label(cid, df_labeled):
    if cid == -1:
        return "Noise"
    cluster_data = df_labeled[df_labeled['cluster'] == cid]
    majority_category = cluster_data['category'].mode()[0]
    mapping = {
        "Reconnaissance": "Recon (Slow)",
        "Fuzzers":        "Recon (Active/Fuzzing)",
        "DoS":            "Impact (DoS)",
        "Exploits":       "Initial Access (Exploit)",
        "Backdoors":      "Persistence (Backdoor)",
        "Shellcode":      "Execution (Shellcode)",
        "Analysis":       "Discovery (Analysis)",
        "Generic":        "Unknown/Generic",
        "Normal":         "Benign"
    }
    return mapping.get(majority_category, "Unknown")


cluster_to_mitre = {}
unique_clusters = sorted(stats["cluster"].unique())
for cid in unique_clusters:
    label = get_majority_label(cid, stats)
    cluster_to_mitre[cid] = label
    print(f"Cluster {cid}: {label}")

le = LabelEncoder()
le.fit(np.unique(bal_clusters))


# ==============================================
# STEP 2 - LEARN A MODEL OFF IMBALANCED DATA
# ==============================================

print("\n--- PHASE 2: Learning Grammar (Imbalanced Data) ---")

IMBALANCED_FILE_MAIN = []
for file_path in IMBALANCED_FILES:
    print(f"Reading {file_path}...")
    imbalanced_temp = pd.read_csv(file_path, low_memory=False, names=RAW_COLUMN_NAMES, header=None)
    IMBALANCED_FILE_MAIN.append(imbalanced_temp)

df_imbal = pd.concat(IMBALANCED_FILE_MAIN, ignore_index=True)

if 'attack_cat' in df_imbal.columns:
    df_imbal.rename(columns={'attack_cat': 'category'}, inplace=True)

df_imbal['category'] = df_imbal.apply(
    lambda row: 'Normal' if pd.isna(row['category']) and row['Label'] == 0
    else str(row['category']).strip(), axis=1
)

# Merge Backdoor -> Backdoors and drop Worms
df_imbal['category'] = df_imbal['category'].replace('Backdoor', 'Backdoors')
df_imbal = df_imbal[df_imbal['category'] != 'Worms']

for col in FEATURES:
    df_imbal[col] = pd.to_numeric(df_imbal[col], errors='coerce')

X_imbal = df_imbal[FEATURES].replace([np.inf, -np.inf], np.nan).dropna()
print(f"Loaded Imbalanced Data: {X_imbal.shape}")

X_imbal_scaled = scaler.transform(X_imbal)
X_imbal_pca = pca.transform(X_imbal_scaled)

df_imbal = df_imbal.loc[X_imbal.index].copy()

imbal_clusters = knn.predict(X_imbal_pca)
df_imbal["cluster"] = imbal_clusters

print("Real-World Cluster Distribution (Should be skewed):")
print(pd.Series(imbal_clusters).value_counts())


# ================================
# STEP 2.5 - SUPERVISED MARKOV CHAIN
# ================================

class SupervisedMarkovChain:
    def __init__(self):
        self.transmat_ = None
        self.states = []
        self.state_to_idx = {}
        self.idx_to_state = {}
        self.state_feature_stats = {}

    def fit(self, df, cluster_col='cluster', label_col='category', feature_cols=None):
        print("Building Composite States (Cluster + Label)...")
        df['composite_state'] = df[cluster_col].astype(str) + "_" + df[label_col].astype(str)

        unique_states = sorted(df['composite_state'].unique())
        self.states = unique_states
        self.state_to_idx = {state: i for i, state in enumerate(unique_states)}
        self.idx_to_state = {i: state for i, state in enumerate(unique_states)}
        n_states = len(unique_states)
        print(f"Identified {n_states} unique composite states.")

        trans_counts = np.ones((n_states, n_states)) * 1e-6

        print("Learning transition probabilities...")
        grouped = df.sort_values("stime").groupby("srcip")
        for srcip, group in tqdm(grouped, desc="Processing source IPs"):
            seq = group['composite_state'].values
            if len(seq) < 2:
                continue
            indices = [self.state_to_idx[s] for s in seq]
            for t in range(len(indices) - 1):
                trans_counts[indices[t], indices[t + 1]] += 1

        self.transmat_ = trans_counts / trans_counts.sum(axis=1, keepdims=True)

        if feature_cols:
            print("Calculating robust feature profiles...")
            for state in tqdm(unique_states, desc="Profiling states"):
                subset = df[df['composite_state'] == state][feature_cols]
                if len(subset) > 1:
                    mean_vec = subset.mean().values
                    cov_mx = subset.cov().values + (np.eye(len(feature_cols)) * 1e-6)
                else:
                    mean_vec = subset.iloc[0].values
                    cov_mx = np.eye(len(feature_cols)) * 1e-6
                self.state_feature_stats[state] = {'mean': mean_vec, 'cov': cov_mx}

        print("Supervised Markov Chain Trained Successfully.")

    def generate_sequence(self, start_state, length=10):
        if start_state not in self.state_to_idx:
            raise ValueError(f"Start state '{start_state}' not found.")
        current_idx = self.state_to_idx[start_state]
        sequence = [start_state]
        for _ in range(length - 1):
            probs = self.transmat_[current_idx]
            next_idx = np.random.choice(len(self.states), p=probs)
            sequence.append(self.idx_to_state[next_idx])
            current_idx = next_idx
        return sequence


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def print_available_composites(model):
    print("\n--- AVAILABLE COMPOSITE STATES ---")
    organized = {}
    for state in model.states:
        parts = state.split('_', 1)
        label = parts[-1]
        if label not in organized:
            organized[label] = []
        organized[label].append(state)
    for label, states in organized.items():
        print(f"\n[{label.upper()}]:")
        for i in range(0, len(states), 5):
            print(f"  {states[i:i+5]}")
    print("\n----------------------------------")


def get_composite_states_for_category(model, category):
    return [s for s in model.states if s.split('_', 1)[1] == category]


def get_scaled_transition_matrix(trans_mat, T=1.0):
    if T == 1.0:
        return trans_mat.copy()
    scaled = np.power(trans_mat, 1.0 / T)
    row_sums = scaled.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return scaled / row_sums


def generate_minority_traffic(model, df_original, generation_targets, seq_len=20,
                              temperature=1.0, output_file="synthetic_unsw_nb15.csv"):
    """
    Generates synthetic UNSW-NB15 traffic focused on minority classes.
    OPTIMIZED: Pre-indexes pools, pre-computes Cholesky, batches random indices.
    """

    print(f"\n{'='*60}")
    print(f"SYNTHETIC DATA GENERATION (Temperature={temperature})")
    print(f"{'='*60}")

    total_target = sum(generation_targets.values())
    print(f"\nGeneration plan:")
    for cat, target in generation_targets.items():
        states = get_composite_states_for_category(model, cat)
        print(f"  {cat}: {target:,} samples across {len(states)} composite states")
    print(f"  Total: {total_target:,} synthetic samples\n")

    # UNSW-NB15 column order (49 columns)
    final_columns = [
        "srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes",
        "dbytes", "sttl", "dttl", "sloss", "dloss", "service", "sload", "dload",
        "spkts", "dpkts", "swin", "dwin", "stcpb", "dtcpb", "smean", "dmean",
        "trans_depth", "res_bdy_len", "sjit", "djit", "stime", "ltime", "sinpkt",
        "dinpkt", "tcprtt", "synack", "ackdat", "is_sm_ips_ports", "ct_state_ttl",
        "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd", "ct_srv_src",
        "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm",
        "ct_dst_sport_ltm", "ct_dst_src_ltm", "category", "Label"
    ]

    # =============================================
    # OPTIMIZATION 1: Pre-index all template pools
    # =============================================
    print("Pre-indexing template pools (one-time cost)...")
    pool_cache = {}
    for state in tqdm(model.states, desc="Caching pools"):
        pool_df = df_original[df_original['composite_state'] == state]
        if not pool_df.empty:
            pool_cache[state] = pool_df.values
            pool_cache[state + '_cols'] = pool_df.columns.tolist()
            pool_cache[state + '_len'] = len(pool_df)

    # =============================================
    # OPTIMIZATION 2: Pre-compute Cholesky decompositions
    # =============================================
    print("Pre-computing Cholesky decompositions...")
    cholesky_cache = {}
    for state, feat_stats in model.state_feature_stats.items():
        cov = feat_stats['cov'] + np.eye(len(feat_stats['mean'])) * 1e-6
        cov = (cov + cov.T) / 2
        try:
            cholesky_cache[state] = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            cholesky_cache[state] = None

    all_rows = []

    original_transmat = model.transmat_.copy()
    if temperature != 1.0:
        model.transmat_ = get_scaled_transition_matrix(original_transmat, T=temperature)

    overall_pbar = tqdm(total=total_target, desc="Total synthetic rows", unit="rows", position=0)

    try:
        for category, target_count in generation_targets.items():
            category_states = get_composite_states_for_category(model, category)

            if not category_states:
                print(f"\n  WARNING: No composite states found for '{category}'. Skipping.")
                continue

            # Weight distribution by real data count
            state_weights = {}
            for state in category_states:
                state_weights[state] = pool_cache.get(state + '_len', 0)

            total_weight = sum(state_weights.values())
            if total_weight == 0:
                print(f"\n  WARNING: No real data for '{category}' states. Skipping.")
                continue

            state_allocations = {}
            for state, weight in state_weights.items():
                proportion = weight / total_weight
                n_rows_needed = int(target_count * proportion)
                n_sequences = max(1, n_rows_needed // seq_len)
                state_allocations[state] = n_sequences

            print(f"\n  [{category}] Target: {target_count:,} rows")
            for state, n_seq in state_allocations.items():
                print(f"    {state}: {n_seq} sequences x {seq_len} steps = ~{n_seq * seq_len:,} rows")

            for start_node, n_sequences in state_allocations.items():
                if start_node not in pool_cache:
                    continue

                start_pool = pool_cache[start_node]
                start_cols = pool_cache[start_node + '_cols']
                start_pool_len = pool_cache[start_node + '_len']

                # =============================================
                # OPTIMIZATION 3: Batch-generate all sequences
                # =============================================
                all_sequences = []
                for _ in range(n_sequences):
                    try:
                        seq = model.generate_sequence(start_state=start_node, length=seq_len)
                        all_sequences.append(seq)
                    except:
                        continue

                # Pre-sample all template row indices at once
                template_indices = {}
                for state in model.states:
                    if state in pool_cache:
                        count = sum(seq.count(state) for seq in all_sequences)
                        if count > 0:
                            template_indices[state] = np.random.randint(
                                0, pool_cache[state + '_len'], size=count
                            )
                idx_counters = {state: 0 for state in template_indices}

                cat_pbar = tqdm(
                    all_sequences,
                    desc=f"    {start_node}",
                    unit="seq",
                    position=1,
                    leave=False
                )

                base_time = int(time.time())

                for seq_i, seq_states in enumerate(cat_pbar):
                    current_time = base_time + (seq_i * seq_len)

                    for t, state in enumerate(seq_states):
                        feat_stats = model.state_feature_stats.get(state)
                        if not feat_stats:
                            continue

                        # Fast template row lookup
                        if state in pool_cache and state in template_indices:
                            idx = idx_counters[state]
                            row_idx = template_indices[state][idx]
                            idx_counters[state] += 1
                            row_array = pool_cache[state][row_idx]
                            cols = pool_cache[state + '_cols']
                        else:
                            row_idx = np.random.randint(0, start_pool_len)
                            row_array = start_pool[row_idx]
                            cols = start_cols

                        template_row = dict(zip(cols, row_array))

                        # =============================================
                        # OPTIMIZATION 4: Fast feature sampling via Cholesky
                        # =============================================
                        L = cholesky_cache.get(state)
                        if L is not None:
                            z = np.random.standard_normal(len(feat_stats['mean']))
                            feats = feat_stats['mean'] + L @ z
                            feats = np.maximum(feats, 0)
                        else:
                            feats = np.maximum(feat_stats['mean'].copy(), 0)

                        gen_dur = feats[0]
                        gen_spkts = max(1, round(feats[1]))
                        gen_dpkts = max(0, round(feats[2]))
                        gen_sbytes = max(60, feats[3])
                        gen_dbytes = max(0, feats[4])

                        # Update template with generated features
                        template_row['dur'] = gen_dur
                        template_row['spkts'] = gen_spkts
                        template_row['dpkts'] = gen_dpkts
                        template_row['sbytes'] = gen_sbytes
                        template_row['dbytes'] = gen_dbytes
                        template_row['sload'] = feats[5]
                        template_row['dload'] = feats[6]
                        template_row['smean'] = feats[7]
                        template_row['dmean'] = feats[8]
                        template_row['sjit'] = feats[9]
                        template_row['djit'] = feats[10]
                        template_row['sinpkt'] = feats[11]
                        template_row['dinpkt'] = feats[12]

                        # Scale dependent fields
                        original_spkts = max(1, template_row.get('spkts', 1))
                        scaling_factor = gen_spkts / original_spkts
                        template_row['sloss'] = template_row.get('sloss', 0) * scaling_factor
                        template_row['dloss'] = template_row.get('dloss', 0) * scaling_factor

                        # Time physics
                        template_row['stime'] = current_time + (t * 0.5)
                        template_row['ltime'] = template_row['stime'] + gen_dur

                        # Assign label
                        state_label = state.split('_', 1)[1]
                        template_row['category'] = state_label
                        template_row['Label'] = 1  # All synthetic are attacks

                        all_rows.append(template_row)
                        overall_pbar.update(1)

                cat_pbar.close()

    finally:
        overall_pbar.close()
        if temperature != 1.0:
            model.transmat_ = original_transmat

    if all_rows:
        print("\nBuilding DataFrame from generated rows...")
        df_gen = pd.DataFrame(all_rows)

        for col in final_columns:
            if col not in df_gen.columns:
                df_gen[col] = 0
        df_gen = df_gen[final_columns]

        # Integer enforcement for UNSW-NB15
        int_cols = [
            "sport", "dsport", "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss",
            "spkts", "dpkts", "swin", "dwin", "stcpb", "dtcpb", "smean", "dmean",
            "trans_depth", "res_bdy_len", "is_sm_ips_ports", "ct_state_ttl",
            "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd", "ct_srv_src",
            "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm",
            "ct_dst_sport_ltm", "ct_dst_src_ltm", "Label"
        ]
        for col in int_cols:
            if col in df_gen.columns:
                df_gen[col] = pd.to_numeric(df_gen[col], errors='coerce').fillna(0).round().astype(int)

        print(f"Saving to {output_file}...")
        df_gen.to_csv(output_file, index=False)

        print(f"\n{'='*60}")
        print(f"GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"Total rows generated: {len(df_gen):,}")
        print(f"Saved to: {output_file}")
        print(f"\nBreakdown by category:")
        print(df_gen['category'].value_counts().to_string())
        print(f"\nThis will increase dataset from ~2.54M to ~{2_540_047 + len(df_gen):,} rows")
        print(f"Synthetic data proportion: {len(df_gen) / (2_540_047 + len(df_gen)):.1%}")

        return df_gen
    else:
        print("FAILED: No data generated.")
        return None


def inspect_transition_probabilities(model, state_name):
    if state_name not in model.state_to_idx:
        print(f"State {state_name} not found.")
        return
    idx = model.state_to_idx[state_name]
    probs = model.transmat_[idx]
    sorted_indices = np.argsort(probs)[::-1]
    print(f"\n--- TRANSITIONS FROM {state_name} ---")
    for i in range(5):
        next_idx = sorted_indices[i]
        prob = probs[next_idx]
        if prob > 0.001:
            next_state = model.idx_to_state[next_idx]
            print(f"  -> {next_state}: {prob:.2%} chance")


# ==========================================
# EXECUTION BLOCK
# ==========================================

print("\nChecking data types BEFORE fix:")
print(df_imbal[FEATURES].dtypes)

for col in FEATURES:
    df_imbal[col] = pd.to_numeric(df_imbal[col], errors='coerce')
df_imbal[FEATURES] = df_imbal[FEATURES].fillna(0)

print("\nChecking data types AFTER fix (Should all be float/int):")
print(df_imbal[FEATURES].dtypes)

# 1. Train
hmm_supervised = SupervisedMarkovChain()
hmm_supervised.fit(df_imbal, cluster_col='cluster', label_col='category', feature_cols=FEATURES)

# 2. View available composite states
print_available_composites(hmm_supervised)

# 3. Generate Standard Traffic (T=1.0) - minority classes only
df_std = generate_minority_traffic(
    model=hmm_supervised,
    df_original=df_imbal,
    generation_targets=GENERATION_TARGETS,
    seq_len=SEQ_LENGTH,
    temperature=1.0,
    output_file="synthetic_unsw_standard.csv"
)

# 4. Generate "Zero-Day" Traffic (T=1.3) - minority classes only
df_wild = generate_minority_traffic(
    model=hmm_supervised,
    df_original=df_imbal,
    generation_targets=GENERATION_TARGETS,
    seq_len=SEQ_LENGTH,
    temperature=1.3,
    output_file="synthetic_unsw_zeroday.csv"
)

# 5. Summary
if df_std is not None and df_wild is not None:
    combined = len(df_std) + len(df_wild)
    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Standard synthetic:  {len(df_std):,} rows")
    print(f"Zero-day synthetic:  {len(df_wild):,} rows")
    print(f"Combined synthetic:  {combined:,} rows")
    print(f"Original dataset:    2,540,047 rows")
    print(f"New total:           {2_540_047 + combined:,} rows")
    print(f"Synthetic proportion: {combined / (2_540_047 + combined):.1%}")

# 6. Inspect transitions for minority classes
for cat in GENERATION_TARGETS.keys():
    cat_states = get_composite_states_for_category(hmm_supervised, cat)
    if cat_states:
        inspect_transition_probabilities(hmm_supervised, cat_states[0])