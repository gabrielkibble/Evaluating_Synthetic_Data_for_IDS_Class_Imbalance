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

# CICIDS2017 raw CSV files (adjust paths as needed)
RAW_DATA_DIR = "./data/raw/"
FULL_SOURCE_FILES = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "*.csv")))

BALANCED_FILE = "balanced_cicids2017.csv"
OUTPUT_SEQ_FILE = "novel_attack_sequences_cicids.npy"

IMBALANCED_FILES = FULL_SOURCE_FILES


# ========================
# Toggles and Parameters
# ========================

RECREATE_BALANCED_DATA = True   # Set to False if balanced file already exists
RECREATE_SEQUENCES = True

# Parameters
TARGET_SAMPLES_PER_CLASS = 300
SEQ_LENGTH = 20
STRIDE = 10
TEMPERATURE = 1.2

# ========================
# Generation Targets
# ========================
# Current training set distribution:
#   Benign:      809,984  (74.5%) - NO generation needed
#   DoS/DDoS:   172,429  (15.9%) - NO generation needed
#   Web Attack:   95,282  ( 8.8%) - NO generation needed
#   Brute Force:   6,545  ( 0.6%) - TARGET: ~90,000
#   PortScan:      1,308  ( 0.1%) - TARGET: ~90,000
#   Botnet ARES:   1,174  ( 0.1%) - TARGET: ~90,000
#
# This brings minority classes to roughly the same order of magnitude
# as Web Attack (~95k), adding ~270k synthetic samples total (~25% of dataset)

GENERATION_TARGETS = {
    "Brute Force": 90_000,
    "Web Attack":  90_000,
    "Botnet ARES": 90_000,
}

REVERSE_LABEL_MAP = {
    "Brute Force": "Brute Force",
    "Web Attack": "Web Attack",
    "Botnet ARES": "Botnet ARES",
}

# CICIDS2017 equivalent features to UNSW-NB15's
FEATURES = [
    "flow_duration",
    "total_fwd_packets", "total_backward_packets",
    "total_length_of_fwd_packets", "total_length_of_bwd_packets",
    "fwd_packets_s", "bwd_packets_s",
    "fwd_packet_length_mean", "bwd_packet_length_mean",
    "fwd_iat_std", "bwd_iat_std",
    "fwd_iat_mean", "bwd_iat_mean"
]

# CICIDS2017 attack grouping (same as your preprocessing.py)
ATTACK_GROUP = {
    'BENIGN': 'Benign',
    'PortScan': 'PortScan',
    'DDoS': 'DoS/DDoS',
    'DoS Hulk': 'DoS/DDoS',
    'DoS GoldenEye': 'DoS/DDoS',
    'DoS slowloris': 'DoS/DDoS',
    'DoS Slowhttptest': 'DoS/DDoS',
    'Heartbleed': 'DoS/DDoS',
    'FTP-Patator': 'Brute Force',
    'SSH-Patator': 'Brute Force',
    'Bot': 'Botnet ARES',
    'Web Attack – Brute Force': 'Web Attack',
    'Web Attack – Sql Injection': 'Web Attack',
    'Web Attack – XSS': 'Web Attack',
    'Infiltration': 'Infiltration'
}


def clean_column_name(col):
    """Clean CICIDS2017 column names to match preprocessor format."""
    return col.strip().replace('/', '_').replace(' ', '_').lower()


# ============================
# Building Balanced Dataset
# ============================

def create_balanced_dataset(input_path_list, output_path, target_samples):
    """
    Scans CICIDS2017 CSV files and extracts a stratified sample of each category.
    """
    print("\n--- Creating Balanced Dataset from CICIDS2017 Files ---")
    
    data_buckets = {
        "Benign": [], "DoS/DDoS": [], "PortScan": [],
        "Brute Force": [], "Web Attack": [], "Botnet ARES": [],
        "Infiltration": []
    }
    
    chunk_size = 100000
    all_done = False

    for file_path in input_path_list:
        if all_done:
            break
            
        print(f"Scanning file: {os.path.basename(file_path)}...")

        try:
            for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False, encoding='cp1252'):
                # Clean column names
                chunk.columns = [clean_column_name(col) for col in chunk.columns]
                
                if 'label' not in chunk.columns:
                    print(f"Warning: 'label' column not found in {file_path}. Skipping.")
                    continue

                # Group attack labels
                chunk['category'] = chunk['label'].map(
                    lambda x: ATTACK_GROUP.get(str(x).strip(), 'Other')
                )

                # Fill buckets
                for cat in data_buckets.keys():
                    current_count = sum(len(df) for df in data_buckets[cat])
                    
                    if current_count < target_samples:
                        subset = chunk[chunk["category"] == cat]
                        if len(subset) > 0:
                            needed = target_samples - current_count
                            data_buckets[cat].append(subset.head(needed))

                # Progress
                total_counts = {k: sum(len(df) for df in v) for k, v in data_buckets.items()}
                print(f"Status: {total_counts}")

                if all(c >= target_samples for c in total_counts.values()):
                    print("All buckets full! Stopping scan.")
                    all_done = True
                    break
        
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

    # Combine and Save
    print("Combining extracted data...")
    frames = []
    for cat, bucket_list in data_buckets.items():
        if bucket_list:
            frames.append(pd.concat(bucket_list))
    
    if not frames:
        print("No data found in any files!")
        return

    balanced_df = pd.concat(frames)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print("\nFinal Balanced Distribution:")
    print(balanced_df["category"].value_counts())
    
    balanced_df.to_csv(output_path, index=False)
    print(f"Saved balanced dataset to '{output_path}'")


if RECREATE_BALANCED_DATA:
    create_balanced_dataset(FULL_SOURCE_FILES, BALANCED_FILE, TARGET_SAMPLES_PER_CLASS)
else:
    print(f"\nSkipping data creation. Using existing '{BALANCED_FILE}'...")


# ================================================
# STEP 1 - LEARNING CLUSTERS OFF BALANCED DATA
# ================================================

print("\n--- PHASE 1: Learning Vocabulary (Balanced Data) ---")

df_bal = pd.read_csv(BALANCED_FILE)

# Ensure feature columns are numeric
for col in FEATURES:
    if col in df_bal.columns:
        df_bal[col] = pd.to_numeric(df_bal[col], errors='coerce')

X_bal = df_bal[FEATURES].replace([np.inf, -np.inf], np.nan).dropna()

# Standardize and PCA
scaler = StandardScaler()
X_bal_scaled = scaler.fit_transform(X_bal)

pca = PCA(n_components=5, random_state=42)
X_bal_pca = pca.fit_transform(X_bal_scaled)

# Clustering using DBSCAN
dbscan = DBSCAN(eps=0.1, min_samples=5, n_jobs=-1)
bal_clusters = dbscan.fit_predict(X_bal_pca)

# Map DBSCAN clusters into KNN for prediction on new data
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_bal_pca, bal_clusters)

print(f"Vocabulary Learned: {len(np.unique(bal_clusters))} Clusters Found.")
print(pd.Series(bal_clusters).value_counts())

# Prepare stats dataframe
stats = X_bal.copy()
stats["cluster"] = bal_clusters
stats["category"] = df_bal.loc[X_bal.index, "category"]


def get_majority_label(cid, df_labeled):
    if cid == -1:
        return "Noise"
    
    cluster_data = df_labeled[df_labeled['cluster'] == cid]
    majority_category = cluster_data['category'].mode()[0]
    
    # Mapping CICIDS2017 categories to MITRE ATT&CK Tactics
    mapping = {
        "Benign":       "Benign",
        "DoS/DDoS":     "Impact (DoS/DDoS)",
        "PortScan":     "Recon (PortScan)",
        "Brute Force":  "Credential Access (Brute Force)",
        "Web Attack":   "Initial Access (Web Attack)",
        "Botnet ARES":  "C2 (Botnet)",
        "Infiltration": "Lateral Movement (Infiltration)"
    }
    
    return mapping.get(majority_category, "Unknown")


cluster_to_mitre = {}
unique_clusters = sorted(stats["cluster"].unique())

for cid in unique_clusters:
    label = get_majority_label(cid, stats)
    cluster_to_mitre[cid] = label
    print(f"Cluster {cid}: {label}")

# Label encoder for HMM compatibility
le = LabelEncoder()
le.fit(np.unique(bal_clusters))


# ==============================================
# STEP 2 - LEARN A MODEL OFF IMBALANCED DATA
# ==============================================

print("\n--- PHASE 2: Learning Grammar (Imbalanced Data) ---")

IMBALANCED_FILE_MAIN = []
for file_path in IMBALANCED_FILES:
    print(f"Reading {os.path.basename(file_path)}...")
    imbalanced_temp = pd.read_csv(file_path, low_memory=False, encoding='cp1252')
    # Clean column names
    imbalanced_temp.columns = [clean_column_name(col) for col in imbalanced_temp.columns]
    IMBALANCED_FILE_MAIN.append(imbalanced_temp)

df_imbal = pd.concat(IMBALANCED_FILE_MAIN, ignore_index=True)

# Group attack labels
df_imbal['category'] = df_imbal['label'].map(
    lambda x: ATTACK_GROUP.get(str(x).strip(), 'Other')
)

# Remove Infiltration (too few samples, same as your DBN pipeline)
df_imbal = df_imbal[df_imbal['category'] != 'Infiltration']

# Force features to numeric
for col in FEATURES:
    df_imbal[col] = pd.to_numeric(df_imbal[col], errors='coerce')

X_imbal = df_imbal[FEATURES].replace([np.inf, -np.inf], np.nan).dropna()

print(f"Loaded Imbalanced Data: {X_imbal.shape}")

# Project into Balanced Space
X_imbal_scaled = scaler.transform(X_imbal)
X_imbal_pca = pca.transform(X_imbal_scaled)

df_imbal = df_imbal.loc[X_imbal.index].copy()

# Assign cluster labels using KNN
imbal_clusters = knn.predict(X_imbal_pca)
df_imbal["cluster"] = imbal_clusters

print("Real-World Cluster Distribution (Should be skewed):")
print(pd.Series(imbal_clusters).value_counts())


# ================================
# STEP 2.5 - TRAINING THE HMM
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

        # Count Transitions
        trans_counts = np.ones((n_states, n_states)) * 1e-6  # Laplace smoothing
        
        print("Learning transition probabilities...")
        grouped = df.sort_values("timestamp").groupby("source_ip")
        
        for src_ip, group in tqdm(grouped, desc="Processing source IPs"):
            seq = group['composite_state'].values
            if len(seq) < 2:
                continue
            
            indices = [self.state_to_idx[s] for s in seq]
            for t in range(len(indices) - 1):
                trans_counts[indices[t], indices[t + 1]] += 1

        # Normalize to probabilities
        self.transmat_ = trans_counts / trans_counts.sum(axis=1, keepdims=True)
        
        # Feature Profiling
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
                
                self.state_feature_stats[state] = {
                    'mean': mean_vec,
                    'cov': cov_mx
                }
        
        print("Supervised Markov Chain Trained Successfully.")

    def apply_temperature(self, temperature=1.0):
        log_probs = np.log(self.transmat_ + 1e-9)
        scaled_logits = log_probs / temperature
        self.transmat_ = softmax(scaled_logits, axis=1)
        print(f"Transition matrix scaled with Temperature={temperature}")

    def generate_sequence(self, start_state, length=10):
        if start_state not in self.state_to_idx:
            raise ValueError(f"Start state '{start_state}' not found. Available: {self.states[:5]}...")
                
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
        parts = state.split('_', 1)  # Split on first underscore only
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
    """
    Returns all composite states belonging to a given category.
    E.g., category='Botnet ARES' -> ['13_Botnet ARES', '17_Botnet ARES', '18_Botnet ARES']
    """
    matching = []
    for state in model.states:
        label = state.split('_', 1)[1]
        if label == category:
            matching.append(state)
    return matching


def get_scaled_transition_matrix(trans_mat, T=1.0):
    if T == 1.0:
        return trans_mat.copy()
    scaled = np.power(trans_mat, 1.0 / T)
    row_sums = scaled.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return scaled / row_sums


def robust_sample(mean, cov):
    epsilon = 1e-6
    cov_reg = cov + np.eye(len(mean)) * epsilon
    cov_reg = (cov_reg + cov_reg.T) / 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            return np.random.multivariate_normal(mean, cov_reg)
        except ValueError:
            return mean


def generate_minority_traffic(model, df_original, generation_targets, seq_len=20,
                              temperature=1.0, output_file="synthetic_cicids2017.csv"):
    """
    Generates synthetic traffic focused on minority classes.
    OPTIMIZED: Pre-indexes template pools, batches numpy sampling, avoids per-row DataFrame ops.
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

    # CICIDS2017 column order (cleaned names)
    final_columns = [
        "flow_id", "source_ip", "source_port", "destination_ip", "destination_port",
        "protocol", "timestamp", "flow_duration", "total_fwd_packets",
        "total_backward_packets", "total_length_of_fwd_packets",
        "total_length_of_bwd_packets", "fwd_packet_length_max",
        "fwd_packet_length_min", "fwd_packet_length_mean", "fwd_packet_length_std",
        "bwd_packet_length_max", "bwd_packet_length_min", "bwd_packet_length_mean",
        "bwd_packet_length_std", "flow_bytes_s", "flow_packets_s",
        "flow_iat_mean", "flow_iat_std", "flow_iat_max", "flow_iat_min",
        "fwd_iat_total", "fwd_iat_mean", "fwd_iat_std", "fwd_iat_max",
        "fwd_iat_min", "bwd_iat_total", "bwd_iat_mean", "bwd_iat_std",
        "bwd_iat_max", "bwd_iat_min", "fwd_psh_flags", "bwd_psh_flags",
        "fwd_urg_flags", "bwd_urg_flags", "fwd_header_length", "bwd_header_length",
        "fwd_packets_s", "bwd_packets_s", "min_packet_length", "max_packet_length",
        "packet_length_mean", "packet_length_std", "packet_length_variance",
        "fin_flag_count", "syn_flag_count", "rst_flag_count", "psh_flag_count",
        "ack_flag_count", "urg_flag_count", "cwe_flag_count", "ece_flag_count",
        "down_up_ratio", "average_packet_size", "avg_fwd_segment_size",
        "avg_bwd_segment_size", "fwd_avg_bytes_bulk", "fwd_avg_packets_bulk",
        "fwd_avg_bulk_rate", "bwd_avg_bytes_bulk", "bwd_avg_packets_bulk",
        "bwd_avg_bulk_rate", "subflow_fwd_packets", "subflow_fwd_bytes",
        "subflow_bwd_packets", "subflow_bwd_bytes", "init_win_bytes_forward",
        "init_win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward",
        "active_mean", "active_std", "active_max", "active_min",
        "idle_mean", "idle_std", "idle_max", "idle_min",
        "label"
    ]

    # =============================================
    # OPTIMIZATION 1: Pre-index all template pools
    # =============================================
    print("Pre-indexing template pools (one-time cost)...")
    # Convert each pool to a numpy array of dicts for fast random access
    pool_cache = {}
    for state in tqdm(model.states, desc="Caching pools"):
        pool_df = df_original[df_original['composite_state'] == state]
        if not pool_df.empty:
            pool_cache[state] = pool_df.values  # numpy array
            pool_cache[state + '_cols'] = pool_df.columns.tolist()
            pool_cache[state + '_len'] = len(pool_df)

    # =============================================
    # OPTIMIZATION 2: Pre-compute Cholesky decompositions
    # =============================================
    print("Pre-computing Cholesky decompositions...")
    cholesky_cache = {}
    for state, stats in model.state_feature_stats.items():
        cov = stats['cov'] + np.eye(len(stats['mean'])) * 1e-6
        cov = (cov + cov.T) / 2
        try:
            cholesky_cache[state] = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            cholesky_cache[state] = None  # fallback to mean

    all_rows = []
    
    # Temperature scaling
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
                # Pre-generate all Markov chain sequences at once
                all_sequences = []
                for _ in range(n_sequences):
                    try:
                        seq = model.generate_sequence(start_state=start_node, length=seq_len)
                        all_sequences.append(seq)
                    except:
                        continue

                total_rows_this_state = len(all_sequences) * seq_len

                # Pre-sample all template row indices at once
                template_indices = {}
                for state in model.states:
                    if state in pool_cache:
                        # Count how many times this state appears across all sequences
                        count = sum(seq.count(state) for seq in all_sequences)
                        if count > 0:
                            template_indices[state] = np.random.randint(
                                0, pool_cache[state + '_len'], size=count
                            )

                # Track which index we're at for each state
                idx_counters = {state: 0 for state in template_indices}

                cat_pbar = tqdm(
                    all_sequences,
                    desc=f"    {start_node}",
                    unit="seq",
                    position=1,
                    leave=False
                )

                base_time = pd.Timestamp.now()

                for seq_i, seq_states in enumerate(cat_pbar):
                    current_time = base_time + pd.Timedelta(hours=seq_i)

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
                            # Fallback to start node pool
                            row_idx = np.random.randint(0, start_pool_len)
                            row_array = start_pool[row_idx]
                            cols = start_cols

                        template_row = dict(zip(cols, row_array))

                        # =============================================
                        # OPTIMIZATION 4: Fast feature sampling
                        # =============================================
                        L = cholesky_cache.get(state)
                        if L is not None:
                            z = np.random.standard_normal(len(feat_stats['mean']))
                            feats = feat_stats['mean'] + L @ z
                            feats = np.maximum(feats, 0)
                        else:
                            feats = np.maximum(feat_stats['mean'].copy(), 0)

                        gen_flow_duration = feats[0]
                        gen_total_fwd_packets = max(1, round(feats[1]))
                        gen_total_bwd_packets = max(0, round(feats[2]))
                        gen_total_len_fwd = max(0, feats[3])
                        gen_total_len_bwd = max(0, feats[4])

                        template_row['flow_duration'] = gen_flow_duration
                        template_row['total_fwd_packets'] = gen_total_fwd_packets
                        template_row['total_backward_packets'] = gen_total_bwd_packets
                        template_row['total_length_of_fwd_packets'] = gen_total_len_fwd
                        template_row['total_length_of_bwd_packets'] = gen_total_len_bwd
                        template_row['fwd_packets_s'] = feats[5]
                        template_row['bwd_packets_s'] = feats[6]
                        template_row['fwd_packet_length_mean'] = feats[7]
                        template_row['bwd_packet_length_mean'] = feats[8]
                        template_row['fwd_iat_std'] = feats[9]
                        template_row['bwd_iat_std'] = feats[10]
                        template_row['fwd_iat_mean'] = feats[11]
                        template_row['bwd_iat_mean'] = feats[12]

                        # Derived fields
                        if gen_total_fwd_packets > 0:
                            template_row['avg_fwd_segment_size'] = gen_total_len_fwd / gen_total_fwd_packets
                        if gen_total_bwd_packets > 0:
                            template_row['avg_bwd_segment_size'] = gen_total_len_bwd / gen_total_bwd_packets

                        total_packets = gen_total_fwd_packets + gen_total_bwd_packets
                        total_bytes = gen_total_len_fwd + gen_total_len_bwd
                        if gen_flow_duration > 0:
                            template_row['flow_bytes_s'] = total_bytes / (gen_flow_duration / 1e6)
                            template_row['flow_packets_s'] = total_packets / (gen_flow_duration / 1e6)

                        template_row['average_packet_size'] = (
                            total_bytes / total_packets if total_packets > 0 else 0
                        )

                        template_row['timestamp'] = (
                            current_time + pd.Timedelta(seconds=t * 0.5)
                        ).strftime('%d/%m/%Y %H:%M')

                        state_label = state.split('_', 1)[1]
                        template_row['label'] = REVERSE_LABEL_MAP.get(state_label, state_label)

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

        print(f"Saving to {output_file}...")
        df_gen.to_csv(output_file, index=False)
        
        print(f"\n{'='*60}")
        print(f"GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"Total rows generated: {len(df_gen):,}")
        print(f"Saved to: {output_file}")
        print(f"\nBreakdown by category:")
        print(df_gen['label'].value_counts().to_string())
        print(f"\nThis will increase training set from ~1.09M to ~{1_086_722 + len(df_gen):,} rows")
        print(f"Synthetic data proportion: {len(df_gen) / (1_086_722 + len(df_gen)):.1%}")
        
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
    output_file="synthetic_cicids_standard.csv"
)

# 4. Generate "Zero-Day" Traffic (T=1.3) - minority classes only
df_wild = generate_minority_traffic(
    model=hmm_supervised,
    df_original=df_imbal,
    generation_targets=GENERATION_TARGETS,
    seq_len=SEQ_LENGTH,
    temperature=1.3,
    output_file="synthetic_cicids_zeroday.csv"
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
    print(f"Original training:   1,086,722 rows")
    print(f"New total:           {1_086_722 + combined:,} rows")
    print(f"Synthetic proportion: {combined / (1_086_722 + combined):.1%}")

# 6. Inspect transitions for minority classes
for cat in GENERATION_TARGETS.keys():
    cat_states = get_composite_states_for_category(hmm_supervised, cat)
    if cat_states:
        inspect_transition_probabilities(hmm_supervised, cat_states[0])

# 7. Temperature sweep for sensitivity analysis
for T in [1.4, 1.6, 1.8, 2.0, 2.5, 3.0]:
    print(f"\n{'#'*60}")
    print(f"GENERATING AT T={T}")  
    print(f"{'#'*60}")
    
    generate_minority_traffic(
        model=hmm_supervised,
        df_original=df_imbal,
        generation_targets=GENERATION_TARGETS,
        seq_len=SEQ_LENGTH,
        temperature=T,
        output_file=f"synthetic_cicids_T{T}.csv"
    )