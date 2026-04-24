import pandas as pd
import numpy as np
import glob
import os
import sys

from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

SCRIPT_PATH = os.path.abspath(__file__)
PREPROCESSING_DIR = os.path.dirname(SCRIPT_PATH)
ROOT_DIR = os.path.dirname(PREPROCESSING_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")

print(f"DEBUG: Data Directory is set to: {DATA_DIR}")

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

# Columns to drop before training (identity/metadata columns)
COLS_TO_DROP = [
    "srcip", "dstip", "sport", "dsport", "proto", "state", "service",
    "stime", "ltime", "attack_cat", "label_category", "Label"
]


class UNSWNB15Preprocessor(object):
    def __init__(self, data_path, training_size, validation_size, testing_size):
        self.data_path = data_path
        self.training_size = training_size
        self.validation_size = validation_size
        self.testing_size = testing_size

        self.data = None
        self.features = None
        self.labels = None
        self.preprocessor = None
        self.label_encoder = None
        self.feature_columns = None

    def read_data(self):
        search_path = os.path.join(self.data_path, 'raw_unsw', '*.csv')
        filenames = glob.glob(search_path)

        # Exclude synthetic files
        filenames = [f for f in filenames if 'synthetic' not in os.path.basename(f).lower()]

        print(f"[1/8] Found {len(filenames)} files in {search_path}")

        if not filenames:
            print("!!! ERROR: No CSV files found. Check your data/raw_unsw folder.")
            sys.exit(1)

        datasets = []
        for filename in sorted(filenames):
            print(f"      Reading {os.path.basename(filename)}...")
            df = pd.read_csv(filename, names=RAW_COLUMN_NAMES, header=None, low_memory=False)
            datasets.append(df)

        print("      Concatenating datasets...")
        self.data = pd.concat(datasets, axis=0, ignore_index=True)

    def clean_labels(self):
        print("[2/8] Cleaning labels...")
        # Fill NaN attack_cat for normal traffic
        self.data['attack_cat'] = self.data.apply(
            lambda row: 'Normal' if pd.isna(row['attack_cat']) and row['Label'] == 0
            else str(row['attack_cat']).strip(), axis=1
        )
        # Merge Backdoor -> Backdoors
        self.data['attack_cat'] = self.data['attack_cat'].replace('Backdoor', 'Backdoors')
        # Drop Worms (too few samples)
        self.data = self.data[self.data['attack_cat'] != 'Worms']
        # Create label_category column (same name as CICIDS for consistency)
        self.data['label_category'] = self.data['attack_cat']

        print(f"      Label distribution:")
        print(self.data['label_category'].value_counts().to_string())

    def remove_duplicate_values(self):
        print("[3/8] Removing duplicates...")
        self.data.drop_duplicates(inplace=True, keep=False, ignore_index=True)

    def remove_missing_values(self):
        print("[4/8] Removing missing values...")
        self.data.dropna(axis=0, inplace=True, how="any")

    def remove_infinite_values(self):
        print("[5/8] Removing infinite values...")
        self.data.replace([-np.inf, np.inf], np.nan, inplace=True)
        self.data.dropna(axis=0, how='any', inplace=True)

    def remove_constant_features(self, threshold=0.01):
        print("[6/8] Removing constant features...")
        data_std = self.data.std(numeric_only=True)
        constant_features = [column for column, std in data_std.items() if std < threshold]
        self.data.drop(labels=constant_features, axis=1, inplace=True)
        print(f"      Removed {len(constant_features)} constant features")

    def remove_correlated_features(self, threshold=0.98):
        print("[7/8] Removing highly correlated features...")
        numeric_data = self.data.select_dtypes(include=[np.number])
        data_corr = numeric_data.corr()
        mask = np.triu(np.ones_like(data_corr, dtype=bool))
        tri_df = data_corr.mask(mask)
        correlated_features = [c for c in tri_df.columns if any(tri_df[c] > threshold)]
        self.data.drop(labels=correlated_features, axis=1, inplace=True)
        print(f"      Removed {len(correlated_features)} correlated features")

    def train_valid_test_split(self):
        print("[8/8] Splitting data into Train/Val/Test...")
        self.labels = self.data['label_category']

        # Drop identity/metadata columns
        existing_drops = [c for c in COLS_TO_DROP if c in self.data.columns]
        self.features = self.data.drop(labels=existing_drops, axis=1)

        # Force all remaining columns to numeric
        for col in self.features.columns:
            self.features[col] = pd.to_numeric(self.features[col], errors='coerce')
        self.features = self.features.fillna(0)

        self.feature_columns = self.features.columns.tolist()
        print(f"      Features: {len(self.feature_columns)} columns")

        X_train, X_temp, y_train, y_temp = train_test_split(
            self.features, self.labels,
            test_size=(self.validation_size + self.testing_size),
            random_state=42, stratify=self.labels
        )

        val_ratio = self.validation_size / (self.validation_size + self.testing_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1 - val_ratio), random_state=42
        )

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def scale(self, training_set, validation_set, testing_set):
        print("      Scaling and Encoding features...")
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = training_set, validation_set, testing_set

        numeric_features = self.features.select_dtypes(include=["number"]).columns

        preprocessor = ColumnTransformer(transformers=[
            ('numericals', QuantileTransformer(), numeric_features)
        ])

        X_train_proc = preprocessor.fit_transform(X_train)
        X_val_proc = preprocessor.transform(X_val)
        X_test_proc = preprocessor.transform(X_test)

        try:
            new_cols = preprocessor.get_feature_names_out()
        except:
            new_cols = [f"f_{i}" for i in range(X_train_proc.shape[1])]

        X_train_final = pd.DataFrame(X_train_proc, columns=new_cols)
        X_val_final = pd.DataFrame(X_val_proc, columns=new_cols)
        X_test_final = pd.DataFrame(X_test_proc, columns=new_cols)

        le = LabelEncoder()
        y_train_final = pd.DataFrame(le.fit_transform(y_train), columns=["label"])
        y_val_final = pd.DataFrame(le.transform(y_val), columns=["label"])
        y_test_final = pd.DataFrame(le.transform(y_test), columns=["label"])

        # Store fitted transformers
        self.preprocessor = preprocessor
        self.label_encoder = le
        self.scaled_columns = new_cols

        print(f"      Label classes: {list(le.classes_)}")

        return (X_train_final, y_train_final), (X_val_final, y_val_final), (X_test_final, y_test_final)


def inject_synthetic_data(proc, X_train, y_train, synthetic_path, undersample_normal=True):
    """
    Loads synthetic UNSW-NB15 data, processes it through the same pipeline,
    and appends it to the training set only.
    """
    if not os.path.exists(synthetic_path):
        print(f"No synthetic file found at {synthetic_path}. Skipping injection.")
        return X_train, y_train

    print(f"\n--- Injecting Synthetic Data (training set only) ---")
    print(f"Reading {synthetic_path}...")
    df_synth = pd.read_csv(synthetic_path, low_memory=False)

    print(f"Loaded {len(df_synth):,} synthetic rows")

    # Get category from the 'category' column (UNSW synthetic uses 'category')
    cat_col = 'category' if 'category' in df_synth.columns else 'attack_cat'
    if cat_col not in df_synth.columns:
        print("WARNING: No category column found in synthetic data. Skipping.")
        return X_train, y_train

    # Already-grouped labels pass through; raw labels get mapped
    GROUPED_LABELS = set(proc.label_encoder.classes_)

    y_synth_raw = df_synth[cat_col].map(
        lambda x: str(x).strip() if str(x).strip() in GROUPED_LABELS else 'Other'
    )

    # Drop unknowns
    valid_mask = y_synth_raw.isin(GROUPED_LABELS)
    df_synth = df_synth[valid_mask].copy()
    y_synth_raw = y_synth_raw[valid_mask]

    print(f"After label filtering: {len(df_synth):,} rows")
    print(f"Synthetic label distribution:")
    print(y_synth_raw.value_counts().to_string())

    # Drop identity columns and select features
    existing_drops = [c for c in COLS_TO_DROP + [cat_col, 'composite_state', 'cluster'] if c in df_synth.columns]
    X_synth_raw = df_synth.drop(labels=existing_drops, axis=1, errors='ignore')

    # Keep only columns that exist in real feature set
    available_cols = [c for c in proc.feature_columns if c in X_synth_raw.columns]
    missing_cols = [c for c in proc.feature_columns if c not in X_synth_raw.columns]

    for col in missing_cols:
        X_synth_raw[col] = 0

    X_synth_raw = X_synth_raw[proc.feature_columns]

    # Force numeric
    for col in X_synth_raw.columns:
        X_synth_raw[col] = pd.to_numeric(X_synth_raw[col], errors='coerce')
    X_synth_raw = X_synth_raw.fillna(0).replace([np.inf, -np.inf], 0)

    # Align indices
    y_synth_raw = y_synth_raw.loc[X_synth_raw.index]

    # Apply fitted preprocessor
    print("Applying fitted preprocessor to synthetic data...")
    X_synth_proc = proc.preprocessor.transform(X_synth_raw)
    X_synth_final = pd.DataFrame(X_synth_proc, columns=proc.scaled_columns)

    # Apply fitted LabelEncoder
    y_synth_final = pd.DataFrame(proc.label_encoder.transform(y_synth_raw), columns=["label"])

    print(f"Synthetic data processed: {X_synth_final.shape}")

    # Undersample Normal to keep total constant
    if undersample_normal:
        n_synthetic = len(X_synth_final)
        normal_label = proc.label_encoder.transform(['Normal'])[0]
        normal_mask = y_train['label'] == normal_label
        n_normal = normal_mask.sum()

        if n_synthetic <= n_normal:
            drop_indices = y_train[normal_mask].sample(n=n_synthetic, random_state=42).index
            X_train = X_train.drop(drop_indices).reset_index(drop=True)
            y_train = y_train.drop(drop_indices).reset_index(drop=True)
            print(f"Dropped {n_synthetic:,} Normal samples to keep total constant")
        else:
            print(f"WARNING: More synthetic ({n_synthetic:,}) than Normal ({n_normal:,}). Dropping all Normal.")
            X_train = X_train[~normal_mask].reset_index(drop=True)
            y_train = y_train[~normal_mask].reset_index(drop=True)

    # Concatenate
    X_train = pd.concat([X_train, X_synth_final], ignore_index=True)
    y_train = pd.concat([y_train, y_synth_final], ignore_index=True)

    # Shuffle
    shuffle_idx = np.random.RandomState(42).permutation(len(X_train))
    X_train = X_train.iloc[shuffle_idx].reset_index(drop=True)
    y_train = y_train.iloc[shuffle_idx].reset_index(drop=True)

    print(f"\nFinal training set: {X_train.shape}")
    print(f"Training label distribution:")
    for cls_name in proc.label_encoder.classes_:
        cls_id = proc.label_encoder.transform([cls_name])[0]
        count = (y_train['label'] == cls_id).sum()
        print(f"  {cls_name}: {count:,}")

    return X_train, y_train


# ==========================================
# PIPELINE EXECUTION
# ==========================================

print("--- PIPELINE STARTING ---")
proc = UNSWNB15Preprocessor(DATA_DIR, 0.6, 0.2, 0.2)
proc.read_data()
proc.clean_labels()
proc.remove_duplicate_values()
proc.remove_missing_values()
proc.remove_infinite_values()
proc.remove_constant_features()
proc.remove_correlated_features()

sets = proc.train_valid_test_split()
(XT, YT), (XV, YV), (XTe, YTe) = proc.scale(*sets)

# Inject synthetic data into training set ONLY (if it exists)
SYNTHETIC_FILE = os.path.join(DATA_DIR, "synthetic_unsw_standard.csv")
XT, YT = inject_synthetic_data(proc, XT, YT, SYNTHETIC_FILE, undersample_normal=True)

print("\nSaving...")
for folder in ['train', 'val', 'test']:
    os.makedirs(os.path.join(DATA_DIR, 'processed_unsw', folder), exist_ok=True)

XT.to_pickle(os.path.join(DATA_DIR, 'processed_unsw/train/train_features.pkl'))
XV.to_pickle(os.path.join(DATA_DIR, 'processed_unsw/val/val_features.pkl'))
XTe.to_pickle(os.path.join(DATA_DIR, 'processed_unsw/test/test_features.pkl'))
YT.to_pickle(os.path.join(DATA_DIR, 'processed_unsw/train/train_labels.pkl'))
YV.to_pickle(os.path.join(DATA_DIR, 'processed_unsw/val/val_labels.pkl'))
YTe.to_pickle(os.path.join(DATA_DIR, 'processed_unsw/test/test_labels.pkl'))
print("--- FINISHED ---")