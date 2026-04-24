import pandas as pd
import numpy as np
import glob
import os
import sys
import pickle

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

# Use a more robust way to find the data directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_PATH = os.path.abspath(__file__)
PREPROCESSING_DIR = os.path.dirname(SCRIPT_PATH)
ROOT_DIR = os.path.dirname(PREPROCESSING_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")

print(f"DEBUG: Data Directory is set to: {DATA_DIR}")

class CICIDS2017Preprocessor(object):
    def __init__(self, data_path, training_size, validation_size, testing_size):
        self.data_path = data_path
        self.training_size = training_size
        self.validation_size = validation_size
        self.testing_size = testing_size
        
        self.data = None
        self.features = None
        self.labels = None
        self.preprocessor = None  # Store fitted ColumnTransformer
        self.label_encoder = None  # Store fitted LabelEncoder
        self.feature_columns = None  # Store selected feature columns

    def read_data(self):
        search_path = os.path.join(self.data_path, 'raw', '*.csv')
        filenames = glob.glob(search_path)
        
        # Exclude synthetic files from raw data
        filenames = [f for f in filenames if 'synthetic' not in os.path.basename(f).lower()]
        
        print(f"[1/8] Found {len(filenames)} files in {search_path}")
        
        if not filenames:
            print("!!! ERROR: No CSV files found. Check your data/raw folder.")
            sys.exit(1)

        datasets = []
        for filename in filenames:
            print(f"      Reading {os.path.basename(filename)}...")
            df = pd.read_csv(filename, encoding='cp1252')
            df.columns = [self._clean_column_name(col) for col in df.columns]
            datasets.append(df)

        print("      Concatenating datasets...")
        self.data = pd.concat(datasets, axis=0, ignore_index=True)
        
        if 'fwd_header_length.1' in self.data.columns:
            self.data.drop(labels=['fwd_header_length.1'], axis=1, inplace=True)

    def _clean_column_name(self, column):
        return column.strip().replace('/', '_').replace(' ', '_').lower()

    def remove_duplicate_values(self):
        print("[2/8] Removing duplicates...")
        self.data.drop_duplicates(inplace=True, keep=False, ignore_index=True)

    def remove_missing_values(self):
        print("[3/8] Removing missing values...")
        self.data.dropna(axis=0, inplace=True, how="any")

    def remove_infinite_values(self):
        print("[4/8] Removing infinite values...")
        self.data.replace([-np.inf, np.inf], np.nan, inplace=True)
        self.data.dropna(axis=0, how='any', inplace=True)

    def remove_constant_features(self, threshold=0.01):
        print("[5/8] Removing constant features...")
        data_std = self.data.std(numeric_only=True)
        constant_features = [column for column, std in data_std.items() if std < threshold]
        self.data.drop(labels=constant_features, axis=1, inplace=True)

    def remove_correlated_features(self, threshold=0.98):
        print("[6/8] Removing highly correlated features (this may take a minute)...")
        numeric_data = self.data.select_dtypes(include=[np.number])
        data_corr = numeric_data.corr()
        mask = np.triu(np.ones_like(data_corr, dtype=bool))
        tri_df = data_corr.mask(mask)
        correlated_features = [c for c in tri_df.columns if any(tri_df[c] > threshold)]
        self.data.drop(labels=correlated_features, axis=1, inplace=True)

    def group_labels(self):
        print("[7/8] Grouping attack labels...")
        attack_group = {
            'BENIGN': 'Benign', 'PortScan': 'PortScan', 'DDoS': 'DoS/DDoS',
            'DoS Hulk': 'DoS/DDoS', 'DoS GoldenEye': 'DoS/DDoS',
            'DoS slowloris': 'DoS/DDoS', 'DoS Slowhttptest': 'DoS/DDoS',
            'Heartbleed': 'DoS/DDoS', 'FTP-Patator': 'Brute Force',
            'SSH-Patator': 'Brute Force', 'Bot': 'Botnet ARES',
            'Web Attack \x96 Brute Force': 'Web Attack',
            'Web Attack \x96 Sql Injection': 'Web Attack',
            'Web Attack \x96 XSS': 'Web Attack',
            'Web Attack – Brute Force': 'Web Attack',
            'Web Attack – Sql Injection': 'Web Attack',
            'Web Attack – XSS': 'Web Attack',
            'Web Attack  Brute Force': 'Web Attack',
            'Web Attack  Sql Injection': 'Web Attack',
            'Web Attack  XSS': 'Web Attack',
            'Infiltration': 'Infiltration'
        }
        self.data['label_category'] = self.data['label'].map(lambda x: attack_group.get(x.strip(), 'Other'))
        
    def train_valid_test_split(self):
        print("[8/8] Splitting data into Train/Val/Test...")
        self.labels = self.data['label_category']
        
        cols_to_drop = ['label', 'label_category', 'flow_id', 'source_ip', 'destination_ip', 'timestamp', 'external_ip']
        existing_drops = [c for c in cols_to_drop if c in self.data.columns]
        self.features = self.data.drop(labels=existing_drops, axis=1)
        
        # Store the feature columns for later use with synthetic data
        self.feature_columns = self.features.columns.tolist()

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
        
        categorical_features = self.features.select_dtypes(exclude=["number"]).columns
        numeric_features = self.features.select_dtypes(include=["number"]).columns

        preprocessor = ColumnTransformer(transformers=[
            ('categoricals', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features),
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

        # Store fitted transformers for synthetic data injection
        self.preprocessor = preprocessor
        self.label_encoder = le
        self.scaled_columns = new_cols

        return (X_train_final, y_train_final), (X_val_final, y_val_final), (X_test_final, y_test_final)


def inject_synthetic_data(proc, X_train, y_train, synthetic_path, undersample_benign=True):
    """
    Loads synthetic data, processes it through the same pipeline as real data,
    and appends it to the training set only.
    """
    if not os.path.exists(synthetic_path):
        print(f"No synthetic file found at {synthetic_path}. Skipping injection.")
        return X_train, y_train
    
    print(f"\n--- Injecting Synthetic Data (training set only) ---")
    print(f"Reading {synthetic_path}...")
    df_synth = pd.read_csv(synthetic_path, encoding='cp1252')
    
    # Clean column names the same way
    df_synth.columns = [col.strip().replace('/', '_').replace(' ', '_').lower() for col in df_synth.columns]
    
    print(f"Loaded {len(df_synth):,} synthetic rows")
    
    # Apply same label grouping
    attack_group = {
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
    
    if 'label' not in df_synth.columns:
        print("WARNING: No 'label' column in synthetic data. Skipping.")
        return X_train, y_train
    
    GROUPED_LABELS = {'Benign', 'Botnet ARES', 'Brute Force', 'DoS/DDoS', 'PortScan', 'Web Attack'}

    y_synth_raw = df_synth['label'].map(
        lambda x: str(x).strip() if str(x).strip() in GROUPED_LABELS 
        else attack_group.get(str(x).strip(), 'Other')
    )
    
    # Drop any 'Other' or categories not in the label encoder
    known_classes = set(proc.label_encoder.classes_)
    valid_mask = y_synth_raw.isin(known_classes)
    df_synth = df_synth[valid_mask].copy()
    y_synth_raw = y_synth_raw[valid_mask]
    
    print(f"After label filtering: {len(df_synth):,} rows")
    print(f"Synthetic label distribution:")
    print(y_synth_raw.value_counts().to_string())
    
    # Select the same feature columns used in real data
    cols_to_drop = ['label', 'label_category', 'flow_id', 'source_ip', 'destination_ip', 'timestamp', 'external_ip']
    existing_drops = [c for c in cols_to_drop if c in df_synth.columns]
    X_synth_raw = df_synth.drop(labels=existing_drops, axis=1)
    
    # Keep only columns that exist in the real feature set
    available_cols = [c for c in proc.feature_columns if c in X_synth_raw.columns]
    missing_cols = [c for c in proc.feature_columns if c not in X_synth_raw.columns]
    
    # Add missing columns as 0
    for col in missing_cols:
        X_synth_raw[col] = 0
    
    X_synth_raw = X_synth_raw[proc.feature_columns]
    
    # Force numeric
    for col in X_synth_raw.columns:
        X_synth_raw[col] = pd.to_numeric(X_synth_raw[col], errors='coerce')
    X_synth_raw = X_synth_raw.fillna(0)
    X_synth_raw = X_synth_raw.replace([np.inf, -np.inf], 0)
    
    # Align indices
    y_synth_raw = y_synth_raw.loc[X_synth_raw.index]
    
    # Apply the SAME fitted preprocessor (QuantileTransformer + OneHotEncoder)
    print("Applying fitted preprocessor to synthetic data...")
    try:
        X_synth_proc = proc.preprocessor.transform(X_synth_raw)
    except Exception as e:
        print(f"WARNING: Transform failed: {e}")
        print("Attempting with error handling...")
        # Some categorical values might not have been seen during fit
        X_synth_proc = proc.preprocessor.transform(X_synth_raw)
    
    X_synth_final = pd.DataFrame(X_synth_proc, columns=proc.scaled_columns)
    
    # Apply the SAME fitted LabelEncoder
    y_synth_final = pd.DataFrame(proc.label_encoder.transform(y_synth_raw), columns=["label"])
    
    print(f"Synthetic data processed: {X_synth_final.shape}")
    
    # Undersample Benign to keep total dataset size constant
    if undersample_benign:
        n_synthetic = len(X_synth_final)
        benign_label = proc.label_encoder.transform(['Benign'])[0]
        benign_mask = y_train['label'] == benign_label
        n_benign = benign_mask.sum()
        
        if n_synthetic <= n_benign:
            drop_indices = y_train[benign_mask].sample(n=n_synthetic, random_state=42).index
            X_train = X_train.drop(drop_indices).reset_index(drop=True)
            y_train = y_train.drop(drop_indices).reset_index(drop=True)
            print(f"Dropped {n_synthetic:,} Benign samples to keep total constant")
        else:
            print(f"WARNING: More synthetic ({n_synthetic:,}) than Benign ({n_benign:,}). Dropping all Benign.")
            X_train = X_train[~benign_mask].reset_index(drop=True)
            y_train = y_train[~benign_mask].reset_index(drop=True)
    
    # Concatenate synthetic into training set
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
proc = CICIDS2017Preprocessor(DATA_DIR, 0.6, 0.2, 0.2)
proc.read_data()
proc.remove_duplicate_values()
proc.remove_missing_values()
proc.remove_infinite_values()
proc.remove_constant_features()
proc.remove_correlated_features()
proc.group_labels()

# Remove Infiltration
proc.data = proc.data[proc.data['label_category'] != 'Infiltration']

sets = proc.train_valid_test_split()
(XT, YT), (XV, YV), (XTe, YTe) = proc.scale(*sets)

# Inject synthetic data into training set ONLY
SYNTHETIC_FILE = os.path.join(DATA_DIR, "synthetic_cicids_standard.csv")
XT, YT = inject_synthetic_data(proc, XT, YT, SYNTHETIC_FILE, undersample_benign=True)

print("\nSaving...")
for folder in ['train', 'val', 'test']:
    os.makedirs(os.path.join(DATA_DIR, 'processed', folder), exist_ok=True)

XT.to_pickle(os.path.join(DATA_DIR, 'processed/train/train_features.pkl'))
XV.to_pickle(os.path.join(DATA_DIR, 'processed/val/val_features.pkl'))
XTe.to_pickle(os.path.join(DATA_DIR, 'processed/test/test_features.pkl'))
YT.to_pickle(os.path.join(DATA_DIR, 'processed/train/train_labels.pkl'))
YV.to_pickle(os.path.join(DATA_DIR, 'processed/val/val_labels.pkl'))
YTe.to_pickle(os.path.join(DATA_DIR, 'processed/test/test_labels.pkl'))
print("--- FINISHED ---")