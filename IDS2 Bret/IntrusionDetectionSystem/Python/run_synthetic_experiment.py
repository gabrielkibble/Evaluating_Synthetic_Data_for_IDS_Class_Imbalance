"""
Experiment: Inject synthetic attack traffic into the autoencoder's test set
to evaluate whether richer attack diversity improves threshold calibration.

Usage:
    python run_synthetic_experiment.py
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import IDS
import DataEncoding as DataE
import Hyper as HyperP

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "trained.h5"
PICKLES_PATH = "Pickles/New/"
TEST_PICKLE = PICKLES_PATH + "combined_test_data.pkl"
SYNTHETIC_CSV = "/mnt/vurm/homes/homes/gk634/Dissertation/DBN/dbn-based-nids/synthetic_cicids_standard.csv"
ENCODING = DataE.AUTOENCODER_PREPROCESS
THRESHOLD_INCREMENTS = 50

# Columns that the test pickle uses (minus Label, Timestamp, Full Label)
FEATURE_COLS_NORM = DataE.AUTOENCODER_PREPROCESS.COL_TO_NORM
PROTOCOL_ONEHOT_COLS = ['Protocol_0', 'Protocol_6', 'Protocol_17',
                        'Protocol_0.0', 'Protocol_6.0', 'Protocol_17.0']
IP_COLS = ['Internal Source IP', 'Internal Dest IP',
           'Public Facing Source IP', 'Public Facing Dest IP',
           'External Source IP', 'External Dest IP']
SCORING_COLS = ['Label', 'Timestamp', 'Full Label']


def clean_column_name(col):
    """Clean column names to match CICIDS2017 format (strip leading spaces)."""
    return col.strip()


def process_synthetic_for_autoencoder(synthetic_path, reference_df):
    """
    Reads synthetic CSV and processes it to match the autoencoder test pickle format.
    
    Parameters
    ----------
    synthetic_path : str
        Path to synthetic CSV file
    reference_df : pd.DataFrame  
        The existing test pickle, used to get scaler fit and column order
        
    Returns
    -------
    pd.DataFrame matching the test pickle format
    """
    print(f"\n--- Processing Synthetic Data for Autoencoder ---")
    print(f"Reading {synthetic_path}...")
    
    df_synth = pd.read_csv(synthetic_path, encoding='cp1252', low_memory=False)
    df_synth.columns = [clean_column_name(c) for c in df_synth.columns]
    # Map cleaned synthetic column names to original CICIDS2017 names
    SYNTH_TO_ORIGINAL = {
        "destination_port": "Destination Port",
        "flow_duration": "Flow Duration",
        "total_fwd_packets": "Total Fwd Packets",
        "total_backward_packets": "Total Backward Packets",
        "total_length_of_fwd_packets": "Total Length of Fwd Packets",
        "total_length_of_bwd_packets": "Total Length of Bwd Packets",
        "fwd_packet_length_mean": "Fwd Packet Length Mean",
        "fwd_packet_length_std": "Fwd Packet Length Std",
        "fwd_iat_mean": "Fwd IAT Mean",
        "fwd_iat_std": "Fwd IAT Std",
        "flow_iat_max": "Flow IAT Max",
        "flow_iat_min": "Flow IAT Min",
        "bwd_iat_mean": "Bwd IAT Mean",
        "bwd_iat_std": "Bwd IAT Std",
        "packet_length_mean": "Packet Length Mean",
        "packet_length_std": "Packet Length Std",
        "fin_flag_count": "FIN Flag Count",
        "syn_flag_count": "SYN Flag Count",
        "rst_flag_count": "RST Flag Count",
        "psh_flag_count": "PSH Flag Count",
        "ack_flag_count": "ACK Flag Count",
        "urg_flag_count": "URG Flag Count",
        "cwe_flag_count": "CWE Flag Count",
        "ece_flag_count": "ECE Flag Count",
        "average_packet_size": "Average Packet Size",
        "avg_fwd_segment_size": "Avg Fwd Segment Size",
        "subflow_fwd_packets": "Subflow Fwd Packets",
        "subflow_fwd_bytes": "Subflow Fwd Bytes",
        "init_win_bytes_forward": "Init_Win_bytes_forward",
        "min_seg_size_forward": "min_seg_size_forward",
        "active_mean": "Active Mean",
        "active_std": "Active Std",
        "active_max": "Active Max",
        "active_min": "Active Min",
        "source_ip": "Source IP",
        "destination_ip": "Destination IP",
        "protocol": "Protocol",
        "label": "Label",
    }

    df_synth.rename(columns=SYNTH_TO_ORIGINAL, inplace=True)
    
    print(f"Loaded {len(df_synth):,} synthetic rows")
    
    # --- LABEL MAPPING ---
    # Map to binary labels (True = attack) matching autoencoder format
    label_mapping = ENCODING.LABEL_MAPPING
    
    # Also handle already-grouped labels from our synthetic pipeline
    grouped_to_raw = {
        'Brute Force': 'FTP-Patator',
        'Web Attack': 'Web Attack \x96 Brute Force',
        'Botnet ARES': 'Bot',
        'DoS/DDoS': 'DoS Hulk',
        'PortScan': 'PortScan',
        'Benign': 'BENIGN',
    }
    
    # Handle lowercase column names from synthetic CSV
    if 'label' in df_synth.columns and 'Label' not in df_synth.columns:
        df_synth.rename(columns={'label': 'Label'}, inplace=True)
    # Create Full Label (raw attack name for flag_by_type)
    df_synth['Full Label'] = df_synth['Label'].map(
        lambda x: grouped_to_raw.get(str(x).strip(), str(x).strip())
    )
    
    # Create binary Label
    df_synth['Label'] = df_synth['Full Label'].map(
        lambda x: label_mapping.get(str(x).strip(), True)  # Default to True (attack) for unknowns
    )
    
    # Add synthetic marker to Full Label so we can track them
    df_synth['Full Label'] = df_synth['Full Label'].apply(lambda x: f"[SYNTH] {x}")
    
    # Timestamp placeholder
    df_synth['Timestamp'] = '01/01/2026 00:00'
    
    print(f"Synthetic label distribution:")
    print(df_synth['Full Label'].value_counts().to_string())
    
    # --- FEATURE SELECTION ---
    # Select the columns that get normalised
    available_norm_cols = [c for c in FEATURE_COLS_NORM if c in df_synth.columns]
    missing_norm_cols = [c for c in FEATURE_COLS_NORM if c not in df_synth.columns]
    
    if missing_norm_cols:
        print(f"\nWARNING: Missing columns in synthetic data: {missing_norm_cols}")
        print("These will be filled with 0.")
    
    # Start building the processed dataframe
    processed = pd.DataFrame()
    
    # Add normalised columns
    for col in FEATURE_COLS_NORM:
        if col in df_synth.columns:
            processed[col] = pd.to_numeric(df_synth[col], errors='coerce').fillna(0)
        else:
            processed[col] = 0.0
    
    # --- NORMALISATION ---
    # Fit scaler on the reference data (same as training pipeline)
    ref_features = reference_df[FEATURE_COLS_NORM].copy()
    for col in FEATURE_COLS_NORM:
        ref_features[col] = pd.to_numeric(ref_features[col], errors='coerce').fillna(0)
    
    scaler = MinMaxScaler()
    scaler.fit(ref_features)
    processed[FEATURE_COLS_NORM] = scaler.transform(processed[FEATURE_COLS_NORM])
    
    # --- PROTOCOL ONE-HOT ENCODING ---
    if 'Protocol' in df_synth.columns:
        proto = pd.to_numeric(df_synth['Protocol'], errors='coerce').fillna(0).astype(int)
        processed['Protocol_0'] = (proto == 0).astype(float)
        processed['Protocol_6'] = (proto == 6).astype(float)
        processed['Protocol_17'] = (proto == 17).astype(float)
        # Duplicate columns that exist in the test pickle
        processed['Protocol_0.0'] = processed['Protocol_0']
        processed['Protocol_6.0'] = processed['Protocol_6']
        processed['Protocol_17.0'] = processed['Protocol_17']
    else:
        for col in PROTOCOL_ONEHOT_COLS:
            processed[col] = 0.0
    
    # --- IP ADDRESS ENCODING ---
    internal_ips = ENCODING.INTERNAL_NETWORK_IPs
    public_ips = ENCODING.PUBLIC_FACING_IPs
    
    if 'Source IP' in df_synth.columns:
        src_ip = df_synth['Source IP'].astype(str)
        dst_ip = df_synth['Destination IP'].astype(str) if 'Destination IP' in df_synth.columns else pd.Series(['0.0.0.0'] * len(df_synth))
    elif 'source_ip' in df_synth.columns:
        src_ip = df_synth['source_ip'].astype(str)
        dst_ip = df_synth['destination_ip'].astype(str) if 'destination_ip' in df_synth.columns else pd.Series(['0.0.0.0'] * len(df_synth))
    else:
        src_ip = pd.Series(['0.0.0.0'] * len(df_synth))
        dst_ip = pd.Series(['0.0.0.0'] * len(df_synth))
    
    processed['Internal Source IP'] = src_ip.isin(internal_ips).astype(int)
    processed['Internal Dest IP'] = dst_ip.isin(internal_ips).astype(int)
    processed['Public Facing Source IP'] = src_ip.isin(public_ips).astype(int)
    processed['Public Facing Dest IP'] = dst_ip.isin(public_ips).astype(int)
    processed['External Source IP'] = (~src_ip.isin(internal_ips + public_ips)).astype(int)
    processed['External Dest IP'] = (~dst_ip.isin(internal_ips + public_ips)).astype(int)
    
    # --- SCORING COLUMNS ---
    processed['Label'] = df_synth['Label'].values
    processed['Timestamp'] = df_synth['Timestamp'].values
    processed['Full Label'] = df_synth['Full Label'].values
    
    # --- ENSURE COLUMN ORDER MATCHES ---
    ref_columns = reference_df.columns.tolist()
    
    # Add any missing columns as 0
    for col in ref_columns:
        if col not in processed.columns:
            processed[col] = 0.0
    
    # Reorder to match
    processed = processed[ref_columns]
    
    print(f"\nProcessed synthetic data shape: {processed.shape}")
    print(f"Reference test data shape: {reference_df.shape}")
    
    return processed


def run_experiment():
    """Run the full experiment: baseline vs baseline + synthetic."""
    
    # ============================================================
    # LOAD MODEL AND DATA
    # ============================================================
    print("=" * 60)
    print("AUTOENCODER SYNTHETIC AUGMENTATION EXPERIMENT")
    print("=" * 60)
    
    print(f"\nLoading model from {MODEL_PATH}...")
    model = keras.models.load_model(MODEL_PATH, compile=False)
    model.compile(loss='mse', optimizer='adam')
    
    print(f"Loading test data from {TEST_PICKLE}...")
    test_data = pd.read_pickle(TEST_PICKLE)
    print(f"Test data shape: {test_data.shape}")
    
    # ============================================================
    # EXPERIMENT 1: BASELINE (no synthetic data)
    # ============================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: BASELINE (Real data only)")
    print("=" * 60)
    
    baseline_thresh, baseline_score = IDS.find_best_thresh(
        model, ENCODING, test_data.copy(), THRESHOLD_INCREMENTS
    )
    
    # ============================================================
    # EXPERIMENT 2: WITH SYNTHETIC ATTACKS
    # ============================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: WITH SYNTHETIC ATTACKS")
    print("=" * 60)
    
    # Process synthetic data
    synth_processed = process_synthetic_for_autoencoder(SYNTHETIC_CSV, test_data)
    
    # Only keep synthetic attack rows (not synthetic benign)
    synth_attacks = synth_processed[synth_processed['Label'] == True].copy()
    print(f"\nKeeping {len(synth_attacks):,} synthetic attack rows (excluding synthetic benign)")
    print(f"Synthetic attack types:")
    print(synth_attacks['Full Label'].value_counts().to_string())
    
    # Combine with real test data
    combined_test = pd.concat([test_data, synth_attacks], ignore_index=True)
    print(f"\nCombined test set: {len(combined_test):,} rows")
    print(f"  Real: {len(test_data):,}")
    print(f"  Synthetic attacks: {len(synth_attacks):,}")
    
    synth_thresh, synth_score = IDS.find_best_thresh(
        model, ENCODING, combined_test.copy(), THRESHOLD_INCREMENTS
    )
    
    # ============================================================
    # EXPERIMENT 3: APPLY SYNTHETIC-CALIBRATED THRESHOLD TO REAL DATA ONLY
    # ============================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: SYNTHETIC-CALIBRATED THRESHOLD ON REAL DATA")
    print("=" * 60)
    
    print(f"\nApplying synthetic-calibrated threshold ({synth_thresh:.6f}) to real-only test data...")

    test_data_clean = test_data.copy()
    test_data_clean = test_data_clean.dropna(subset=['Label'])
    test_data_clean['Label'] = test_data_clean['Label'].astype(bool)

    pred_data = IDS.test(model, test_data_clean, ENCODING)
    scored = IDS.apply_thresh(test_data_clean, pred_data, synth_thresh)

    print("\nPer-attack-type detection with synthetic-calibrated threshold:")
    IDS.flag_by_type(scored)
        
    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nBaseline threshold:              {baseline_thresh:.6f}")
    print(f"Baseline score:                  {baseline_score:,}")
    print(f"\nSynthetic-calibrated threshold:  {synth_thresh:.6f}")
    print(f"Synthetic-calibrated score:      {synth_score:,}")
    print(f"\nThreshold difference:            {synth_thresh - baseline_thresh:.6f}")
    

if __name__ == "__main__":
    run_experiment()