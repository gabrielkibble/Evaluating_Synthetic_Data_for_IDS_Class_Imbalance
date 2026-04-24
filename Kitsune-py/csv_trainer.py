import pandas as pd
import numpy as np
import time
from sklearn.metrics import roc_auc_score, precision_recall_curve, classification_report, confusion_matrix
from KitNET.KitNET import KitNET  # Adjust import path as needed

################ USER CONFIG #################
CSV_PATH = "/mnt/vurm/homes/homes/gk634/Dissertation/DBN/dbn-based-nids/data/raw/Monday-WorkingHours.pcap_ISCX.csv"
LABEL_COL_NAME = "label_binary"  # We'll create this
ATTACK_TYPE_COL = "label_original"  # We'll create this
# Limit rows for testing? (Set None for full dataset)
LIMIT_ROWS = 20000 
##############################################

def main():
    # 1. Load Data
    import glob

    print("Loading CICIDS2017 data...")
    files = glob.glob("/mnt/vurm/homes/homes/gk634/Dissertation/DBN/dbn-based-nids/data/raw/*.csv")
    files = [f for f in files if 'synthetic' not in f.lower()]

    dfs = []
    for f in sorted(files):
        print(f"  Reading {f}...")
        temp = pd.read_csv(f, encoding='cp1252', low_memory=False)
        temp.columns = [c.strip() for c in temp.columns]
        
        # Sample: keep all attacks + sample benign
        benign = temp[temp['Label'].str.strip() == 'BENIGN'].sample(n=min(2000, len(temp[temp['Label'].str.strip() == 'BENIGN'])), random_state=42)
        attacks = temp[temp['Label'].str.strip() != 'BENIGN'].sample(n=min(2000, len(temp[temp['Label'].str.strip() != 'BENIGN'])), random_state=42)
        dfs.append(pd.concat([benign, attacks], ignore_index=True))
        print(f"    Benign: {len(benign)}, Attacks: {len(attacks)}")

    df = pd.concat(dfs, ignore_index=True)

    # Create binary label and keep original
    df['label_original'] = df['Label']
    df['label_binary'] = df['Label'].apply(lambda x: 0 if str(x).strip() == 'BENIGN' else 1)
    df = df.drop(columns=['Flow ID', 'Source IP', 'Destination IP', 'Timestamp', 'Label'], errors='ignore')

    LABEL_COL_NAME = "label_binary"
    ATTACK_TYPE_COL = "label_original"
    
    # --- THE FIX: SORT BY LABEL TO ENSURE CLEAN TRAINING ---
    # We assume '0' is Benign and '1' is Attack. 
    # We sort values so all '0's come first.
    print("Sorting dataset to put BENIGN traffic first for training...")
    df = df.sort_values(by=LABEL_COL_NAME, ascending=True)
    
    # reset_index is crucial so the loop iterates correctly
    df = df.reset_index(drop=True) 
    
    # Check if we actually have enough benign data for the grace period
    num_benign = len(df[df[LABEL_COL_NAME] == 0])
    print(f"Dataset contains {num_benign} Benign rows out of {len(df)} total.")
    
    if num_benign < 5000:
        print("WARNING: Not enough Benign data for a clean Grace Period!")
        # You might need to lower the grace period in the KitNET init line
    # -------------------------------------------------------
    # ... (After sorting df so Benign is at the top) ...

    # 1.1 Extract the Benign rows
    df_benign = df[df[LABEL_COL_NAME] == 0]
    df_attack = df[df[LABEL_COL_NAME] == 1]
    
    print(f"Original Benign Count: {len(df_benign)}")

    # 1.2 CHECK: Do we have enough?
    REQUIRED_GRACE = 5000
    if len(df_benign) < REQUIRED_GRACE:
        print(f"Warning: Only {len(df_benign)} benign rows. Duplicating to satisfy Grace Period...")
        
        # Calculate how many times we need to repeat the benign data
        repeat_factor = (REQUIRED_GRACE // len(df_benign)) + 1
        
        # Duplicate the benign data
        df_benign_extended = pd.concat([df_benign] * repeat_factor, ignore_index=True)
        
        # Trim it to be just a bit larger than the grace period if you want, 
        # or just keep it all. Let's keep the first 6000 for safety.
        df_benign_extended = df_benign_extended.iloc[:6000]
        
        # 3. Stitch it back together: [Big Chunk of Normal] + [Attacks]
        df = pd.concat([df_benign_extended, df_attack], ignore_index=True)
        
        print(f"New Benign Count (Artificial): {len(df_benign_extended)}")
        print(f"New Total Dataset Size: {len(df)}")
    
    # 2. "The Split": Save Labels separately
    # We need binary labels (0/1) for metrics
    y_true = df[LABEL_COL_NAME].values
    
    # We (optionally) save the text attack types for the detailed breakdown later
    if ATTACK_TYPE_COL in df.columns:
        y_types = df[ATTACK_TYPE_COL].values
    else:
        y_types = None

    # 3. Clean Features (X)
    # Drop Label columns and any text columns
    # Note: UNSW-NB15 has text columns like 'proto', 'service', 'state'. We MUST drop them or encode them.
    # Here we drop them for simplicity.
    df_features = df.drop(columns=[LABEL_COL_NAME], errors='ignore')
    if ATTACK_TYPE_COL in df.columns:
        df_features = df_features.drop(columns=[ATTACK_TYPE_COL], errors='ignore')
    
    # Auto-drop any other non-numeric columns (IPs, strings)
    df_features = df_features.select_dtypes(include=[np.number])
    
    # Normalize using ONLY benign statistics (first 16000 rows after sort)
    # This prevents attack outliers from skewing the normalization
    benign_data = df_features.iloc[:num_benign]
    feat_min = benign_data.min()
    feat_max = benign_data.max()
    df_features = (df_features - feat_min) / (feat_max - feat_min + 1e-6)
    df_features = df_features.clip(-10, 10)  # Clip extreme outliers
    df_features = df_features.fillna(0)
    df_features = df_features.replace([np.inf, -np.inf], 0)

    # DEBUG: Find problematic columns
    print("\nDEBUG: Checking for extreme values after normalization...")
    for col in df_features.columns:
        col_max = df_features[col].max()
        col_min = df_features[col].min()
        benign_max = df_features[col].iloc[:num_benign].max()
        if col_max > 5 or col_min < -5 or benign_max > 5:
            print(f"  SUSPECT: {col} | benign_max={benign_max:.4f} | full_max={col_max:.4f} | full_min={col_min:.4f}")

    # Also check for NaN in the numpy array
    X = df_features.to_numpy()
    print(f"\nDEBUG: NaN count in X: {np.isnan(X).sum()}")
    print(f"DEBUG: Inf count in X: {np.isinf(X).sum()}")
    print(f"DEBUG: Max value in X: {np.max(X):.4f}")
    print(f"DEBUG: Max value in benign rows: {np.max(X[:num_benign]):.4f}")
    print(f"Features ready: {X.shape}")

    # 4. Initialize KitNET
    max_ae = 10 
    # Grace period: Train on first 10% (Assuming usually benign start) or fixed number
    grace_period = 10000

    print(f"First {grace_period} rows label distribution:")
    print(pd.Series(y_true[:grace_period]).value_counts())

    K = KitNET(X.shape[1], max_ae, int(grace_period*0.1), grace_period)

    # 5. Run Detection
    print("Running Kitsune (this may take time)...")
    scores = []
    start_time = time.time()
    
    for i in range(len(X)):
        rmse = K.process(X[i])
        scores.append(rmse)
        if i % 5000 == 0:
            print(f"Processing row {i}...", end='\r')
            
    print(f"\nFinished in {time.time() - start_time:.2f}s")

    # Cap exploding scores at a reasonable maximum
    score_cap = 1.0  # No legitimate RMSE should exceed 1.0 with normalized inputs
    n_capped = sum(1 for s in scores if s > score_cap)
    scores = [min(s, score_cap) for s in scores]
    print(f"Capped {n_capped} exploding scores at {score_cap}")

    # Add here:
    benign_scores = [scores[i] for i in range(grace_period, len(scores)) if y_true[i] == 0]
    attack_scores = [scores[i] for i in range(grace_period, len(scores)) if y_true[i] == 1]

    print(f"Benign RMSE: mean={np.mean(benign_scores):.6f}, std={np.std(benign_scores):.6f}")
    print(f"Benign RMSE: median={np.median(benign_scores):.6f}")
    print(f"Attack RMSE: mean={np.mean(attack_scores):.6f}, std={np.std(attack_scores):.6f}")

    # Find the outliers
    benign_sorted = sorted(benign_scores, reverse=True)
    print(f"\nTop 10 benign RMSE scores: {benign_sorted[:10]}")
    print(f"Bottom 10 benign RMSE scores: {benign_sorted[-10:]}")
    print(f"Benign scores > 1.0: {sum(1 for s in benign_scores if s > 1.0)}")
    print(f"Benign scores > 100: {sum(1 for s in benign_scores if s > 100)}")

    # 6. CALCULATE METRICS
    
    # Filter out the grace period (training phase) from evaluation
    # We can't evaluate accuracy while it was still learning!
    y_true_eval = y_true[grace_period:]
    scores_eval = scores[grace_period:]
    if y_types is not None:
        y_types_eval = y_types[grace_period:]

    # 6. CALCULATE METRICS
    
    # A. Area Under ROC Curve (The best "Threshold-Free" metric)
    try:
        auc = roc_auc_score(y_true_eval, scores_eval)
        print(f"\n>>> Global AUROC Score: {auc:.4f} (1.0 is perfect)")
    except:
        print("Could not calc AUC (maybe only one class in dataset?)")

    # B. Determine Best Threshold (Maximize F1 Score)
    # Since Kitsune gives a float, we need to pick a cutoff line.
    precision, recall, thresholds = precision_recall_curve(y_true_eval, scores_eval)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    
    print(f">>> Optimal Threshold found: {best_threshold:.4f}")

    # C. Binary Classification Report
    # Convert continuous scores to 0 or 1 based on threshold
    y_pred = [1 if s > best_threshold else 0 for s in scores_eval]
    
    print("\n--- Classification Report ---")
    print(classification_report(y_true_eval, y_pred, target_names=['Benign', 'Attack']))
    
    # D. "The Detail": Per-Attack Detection Rate
    if y_types is not None:
        print("\n--- Detection Rate by Attack Type ---")
        results_df = pd.DataFrame({'Type': y_types_eval, 'Pred': y_pred})
        # Calculate what % of each attack type was flagged as '1'
        breakdown = results_df.groupby('Type')['Pred'].mean()
        print(breakdown)

if __name__ == "__main__":
    main()