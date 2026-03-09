import pandas as pd
import numpy as np
import time
from sklearn.metrics import roc_auc_score, precision_recall_curve, classification_report, confusion_matrix
from KitNET.KitNET import KitNET  # Adjust import path as needed

################ USER CONFIG #################
CSV_PATH = "balanced_bot_iot.csv" 
# The column containing 0 (Normal) and 1 (Attack)
LABEL_COL_NAME = "attack"
# The column containing text names like 'DoS', 'Reconnaissance' (Optional, for detailed breakdown)
ATTACK_TYPE_COL = "category"
# Limit rows for testing? (Set None for full dataset)
LIMIT_ROWS = 50000 
##############################################

def main():
    # 1. Load Data
    print(f"Loading {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH, nrows=LIMIT_ROWS)
    
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
    
    # Normalize Features (Crucial for Kitsune on different datasets)
    df_features = (df_features - df_features.min()) / (df_features.max() - df_features.min() + 1e-6)
    df_features = df_features.fillna(0) # Handle divide by zero
    
    X = df_features.to_numpy()
    print(f"Features ready: {X.shape}")

    # 4. Initialize KitNET
    max_ae = 10 
    # Grace period: Train on first 10% (Assuming usually benign start) or fixed number
    grace_period = int(min(len(X) * 0.1, 5000))
    
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