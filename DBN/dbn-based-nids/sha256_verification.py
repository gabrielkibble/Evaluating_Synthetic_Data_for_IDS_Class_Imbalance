"""
SHA-256 Privacy Verification
=============================
Verifies that no synthetic rows are exact copies of real source data.
Each row is hashed using SHA-256 and compared against a precomputed
hash set of all source rows.

Usage:
    python sha256_verification.py

Output is printed to console and saved to sha256_verification.txt
"""

import pandas as pd
import hashlib
import os
import sys


def hash_dataframe_rows(df):
    """Hash each row of a DataFrame using SHA-256."""
    hashes = set()
    for _, row in df.iterrows():
        h = hashlib.sha256(str(row.values).encode()).hexdigest()
        hashes.add(h)
    return hashes


def verify_dataset(name, real_files, synthetic_file, encoding='cp1252'):
    """Run SHA-256 verification for a single dataset."""
    print(f"\n{'='*60}")
    print(f"SHA-256 VERIFICATION: {name}")
    print(f"{'='*60}")

    if not os.path.exists(synthetic_file):
        print(f"Synthetic file not found: {synthetic_file}. Skipping.")
        return

    # Build hash set of real data
    print("Building hash set of real data...")
    real_hashes = set()
    total_real = 0

    for f in real_files:
        if not os.path.exists(f):
            print(f"  WARNING: {f} not found. Skipping.")
            continue
        try:
            df = pd.read_csv(f, encoding=encoding, low_memory=False)
        except Exception:
            df = pd.read_csv(f, header=None, low_memory=False)
        row_hashes = hash_dataframe_rows(df)
        real_hashes.update(row_hashes)
        total_real += len(df)
        print(f"  {f}: {len(df)} rows hashed")

    # Hash synthetic data
    print(f"\nHashing synthetic data from {synthetic_file}...")
    synth = pd.read_csv(synthetic_file, low_memory=False)
    print(f"Synthetic rows: {len(synth)}")

    # Compare
    matches = 0
    for _, row in synth.iterrows():
        h = hashlib.sha256(str(row.values).encode()).hexdigest()
        if h in real_hashes:
            matches += 1

    # Results
    print(f"\nTotal real hashes:   {len(real_hashes):,}")
    print(f"Synthetic rows:      {len(synth):,}")
    print(f"Exact matches found: {matches}")
    print(f"Verification:        {'PASS - no duplicates' if matches == 0 else 'FAIL'}")

    return {
        'dataset': name,
        'real_hashes': len(real_hashes),
        'synthetic_rows': len(synth),
        'matches': matches,
        'result': 'PASS' if matches == 0 else 'FAIL'
    }


def main():
    # CICIDS2017
    cicids_real = [
        "data/raw/Monday-WorkingHours.pcap_ISCX.csv",
        "data/raw/Tuesday-WorkingHours.pcap_ISCX.csv",
        "data/raw/Wednesday-workingHours.pcap_ISCX.csv",
        "data/raw/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        "data/raw/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        "data/raw/Friday-WorkingHours-Morning.pcap_ISCX.csv",
        "data/raw/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        "data/raw/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    ]
    cicids_synth = "synthetic_cicids_standard.csv"

    # UNSW-NB15
    unsw_real = [
        "data/raw_unsw/UNSW-NB15_1.csv",
        "data/raw_unsw/UNSW-NB15_2.csv",
        "data/raw_unsw/UNSW-NB15_3.csv",
        "data/raw_unsw/UNSW-NB15_4.csv",
    ]
    unsw_synth = "synthetic_unsw_standard.csv"

    results = []

    # Run CICIDS2017 verification
    r = verify_dataset("CICIDS2017", cicids_real, cicids_synth, encoding='cp1252')
    if r:
        results.append(r)

    # Run UNSW-NB15 verification
    r = verify_dataset("UNSW-NB15", unsw_real, unsw_synth, encoding='utf-8')
    if r:
        results.append(r)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['dataset']}: {r['real_hashes']:,} real vs {r['synthetic_rows']:,} synthetic -> {r['result']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Tee output to file
    import io

    class Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
        def flush(self):
            for s in self.streams:
                s.flush()

    log_file = open("sha256_verification.txt", "w")
    sys.stdout = Tee(sys.__stdout__, log_file)

    main()

    sys.stdout = sys.__stdout__
    log_file.close()
    print("\nResults saved to sha256_verification.txt")