"""
prepare_dataset.py (FIXED - 11 column version)
------------------------------------------------
Reads the 2 UR Fall Detection Dataset CSV files,
cleans them, adds velocity features, saves features.csv
Run this FIRST before training.
"""

import pandas as pd
import numpy as np
import os

FALLS_CSV  = 'data/raw/urfall-cam0-falls.csv'
ADLS_CSV   = 'data/raw/urfall-cam0-adls.csv'
OUTPUT_CSV = 'data/processed/features.csv'

# CORRECT column names — 11 columns total (verified from actual CSV)
COLUMNS = [
    'sequence',
    'frame',
    'label_raw',
    'HeightWidthRatio',
    'MajorMinorRatio',
    'BoundingBoxOccupancy',
    'MaxStdXZ',
    'HHmaxRatio',
    'H',
    'D',
    'P40'
]

FEATURE_COLS = [
    'HeightWidthRatio',
    'MajorMinorRatio',
    'BoundingBoxOccupancy',
    'MaxStdXZ',
    'HHmaxRatio',
    'H',
    'D',
    'P40',
    'HHmaxRatio_velocity',
    'D_velocity',
]


def load_csv(path, is_fall_file):
    """Load one CSV, assign column names, assign binary label."""
    df = pd.read_csv(path, header=None, names=COLUMNS)
    print(f"Loaded: {path}  ({len(df)} rows)")
    print(f"  label_raw unique values: {sorted(df['label_raw'].unique())}")

    if is_fall_file:
        # Remove ambiguous mid-fall frames (label_raw == 0)
        df = df[df['label_raw'] != 0].copy()
        # label_raw 1 (lying) → binary 1 (fall)
        # label_raw -1 (standing) → binary 0 (no fall)
        df['label'] = (df['label_raw'] == 1).astype(int)
    else:
        # All ADL frames are no-fall
        df['label'] = 0

    print(f"  After filtering — Fall: {df['label'].sum()}  No-fall: {(df['label']==0).sum()}")
    return df


def add_velocity(df):
    """
    Compute frame-to-frame velocity for HHmaxRatio and D.
    Processed per sequence to avoid mixing different videos.
    """
    df = df.copy()
    df['HHmaxRatio_velocity'] = 0.0
    df['D_velocity']          = 0.0

    for seq in df['sequence'].unique():
        idx = df[df['sequence'] == seq].sort_values('frame').index
        df.loc[idx, 'HHmaxRatio_velocity'] = (
            df.loc[idx, 'HHmaxRatio'].diff().fillna(0).values
        )
        df.loc[idx, 'D_velocity'] = (
            df.loc[idx, 'D'].diff().fillna(0).values
        )
    return df


def prepare():
    os.makedirs('data/processed', exist_ok=True)

    falls = load_csv(FALLS_CSV, is_fall_file=True)
    adls  = load_csv(ADLS_CSV,  is_fall_file=False)

    df = pd.concat([falls, adls], ignore_index=True)
    print(f"\nCombined: {len(df)} rows")
    print(f"  Fall frames    : {df['label'].sum()}")
    print(f"  No-fall frames : {(df['label']==0).sum()}")

    df = add_velocity(df)

    # Show NaN counts to help debug if anything goes wrong
    nan_counts = df[FEATURE_COLS].isna().sum()
    print(f"\nNaN counts per column:\n{nan_counts}")

    df = df.dropna(subset=FEATURE_COLS)

    save_cols = ['sequence', 'frame'] + FEATURE_COLS + ['label']
    df[save_cols].to_csv(OUTPUT_CSV, index=False)

    print(f"\n✅ Saved {len(df)} rows → {OUTPUT_CSV}")
    print(f"\nFirst 5 rows preview:")
    print(df[save_cols].head().to_string())


if __name__ == "__main__":
    prepare()