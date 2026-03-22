"""
evaluate_model.py
-----------------
Loads trained model, runs evaluation, shows all metrics and plots.
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    f1_score, precision_score, recall_score
)
from sklearn.model_selection import train_test_split
import tensorflow as tf

SEQUENCE_LENGTH = 20

FEATURES = [
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


def make_sequences(df):
    X, y = [], []
    for seq_name in df['sequence'].unique():
        seq    = df[df['sequence'] == seq_name].sort_values('frame')
        data   = seq[FEATURES].values
        labels = seq['label'].values
        for i in range(len(data) - SEQUENCE_LENGTH):
            X.append(data[i : i + SEQUENCE_LENGTH])
            y.append(labels[i + SEQUENCE_LENGTH - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def evaluate():
    print("Loading model ...")
    model = tf.keras.models.load_model('models/fall_detector.h5')

    print("Loading scaler ...")
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    print("Loading features ...")
    df = pd.read_csv('data/processed/features.csv').dropna()
    df[FEATURES] = scaler.transform(df[FEATURES])

    X, y = make_sequences(df)

    # Use same split as training to get validation set
    _, X_val, _, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Validation samples: {len(X_val)}")
    print("Running predictions ...")
    y_prob = model.predict(X_val, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    # ── Print report ───────────────────────────────────────────────────
    print("\n" + "="*55)
    print("CLASSIFICATION REPORT")
    print("="*55)
    print(classification_report(
        y_val, y_pred, target_names=['No Fall', 'Fall']
    ))
    print(f"ROC-AUC Score: {roc_auc_score(y_val, y_prob):.4f}")

    # ── Plots ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Fall Detection Model — Evaluation Results', fontsize=14)

    # 1. Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['No Fall', 'Fall'],
        yticklabels=['No Fall', 'Fall'],
        ax=axes[0]
    )
    axes[0].set_title('Confusion Matrix')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')

    # 2. ROC curve
    fpr, tpr, _ = roc_curve(y_val, y_prob)
    auc = roc_auc_score(y_val, y_prob)
    axes[1].plot(fpr, tpr, 'b-', lw=2, label=f'AUC = {auc:.3f}')
    axes[1].plot([0, 1], [0, 1], '--', color='gray')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curve')
    axes[1].legend()

    # 3. Training loss history
    try:
        hist = pd.read_csv('data/processed/training_history.csv')
        axes[2].plot(hist['loss'],     label='Train loss', color='blue')
        axes[2].plot(hist['val_loss'], label='Val loss',   color='orange')
        axes[2].set_title('Training History')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss')
        axes[2].legend()
    except Exception:
        axes[2].text(0.5, 0.5, 'No history file found',
                     ha='center', va='center')

    plt.tight_layout()
    plt.savefig('data/processed/evaluation_plots.png', dpi=150)
    print("✅ Plot saved → data/processed/evaluation_plots.png")
    plt.show()

    # ── Threshold sensitivity table ────────────────────────────────────
    print("\nThreshold sensitivity (pick based on your tolerance):")
    print(f"{'Threshold':<12}{'Precision':<12}{'Recall':<12}{'F1':<10}")
    for t in [0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70]:
        yp = (y_prob >= t).astype(int)
        p  = precision_score(y_val, yp, zero_division=0)
        r  = recall_score(y_val, yp, zero_division=0)
        f1 = f1_score(y_val, yp, zero_division=0)
        print(f"{t:<12.2f}{p:<12.3f}{r:<12.3f}{f1:<10.3f}")

    print("\n💡 For fall detection: prioritise RECALL over Precision.")
    print("   A missed fall is more dangerous than a false alarm.")
    print("   Recommended threshold: 0.35 to 0.45")


if __name__ == "__main__":
    evaluate()