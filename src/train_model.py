"""
train_model.py
--------------
Loads features.csv, builds sliding-window sequences,
trains a Bidirectional LSTM model, saves model + scaler.
"""

import numpy as np
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, Bidirectional, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)

# ── Config ────────────────────────────────────────────────────────────────────
SEQUENCE_LENGTH = 20
BATCH_SIZE      = 32
EPOCHS          = 100

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
    """
    Sliding window over each video sequence separately.
    This prevents a window from spanning two different videos.
    Label = label of the LAST frame in the window.
    """
    X, y = [], []

    for seq_name in df['sequence'].unique():
        seq    = df[df['sequence'] == seq_name].sort_values('frame')
        data   = seq[FEATURES].values
        labels = seq['label'].values

        for i in range(len(data) - SEQUENCE_LENGTH):
            X.append(data[i : i + SEQUENCE_LENGTH])
            y.append(labels[i + SEQUENCE_LENGTH - 1])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def build_model(input_shape):
    """
    Bidirectional LSTM classifier.
    Reads sequence both forward and backward for better pattern detection.
    """
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
        Dropout(0.3),
        Bidirectional(LSTM(32)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    return model


def train():
    print("Loading features.csv ...")
    df = pd.read_csv('data/processed/features.csv').dropna()
    print(f"Total frames : {len(df)}")
    print(f"Fall frames  : {df['label'].sum()}")
    print(f"No-fall      : {(df['label']==0).sum()}")

    # Scale features to mean=0, std=1
    scaler = StandardScaler()
    df[FEATURES] = scaler.fit_transform(df[FEATURES])

    os.makedirs('models', exist_ok=True)
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("✅ Scaler saved → models/scaler.pkl")

    print("\nBuilding sequences ...")
    X, y = make_sequences(df)
    print(f"Sequences shape  : {X.shape}")
    print(f"Fall sequences   : {y.sum()}")
    print(f"No-fall sequences: {(y==0).sum()}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {X_train.shape}  |  Val: {X_val.shape}")

    # Compute class weights to handle imbalance
    cw = compute_class_weight(
        'balanced', classes=np.unique(y_train), y=y_train
    )
    class_weight = {0: cw[0], 1: cw[1]}
    print(f"Class weights: {class_weight}")

    model = build_model((SEQUENCE_LENGTH, len(FEATURES)))
    model.summary()

    callbacks = [
        EarlyStopping(
            monitor='val_loss', patience=15,
            restore_best_weights=True, verbose=1
        ),
        ModelCheckpoint(
            'models/fall_detector.h5', monitor='val_loss',
            save_best_only=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=7, min_lr=1e-6, verbose=1
        )
    ]

    print("\nTraining started ...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )

    model.save('models/fall_detector.h5')
    pd.DataFrame(history.history).to_csv(
        'data/processed/training_history.csv', index=False
    )
    print("\n✅ Model saved → models/fall_detector.h5")
    print("✅ Training history saved → data/processed/training_history.csv")


if __name__ == "__main__":
    train()