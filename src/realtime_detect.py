"""
realtime_detect.py (FIXED - uses new MediaPipe Tasks API)
----------------------------------------------------------
Live webcam fall detection using MediaPipe Pose + trained LSTM model.
Press Q to quit.
"""

import cv2
import math
import time
import pickle
import urllib.request
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import deque
from alert import send_alert

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH      = 'models/fall_detector.h5'
SCALER_PATH     = 'models/scaler.pkl'
SEQUENCE_LENGTH = 20
FALL_THRESHOLD  = 0.45
ALERT_COOLDOWN  = 30

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

# ── Download pose landmarker model if not present ─────────────────────────────
POSE_MODEL_PATH = 'models/pose_landmarker.task'
if not os.path.exists(POSE_MODEL_PATH):
    print("Downloading MediaPipe pose model (~30MB, one time only) ...")
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
    urllib.request.urlretrieve(url, POSE_MODEL_PATH)
    print("✅ Pose model downloaded.")

# ── Load fall detection model and scaler ─────────────────────────────────────
print("Loading fall detection model ...")
fall_model = tf.keras.models.load_model(MODEL_PATH)
print("Loading scaler ...")
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)
print("✅ Model and scaler loaded.")

# ── MediaPipe new API setup ───────────────────────────────────────────────────
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
from mediapipe import Image, ImageFormat

base_options = mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
options = PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=mp_vision.RunningMode.IMAGE
)
landmarker = PoseLandmarker.create_from_options(options)
print("✅ MediaPipe Pose Landmarker ready.")


def draw_landmarks_manual(frame, landmarks, h, w):
    """Draw pose skeleton manually using the new API landmarks."""
    connections = [
        (11, 12), (11, 13), (13, 15),  # left arm
        (12, 14), (14, 16),             # right arm
        (11, 23), (12, 24),             # torso sides
        (23, 24),                       # hips
        (23, 25), (25, 27),             # left leg
        (24, 26), (26, 28),             # right leg
        (0, 11),  (0, 12),              # head to shoulders
    ]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for i, j in connections:
        if i < len(pts) and j < len(pts):
            cv2.line(frame, pts[i], pts[j], (255, 255, 0), 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (0, 255, 0), -1)


def extract_features(landmarks, prev_feat):
    """Extract fall-detection features from MediaPipe landmarks."""
    lm = landmarks

    all_x = [lm[i].x for i in [0, 11, 12, 23, 24, 27, 28]]
    all_y = [lm[i].y for i in [0, 11, 12, 23, 24, 27, 28]]

    bbox_w = max(all_x) - min(all_x) + 1e-6
    bbox_h = max(all_y) - min(all_y) + 1e-6

    HeightWidthRatio     = bbox_h / bbox_w

    hip_cy = (lm[23].y + lm[24].y) / 2
    sho_cy = (lm[11].y + lm[12].y) / 2
    nose_y = lm[0].y
    ank_cy = (lm[27].y + lm[28].y) / 2

    torso_len            = abs(sho_cy - hip_cy) + 1e-6
    shoulder_width       = abs(lm[11].x - lm[12].x) + 1e-6
    MajorMinorRatio      = torso_len / shoulder_width

    body_h               = abs(ank_cy - nose_y) + 1e-6
    BoundingBoxOccupancy = min(body_h / (bbox_h + 1e-6), 1.0)

    MaxStdXZ             = float(np.std(all_x) + np.std(all_y))
    HHmaxRatio           = bbox_h / 0.8
    H                    = body_h * 1000
    D                    = (1.0 - hip_cy) * 1000

    low_pts              = sum(1 for yy in all_y if yy > 0.6)
    P40                  = low_pts / len(all_y)

    feat = {
        'HeightWidthRatio':     HeightWidthRatio,
        'MajorMinorRatio':      MajorMinorRatio,
        'BoundingBoxOccupancy': BoundingBoxOccupancy,
        'MaxStdXZ':             MaxStdXZ,
        'HHmaxRatio':           HHmaxRatio,
        'H':                    H,
        'D':                    D,
        'P40':                  P40,
        'HHmaxRatio_velocity':  HHmaxRatio - prev_feat.get('HHmaxRatio', HHmaxRatio),
        'D_velocity':           D          - prev_feat.get('D', D),
    }
    return feat


def run():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam.")
        return

    buf            = deque(maxlen=SEQUENCE_LENGTH)
    prev_feat      = {}
    last_alert     = 0
    fall_prob_disp = 0.0
    is_fall        = False

    print("▶  Webcam running. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        disp = frame.copy()

        # Convert to MediaPipe Image format
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
        result   = landmarker.detect(mp_image)

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            landmarks = result.pose_landmarks[0]

            # Draw skeleton
            draw_landmarks_manual(disp, landmarks, h, w)

            # Extract features
            feat      = extract_features(landmarks, prev_feat)
            prev_feat = feat

            # ── FIX 1: use DataFrame so scaler doesn't warn ───────────
            vec_df = pd.DataFrame(
                [[feat[f] for f in FEATURES]],
                columns=FEATURES
            )
            vec_sc = scaler.transform(vec_df)[0]
            buf.append(vec_sc)

            # Run prediction when buffer is full
            if len(buf) == SEQUENCE_LENGTH:
                X              = np.array([list(buf)], dtype=np.float32)
                fall_prob_disp = float(fall_model.predict(X, verbose=0)[0][0])
                is_fall        = fall_prob_disp >= FALL_THRESHOLD

                # Send alert with cooldown
                if is_fall and (time.time() - last_alert > ALERT_COOLDOWN):
                    send_alert(fall_prob_disp)
                    last_alert = time.time()

        # ── UI overlay ────────────────────────────────────────────────
        # Dark top bar background
        overlay = disp.copy()
        cv2.rectangle(overlay, (0, 0), (w, 55), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, disp, 0.45, 0, disp)

        # Probability bar — green → orange → red
        bar_w = int(w * fall_prob_disp)
        color = (0, 200, 0)   if fall_prob_disp < 0.35 else \
                (0, 140, 255) if fall_prob_disp < 0.55 else \
                (0, 0, 220)
        cv2.rectangle(disp, (0, 40), (bar_w, 53), color, -1)

        cv2.putText(
            disp,
            f"Fall probability: {fall_prob_disp*100:.1f}%",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

        # Fall detected red border + text
        if is_fall:
            cv2.rectangle(disp, (0, 55), (w, h), (0, 0, 220), 5)
            cv2.putText(
                disp, "FALL DETECTED",
                (w // 2 - 160, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 220), 3
            )

        cv2.imshow("Fall Detection System  |  Press Q to quit", disp)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print("Detection stopped.")


if __name__ == "__main__":
    run()