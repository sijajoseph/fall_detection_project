# Elderly Fall Detection and SOS Alert System using Edge AI

A computer vision based fall detection system that works without wearables,
using a fixed camera (smartphone/webcam as CCTV).

## System Overview
- **Pose Estimation**: MediaPipe Pose Landmarker (33 keypoints)
- **Model**: Bidirectional LSTM trained on UR Fall Detection Dataset
- **Accuracy**: 100% | Recall: 100% | ROC-AUC: 1.000
- **Alert**: Automatic email + SMS on fall detection
- **Edge Deployment**: TFLite model (141 KB) for Raspberry Pi / Android

## Dataset
UR Fall Detection Dataset — University of Rzeszów
- 30 fall sequences + 40 ADL sequences
- 10,644 frames total after preprocessing

## Project Structure
```
fall_detection_project/
├── src/
│   ├── prepare_dataset.py   # Dataset preprocessing
│   ├── train_model.py       # LSTM model training
│   ├── evaluate_model.py    # Metrics and plots
│   ├── realtime_detect.py   # Live webcam detection
│   ├── alert.py             # Email/SMS SOS alerts
│   └── convert_tflite.py    # Edge deployment conversion
├── data/
│   └── raw/                 # Place UR Fall dataset CSVs here
├── models/                  # Trained models saved here
└── requirements.txt
```

## Setup Instructions

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd fall_detection_project
```

### 2. Create virtual environment
```bash
py -3.11 -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download dataset
Download these 2 files from http://fenix.ur.edu.pl/mkepski/ds/uf.html
and place them in `data/raw/`:
- `urfall-cam0-falls.csv`
- `urfall-cam0-adls.csv`

### 5. Run the pipeline
```bash
python src/prepare_dataset.py   # Process dataset
python src/train_model.py       # Train model
python src/evaluate_model.py    # Evaluate
python src/realtime_detect.py   # Live detection
python src/convert_tflite.py    # Convert for edge
```

## Results
| Metric | Score |
|--------|-------|
| Accuracy | 100% |
| Precision | 99.4% |
| Recall | 100% |
| F1-Score | 99.7% |
| ROC-AUC | 1.000 |

## Alert System
Configure `src/alert.py` with your Gmail App Password for email alerts.
Optional Twilio SMS integration also available.

## Edge Deployment
The TFLite model (141 KB) can be deployed on:
- Raspberry Pi 4 (with full tensorflow package)
- Android (with tensorflow-lite-select-tf-ops dependency)

## References
Kwolek B., Kepski M. (2014). Human fall detection on embedded platform
using depth maps and wireless accelerometer. Computer Methods and Programs
in Biomedicine, 117(3), 489-501.