"""
convert_tflite.py (FINAL FIXED VERSION)
-----------------------------------------
Converts Bidirectional LSTM to TFLite with SELECT_TF_OPS.
"""

import tensorflow as tf
import numpy as np
import os


def convert():
    print("Loading model ...")
    model = tf.keras.models.load_model('models/fall_detector.h5')

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Required for Bidirectional LSTM
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter._experimental_lower_tensor_list_ops = False
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    print("Converting ...")
    tflite_model = converter.convert()

    out = 'models/fall_detector.tflite'
    with open(out, 'wb') as f:
        f.write(tflite_model)

    # ── Check file exists and has content ────────────────────────────
    size_kb = os.path.getsize(out) / 1024
    keras_kb = os.path.getsize('models/fall_detector.h5') / 1024

    print(f"\n✅ TFLite model saved → {out}")
    print(f"   Keras model size  : {keras_kb:.1f} KB")
    print(f"   TFLite model size : {size_kb:.1f} KB")

    if size_kb > 10:
        print(f"\n✅ Conversion SUCCESSFUL — file looks valid ({size_kb:.1f} KB)")
        print(f"\n📋 Deployment Instructions:")
        print(f"   This model uses SELECT_TF_OPS (required for Bidirectional LSTM).")
        print(f"   ")
        print(f"   For Raspberry Pi:")
        print(f"     pip install tensorflow  (full package needed for Flex ops)")
        print(f"     Use tf.lite.Interpreter() as normal")
        print(f"   ")
        print(f"   For Android:")
        print(f"     Add to build.gradle:")
        print(f"     implementation 'org.tensorflow:tensorflow-lite:+'")
        print(f"     implementation 'org.tensorflow:tensorflow-lite-select-tf-ops:+'")
    else:
        print(f"⚠️  File seems too small — something may have gone wrong.")


if __name__ == "__main__":
    convert()