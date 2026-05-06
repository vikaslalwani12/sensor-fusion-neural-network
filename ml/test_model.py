# =============================
# FINAL SENSOR FUSION TESTING
# =============================

import numpy as np
import pandas as pd
from keras.models import load_model
import joblib

# -----------------------------
# LOAD MODEL & SCALER
# -----------------------------
model = load_model("../models/sensor_fusion_model_noisy.keras", compile=False)
scaler = joblib.load("../models/scaler.pkl")

# -----------------------------
# FEATURE NAMES (IMPORTANT FIX)
# -----------------------------
columns = ["distance", "relative_speed", "angle", "camera", "ultrasonic"]

# -----------------------------
# TEST CASES (UPDATED)
# -----------------------------
test_cases = [
    ("Normal safe case", [50, 2, 0.5, 0.8, 0]),
    ("High speed danger", [20, 15, 0.2, 0.7, 0]),
    ("Camera failure", [5, 10, 0.3, 0.2, 1]),
    ("Ultrasonic failure", [5, 10, 0.3, 0.9, 0]),
    ("Conflicting sensors", [5, 12, 0.4, 0.8, 0]),
    ("Very close + high speed", [3, 20, 0.1, 0.9, 1]),

    # 🔥 NEW STRONG TEST CASES
    ("Camera completely broken", [5, 15, 0.2, 0.1, 1]),
    ("Ultrasonic stuck at 1", [30, 5, 0.5, 0.7, 1]),
    ("Far but high speed", [80, 20, 1.0, 0.4, 0]),
    ("All sensors unreliable", [10, 12, 0.5, 0.2, 0]),
]

# -----------------------------
# RUN TESTS
# -----------------------------
print("\n==============================")
print("SENSOR FUSION TEST RESULTS")
print("==============================")

for name, sample in test_cases:
    sample_df = pd.DataFrame([sample], columns=columns)

    # scale properly (fix warning)
    sample_scaled = scaler.transform(sample_df)

    pred = model.predict(sample_scaled, verbose=0)[0][0]

    print("\n----------------------------------")
    print(f"Test Case: {name}")
    print(f"Input: {sample}")
    print(f"Predicted Risk: {pred:.3f}")

print("\n==============================")
print("Testing Complete")
print