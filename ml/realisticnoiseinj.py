# =============================
# REALISTIC NOISE INJECTION (FINAL VERSION)
# =============================

import pandas as pd
import numpy as np

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("../data/dataset_final.csv")

np.random.seed(42)

# -----------------------------
# 1. DISTANCE NOISE (sensor-dependent)
# -----------------------------
# error increases with distance
distance_noise = np.random.normal(0, 0.02 * df["distance"] + 0.5)
df["distance"] += distance_noise
df["distance"] = df["distance"].clip(0, None)

# -----------------------------
# 2. RELATIVE SPEED NOISE
# -----------------------------
df["relative_speed"] += np.random.normal(0, 0.5, len(df))

# allow small negative (sensor drift)
df["relative_speed"] = df["relative_speed"].clip(-5, None)

# -----------------------------
# 3. ANGLE NOISE
# -----------------------------
df["angle"] += np.random.normal(0, 0.05, len(df))

# keep angle in [-pi, pi]
df["angle"] = np.arctan2(np.sin(df["angle"]), np.cos(df["angle"]))

# -----------------------------
# 4. CAMERA NOISE (distance dependent)
# -----------------------------
camera_noise = np.random.normal(0, 0.03 + 0.001 * df["distance"])
df["camera"] += camera_noise

# clamp
df["camera"] = df["camera"].clip(0.1, 1)

# -----------------------------
# 5. CAMERA FAILURE CASES
# -----------------------------
# partial failure
mask = np.random.rand(len(df)) < 0.12
df.loc[mask, "camera"] *= np.random.uniform(0.2, 0.6)

# complete failure (rare but important)
mask_fail = np.random.rand(len(df)) < 0.02
df.loc[mask_fail, "camera"] = np.random.uniform(0.1, 0.3)

# -----------------------------
# 6. ULTRASONIC NOISE
# -----------------------------
# false positives
mask_fp = np.random.rand(len(df)) < 0.05
df.loc[mask_fp, "ultrasonic"] = 1

# false negatives (dangerous)
mask_fn = (df["distance"] < 5) & (np.random.rand(len(df)) < 0.1)
df.loc[mask_fn, "ultrasonic"] = 0

# -----------------------------
# 7. SENSOR CONFLICT (CRITICAL)
# -----------------------------

# CASE 1: close object but camera thinks safe
mask_close = df["distance"] < 10
conflict1 = mask_close & (np.random.rand(len(df)) < 0.25)
df.loc[conflict1, "camera"] = np.random.uniform(0.7, 1)

# CASE 2: far object but camera thinks danger
mask_far = df["distance"] > 50
conflict2 = mask_far & (np.random.rand(len(df)) < 0.2)
df.loc[conflict2, "camera"] = np.random.uniform(0.1, 0.4)

# -----------------------------
# 8. FINAL CLEANUP
# -----------------------------
df["camera"] = df["camera"].clip(0, 1)
df["distance"] = df["distance"].clip(0, None)

# IMPORTANT: DO NOT TOUCH risk_score
# it remains the ground truth

# -----------------------------
# 9. SAVE DATASET (OVERWRITE)
# -----------------------------
df.to_csv("../data/dataset_noisy.csv", index=False)

print("✅ Realistic noisy dataset created successfully")