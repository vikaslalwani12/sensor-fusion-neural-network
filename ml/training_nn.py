# =============================
# FINAL TRAINING ON NOISY DATA
# =============================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from keras.models import Sequential
from keras.layers import Dense, Input
import joblib

# -----------------------------
# 1. LOAD DATA
# -----------------------------
df = pd.read_csv("../data/dataset_noisy.csv")

print("Dataset loaded")
print(df.head())

# -----------------------------
# 2. SELECT FEATURES
# -----------------------------
X = df[["distance", "relative_speed", "angle", "camera", "ultrasonic"]]
y = df["risk_score"]

# -----------------------------
# 3. TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 4. NORMALIZATION
# -----------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# save scaler (IMPORTANT)
joblib.dump(scaler, "../models/scaler.pkl")

# -----------------------------
# 5. BUILD MODEL
# -----------------------------
model = Sequential([
    Input(shape=(5,)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1)  # regression output
])

model.compile(
    optimizer='adam',
    loss='mse'
)

model.summary()

# -----------------------------
# 6. TRAIN MODEL
# -----------------------------
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2
)

# -----------------------------
# 7. EVALUATE MODEL
# -----------------------------
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")
print("MSE:", mse)
print("MAE:", mae)
print("R² :", r2)

# -----------------------------
# 8. SAMPLE PREDICTIONS
# -----------------------------
print("\nSample Predictions:\n")

for i in range(10):
    print(f"Actual: {y_test.iloc[i]:.3f} | Predicted: {y_pred[i][0]:.3f}")

# -----------------------------
# 9. SAVE MODEL
# -----------------------------
model.save("../models/sensor_fusion_model_noisy.keras")

print("\nModel saved successfully!")