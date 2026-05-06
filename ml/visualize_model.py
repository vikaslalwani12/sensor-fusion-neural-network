import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import joblib

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("../data/dataset_noisy.csv")

X = df[["distance", "relative_speed", "angle", "camera", "ultrasonic"]]
y = df["risk_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# LOAD SCALER & MODEL
# -----------------------------
scaler = joblib.load("../models/scaler.pkl")
model = load_model("../models/sensor_fusion_model_noisy.keras", compile=False)

X_test_scaled = scaler.transform(X_test)

# -----------------------------
# PREDICT
# -----------------------------
y_pred = model.predict(X_test_scaled)

# -----------------------------
# 1. ACTUAL vs PREDICTED
# -----------------------------
plt.figure()
plt.scatter(y_test, y_pred, alpha=0.3)
plt.xlabel("Actual Risk")
plt.ylabel("Predicted Risk")
plt.title("Actual vs Predicted Risk")
plt.show()

# -----------------------------
# 2. ERROR DISTRIBUTION
# -----------------------------
errors = y_test - y_pred.flatten()

plt.figure()
plt.hist(errors, bins=50)
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.title("Error Distribution")
plt.show()

# -----------------------------
# 3. RESIDUAL PLOT
# -----------------------------
plt.figure()
plt.scatter(y_pred, errors, alpha=0.3)
plt.xlabel("Predicted")
plt.ylabel("Error")
plt.title("Residual Plot")
plt.show()