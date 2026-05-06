import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("dataset_realistic.csv")

# -------- GRAPH 1: Distance Distribution --------
plt.figure()
plt.hist(df["distance"], bins=50)
plt.title("Distance Distribution")
plt.xlabel("Distance")
plt.ylabel("Frequency")
plt.show()

# -------- GRAPH 2: Distance vs Camera --------
plt.figure()
plt.scatter(df["distance"], df["camera"])
plt.title("Distance vs Camera Confidence")
plt.xlabel("Distance")
plt.ylabel("Camera Value")
plt.show()

# -------- GRAPH 3: Ultrasonic Activation --------
plt.figure()
plt.hist(df["ultrasonic"], bins=2)
plt.title("Ultrasonic Sensor Activation")
plt.xlabel("Ultrasonic (0 = far, 1 = near)")
plt.ylabel("Count")
plt.show()