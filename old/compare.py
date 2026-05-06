import pandas as pd
import matplotlib.pyplot as plt

# Load datasets
ideal = pd.read_csv("dataset_large.csv")
real = pd.read_csv("dataset_realistic.csv")

# -------- GRAPH 1: Distance Distribution Comparison --------
plt.figure()
plt.hist(ideal["distance"], bins=50, alpha=0.5, label="Ideal")
plt.hist(real["distance"], bins=50, alpha=0.5, label="Realistic")
plt.title("Distance Distribution Comparison")
plt.xlabel("Distance")
plt.ylabel("Frequency")
plt.legend()
plt.grid()
plt.show()

# -------- GRAPH 2: Camera Comparison --------
plt.figure()
plt.scatter(ideal["distance"], ideal["camera"], alpha=0.3, label="Ideal")
plt.scatter(real["distance"], real["camera"], alpha=0.3, label="Realistic")
plt.title("Camera Confidence Comparison")
plt.xlabel("Distance")
plt.ylabel("Camera Value")
plt.legend()
plt.grid()
plt.show()

# -------- GRAPH 3: Ultrasonic Comparison --------
plt.figure()
plt.hist(ideal["ultrasonic"], bins=2, alpha=0.5, label="Ideal")
plt.hist(real["ultrasonic"], bins=2, alpha=0.5, label="Realistic")
plt.title("Ultrasonic Sensor Comparison")
plt.xlabel("Ultrasonic Output")
plt.ylabel("Count")
plt.legend()
plt.grid()
plt.show()