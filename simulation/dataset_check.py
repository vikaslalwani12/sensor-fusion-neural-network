import pandas as pd

df = pd.read_csv("../data/dataset_final.csv")

print("\n📊 DATA SUMMARY\n")
print(df.describe())

print("\n🔊 Ultrasonic counts:")
print(df["ultrasonic"].value_counts())

print("\n⚠ Risk distribution:")
print(df["risk_score"].describe())