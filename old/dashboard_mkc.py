import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("🚗 Intelligent Transportation System Dashboard")

# ---------- UPLOAD ----------
col1, col2 = st.columns(2)

with col1:
    file1 = st.file_uploader("Upload Dataset 1 (Ideal / Realistic)", type=["csv"])

with col2:
    file2 = st.file_uploader("Upload Dataset 2 (Optional)", type=["csv"])

df1 = pd.read_csv(file1) if file1 else None
df2 = pd.read_csv(file2) if file2 else None


# ---------- FUNCTION ----------
def plot_all(df, label):
    st.subheader(f"📊 {label} Dataset Overview")

    st.dataframe(df.head())

    st.write("### Statistical Summary")
    st.write(df.describe())

    col1, col2 = st.columns(2)

    # Distance
    with col1:
        fig = plt.figure()
        plt.hist(df["distance"], bins=60)
        plt.title("Distance Distribution")
        plt.xlabel("Distance (meters)")
        plt.ylabel("Frequency")
        st.pyplot(fig)

    # Relative Speed
    with col2:
        fig = plt.figure()
        plt.hist(df["relative_speed"], bins=60)
        plt.title("Relative Speed Distribution")
        plt.xlabel("Relative Speed (m/s)")
        plt.ylabel("Frequency")
        st.pyplot(fig)

    col3, col4 = st.columns(2)

    # Camera
    with col3:
        fig = plt.figure()
        plt.scatter(df["distance"], df["camera"], alpha=0.2)
        plt.title("Camera Confidence vs Distance")
        plt.xlabel("Distance (m)")
        plt.ylabel("Camera Confidence")
        st.pyplot(fig)

    # Angle
    with col4:
        fig = plt.figure()
        plt.hist(df["angle"], bins=60)
        plt.title("Angle Distribution")
        plt.xlabel("Angle (radians)")
        plt.ylabel("Frequency")
        st.pyplot(fig)

    col5, col6 = st.columns(2)

    # Ultrasonic
    with col5:
        fig = plt.figure()
        plt.hist(df["ultrasonic"], bins=2)
        plt.title("Ultrasonic Activation")
        plt.xlabel("0 = Far, 1 = Near")
        plt.ylabel("Count")
        st.pyplot(fig)

    # Risk
    with col6:
        fig = plt.figure()
        plt.hist(df["risk_score"], bins=60)
        plt.title("Risk Score Distribution")
        plt.xlabel("Risk Score")
        plt.ylabel("Frequency")
        st.pyplot(fig)

    st.write("### Correlation Matrix")
    st.write(df.corr())


# ---------- SINGLE VIEW ----------
if df1 is not None and df2 is None:
    plot_all(df1, "Single")


elif df2 is not None and df1 is None:
    plot_all(df2, "Single")


# ---------- COMPARISON ----------
elif df1 is not None and df2 is not None:

    st.header("📊 Dataset Comparison")

    colA, colB = st.columns(2)

    with colA:
        st.subheader("Dataset 1")
        st.dataframe(df1.head())

    with colB:
        st.subheader("Dataset 2")
        st.dataframe(df2.head())

    # Distance comparison
    fig = plt.figure()
    plt.hist(df1["distance"], bins=60, alpha=0.5, label="Dataset 1")
    plt.hist(df2["distance"], bins=60, alpha=0.5, label="Dataset 2")
    plt.title("Distance Distribution Comparison")
    plt.xlabel("Distance (m)")
    plt.ylabel("Frequency")
    plt.legend()
    st.pyplot(fig)

    # Camera comparison
    fig = plt.figure()
    plt.scatter(df1["distance"], df1["camera"], alpha=0.2, label="Dataset 1")
    plt.scatter(df2["distance"], df2["camera"], alpha=0.2, label="Dataset 2")
    plt.title("Camera Confidence Comparison")
    plt.xlabel("Distance")
    plt.ylabel("Camera Confidence")
    plt.legend()
    st.pyplot(fig)

    # Risk comparison
    fig = plt.figure()
    plt.hist(df1["risk_score"], bins=60, alpha=0.5, label="Dataset 1")
    plt.hist(df2["risk_score"], bins=60, alpha=0.5, label="Dataset 2")
    plt.title("Risk Score Comparison")
    plt.xlabel("Risk Score")
    plt.ylabel("Frequency")
    plt.legend()
    st.pyplot(fig)


# ---------- DEFAULT ----------
else:
    st.info("👆 Upload at least one dataset to begin.")