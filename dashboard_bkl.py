import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- SETTINGS ----------------
st.set_page_config(layout="wide")
st.title("🚗 Intelligent Transportation System Dashboard")

# ---------------- FILE UPLOAD ----------------
file = st.file_uploader("Upload Dataset (CSV)", type=["csv"])

# ---------------- LOAD DATA ----------------
if file is not None:
    df = pd.read_csv(file)

    st.success(f"Dataset Loaded: {len(df)} rows")

    # ---------------- SIDEBAR FILTERS ----------------
    st.sidebar.header("🔧 Filters")

    dist_range = st.sidebar.slider(
        "Distance (m)",
        float(df["distance"].min()),
        float(df["distance"].max()),
        (float(df["distance"].min()), float(df["distance"].max()))
    )

    speed_range = st.sidebar.slider(
        "Relative Speed (m/s)",
        float(df["relative_speed"].min()),
        float(df["relative_speed"].max()),
        (float(df["relative_speed"].min()), float(df["relative_speed"].max()))
    )

    risk_range = st.sidebar.slider(
        "Risk Score",
        float(df["risk_score"].min()),
        float(df["risk_score"].max()),
        (float(df["risk_score"].min()), float(df["risk_score"].max()))
    )

    # ---------------- APPLY FILTER ----------------
    filtered = df[
        (df["distance"] >= dist_range[0]) & (df["distance"] <= dist_range[1]) &
        (df["relative_speed"] >= speed_range[0]) & (df["relative_speed"] <= speed_range[1]) &
        (df["risk_score"] >= risk_range[0]) & (df["risk_score"] <= risk_range[1])
    ]

    st.subheader("📊 Filtered Dataset Preview")
    st.dataframe(filtered.head())

    # ---------------- SUMMARY ----------------
    st.subheader("📈 Statistical Summary")
    st.write(filtered.describe())

    # ---------------- PLOTS ----------------

    col1, col2 = st.columns(2)

    # Distance Distribution
    with col1:
        st.subheader("Distance Distribution")
        fig = plt.figure()
        plt.hist(filtered["distance"], bins=50)
        plt.xlabel("Distance (m)")
        plt.ylabel("Frequency")
        st.pyplot(fig)

    # Relative Speed
    with col2:
        st.subheader("Relative Speed Distribution")
        fig = plt.figure()
        plt.hist(filtered["relative_speed"], bins=50)
        plt.xlabel("Relative Speed (m/s)")
        plt.ylabel("Frequency")
        st.pyplot(fig)

    col3, col4 = st.columns(2)

    # Camera vs Distance
    with col3:
        st.subheader("Camera Confidence vs Distance")
        fig = plt.figure()
        plt.scatter(filtered["distance"], filtered["camera"], alpha=0.3)
        plt.xlabel("Distance")
        plt.ylabel("Camera Confidence")
        st.pyplot(fig)

    # Risk vs Distance
    with col4:
        st.subheader("Risk Score vs Distance")
        fig = plt.figure()
        plt.scatter(filtered["distance"], filtered["risk_score"], alpha=0.3)
        plt.xlabel("Distance")
        plt.ylabel("Risk Score")
        st.pyplot(fig)

    col5, col6 = st.columns(2)

    # Ultrasonic
    with col5:
        st.subheader("Ultrasonic Activation")
        fig = plt.figure()
        plt.hist(filtered["ultrasonic"], bins=2)
        plt.xlabel("Ultrasonic (0/1)")
        st.pyplot(fig)

    # Angle Distribution
    with col6:
        st.subheader("Angle Distribution")
        fig = plt.figure()
        plt.hist(filtered["angle"], bins=50)
        plt.xlabel("Angle (radians)")
        st.pyplot(fig)

    # ---------------- RISK ANALYSIS ----------------
    st.subheader("⚠ Risk Score Distribution")
    fig = plt.figure()
    plt.hist(filtered["risk_score"], bins=50)
    plt.xlabel("Risk Score")
    st.pyplot(fig)

    # ---------------- CORRELATION ----------------
    st.subheader("🔗 Correlation Matrix")
    st.write(filtered.corr())

else:
    st.info("👆 Upload your dataset to begin.")