import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("🚗 Intelligent Transportation System Dashboard")

file = st.file_uploader("Upload Dataset (Ideal or Realistic)", type=["csv"])

if file is not None:
    df = pd.read_csv(file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Statistical Summary")
    st.write(df.describe())

    col1, col2 = st.columns(2)

    # Distance
    with col1:
        st.subheader("Distance Distribution")
        fig = plt.figure()
        plt.hist(df["distance"], bins=50)
        plt.xlabel("Distance (m)")
        st.pyplot(fig)

    # Relative Speed
    with col2:
        st.subheader("Relative Speed Distribution")
        fig = plt.figure()
        plt.hist(df["relative_speed"], bins=50)
        plt.xlabel("Relative Speed (m/s)")
        st.pyplot(fig)

    col3, col4 = st.columns(2)

    # Camera
    with col3:
        st.subheader("Camera Confidence vs Distance")
        fig = plt.figure()
        plt.scatter(df["distance"], df["camera"], alpha=0.3)
        plt.xlabel("Distance")
        plt.ylabel("Camera Confidence")
        st.pyplot(fig)

    # Angle
    with col4:
        st.subheader("Angle Distribution")
        fig = plt.figure()
        plt.hist(df["angle"], bins=50)
        plt.xlabel("Angle (radians)")
        st.pyplot(fig)

    col5, col6 = st.columns(2)

    # Ultrasonic
    with col5:
        st.subheader("Ultrasonic Activation")
        fig = plt.figure()
        plt.hist(df["ultrasonic"], bins=2)
        st.pyplot(fig)

    # Risk Score
    with col6:
        st.subheader("Risk Score Distribution")
        fig = plt.figure()
        plt.hist(df["risk_score"], bins=50)
        st.pyplot(fig)

    st.subheader("Correlation Matrix")
    st.write(df.corr())