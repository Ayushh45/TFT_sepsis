import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tft_sepsiiss import *  # Import your model and functions

# --- Page Configuration ---
st.set_page_config(page_title="Sepsis Prediction Dashboard", page_icon="⚕️", layout="wide")

# --- Sidebar ---
st.sidebar.title("🔍 Sepsis Prediction App")
st.sidebar.subheader("Navigate")

page = st.sidebar.radio("Go to", ["Home", "Prediction", "Visualizations", "About"])

# --- Home Page ---
if page == "Home":
    st.title("Sepsis Prediction Dashboard")
    st.image("https://cdn.pixabay.com/photo/2017/08/30/07/54/heart-2698050_960_720.jpg", use_column_width=True)
    st.markdown("""
    ### 🎯 **Goal**  
    This app helps predict sepsis early based on patient data.
    
    ### ⚡ **Features**  
    - Make live predictions  
    - Visualize trends  
    - Explore patient stats  

    **Start by selecting a page from the sidebar ➡️**
    """)

# --- Prediction Page ---
if page == "Prediction":
    st.header("🩺 Make a Sepsis Prediction")
    st.write("Fill in the details below to get a prediction:")

    # User Input Form
    col1, col2, col3 = st.columns(3)
    with col1:
        hr = st.number_input("Heart Rate (HR)", min_value=30, max_value=200, value=80)
        o2sat = st.number_input("Oxygen Saturation (O2Sat)", min_value=50.0, max_value=100.0, value=95.0)
        temp = st.number_input("Temperature (°C)", min_value=30.0, max_value=45.0, value=37.0)
    with col2:
        wbc = st.number_input("White Blood Cells (WBC)", min_value=1.0, max_value=30.0, value=8.0)
        sbp = st.number_input("Systolic Blood Pressure (SBP)", min_value=50, max_value=200, value=120)
        lactate = st.number_input("Lactate", min_value=0.1, max_value=10.0, value=1.5)
    with col3:
        dbp = st.number_input("Diastolic Blood Pressure (DBP)", min_value=30, max_value=150, value=80)
        creatinine = st.number_input("Creatinine", min_value=0.1, max_value=10.0, value=1.0)
        resp = st.number_input("Respiratory Rate (Resp)", min_value=5, max_value=50, value=20)

    # Predict Button
    if st.button("Predict"):
        # Prepare input data
        patient_data = np.array([[hr, o2sat, temp, wbc, sbp, lactate, dbp, creatinine, resp]])
        patient_data_scaled = scaler.transform(patient_data)  # Ensure scaling matches training

        # Make prediction
        result = TemporalFusionTransformer.predict(patient_data_scaled)
        if result[0] == 1:
            st.error("✅ Sepsis Detected!")
        else:
            st.success("🟢 No Sepsis Detected")

# --- Visualization Page ---
if page == "Visualizations":
    st.header("📊 Data Visualizations")
    st.write("Explore patient data through charts and graphs.")

    # Load dataset (replace with your dataset)
    df = pd.read_csv("/content/sepsis_data_part_2 - Copy.csv")
    columns = ['Patient_ID', 'Temp', 'HR', 'O2Sat', 'WBC', 'SBP', 'DBP', 'Lactate', 'Resp', 'Hour', 'Creatinine']
    df = df[columns]

    # Example: Heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Example: Distribution of Vital Signs
    st.subheader("Distribution of Vital Signs")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df[["HR", "Temp", "Resp", "O2Sat", "SBP", "DBP", "WBC", "Lactate", "Creatinine"]], ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# --- About Page ---
if page == "About":
    st.title("📘 About This Project")
    st.markdown("""
    - Built with **Streamlit**
    - Data sourced from your dataset
    - Model trained using the latest machine learning techniques
    
    💡 **Idea**: This app helps hospitals monitor and predict sepsis to save lives faster.
    """)

st.sidebar.info("🚀 Created by Team Aditya")