import streamlit as st
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_forecasting import TemporalFusionTransformer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Streamlit page configuration
st.set_page_config(page_title="Sepsis Prediction", layout="wide")

# Title and description
st.title("Sepsis Prediction Dashboard")
st.write("""
This app predicts sepsis based on vital signs and other clinical data. 
You can upload your dataset and click on "Predict" to see the prediction results.
""")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load the dataset
    df = pd.read_csv(uploaded_file)
    
    # Display dataset preview
    st.write("Dataset preview:")
    st.write(df.head())

    # Preprocess data
    columns = ['Patient_ID', 'Temp', 'HR', 'O2Sat', 'WBC', 'SBP', 'DBP', 'Lactate', 'Resp', 'Hour', 'Creatinine']
    df = df[columns]

    # Handle missing values
    numeric_cols = ['Temp', 'HR', 'O2Sat', 'WBC', 'SBP', 'DBP', 'Lactate', 'Resp', 'Creatinine']
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Normalize the data
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Define stricter sepsis condition
    def sepsis_condition(row):
        return (
            (row['Temp'] > 39) or (row['Temp'] < 35) or
            (row['HR'] > 100) or (row['Resp'] > 25) or
            (row['WBC'] > 15) or (row['WBC'] < 3) or
            (row['SBP'] < 85) or (row['Lactate'] > 3.0) or
            (row['Creatinine'] > 1.5)
        )
    
    df['Sepsis_Condition'] = df.apply(sepsis_condition, axis=1).astype(int)
    
    # Create the final label
    sepsis_counts = df.groupby('Patient_ID')['Sepsis_Condition'].sum()
    df['Final_Sepsis_Label'] = df['Patient_ID'].map(lambda x: 1 if sepsis_counts[x] > 3 else 0)

    # Display Sepsis Distribution
    st.write("Sepsis Distribution:")
    st.write(df['Final_Sepsis_Label'].value_counts())

    # Load the pre-trained model
    model = TemporalFusionTransformer.load_from_checkpoint("sepsis_tft_model.ckpt")

    # Prepare the data for prediction
    df['Patient_ID'] = df['Patient_ID'].astype(str)
    df['time_idx'] = df.groupby('Patient_ID').cumcount()

    # Define the TimeSeriesDataSet
    max_encoder_length = 30
    max_prediction_length = 10

    prediction_data = TimeSeriesDataSet(
        data=df,
        time_idx="time_idx",
        target="Final_Sepsis_Label",
        group_ids=["Patient_ID"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=["Patient_ID"],
        time_varying_known_reals=numeric_cols + ['time_idx'],
        time_varying_unknown_reals=["Final_Sepsis_Label"],
        target_normalizer=None,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    # Create dataloaders for prediction
    prediction_dataloader = prediction_data.to_dataloader(train=False, batch_size=256, num_workers=4)

    # Function to get predictions
    def make_predictions():
        predictions = []
        model.eval()
        for batch in prediction_dataloader:
            x, y = batch
            output = model(x)
            predictions.append(output.prediction)
        
        # Combine predictions into a single tensor
        predictions = torch.cat(predictions, dim=0)
        return predictions

    # Predict button
    if st.button('Predict'):
        st.write("Generating predictions...")
        
        # Make predictions
        predictions = make_predictions()

        # Convert predictions to DataFrame for easy interpretation
        prediction_df = pd.DataFrame(predictions.numpy(), columns=["Sepsis Prediction"])
        
        # Display the predictions
        st.write("Predictions:")
        st.write(prediction_df)

        # Plot the predictions
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(prediction_df["Sepsis Prediction"], label="Predicted Sepsis")
        ax.set_title("Predicted Sepsis Over Time")
        ax.set_xlabel("Time")
        ax.set_ylabel("Prediction")
        ax.legend()
        st.pyplot(fig)

