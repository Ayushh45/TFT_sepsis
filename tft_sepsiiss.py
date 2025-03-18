#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
# Load the dataset
df = pd.read_csv('sepsis_data_part_1.csv')


# In[2]:


# Display the first few rows
print(df.head())


# In[3]:


# Select relevant columns
columns = ['Patient_ID', 'Temp', 'HR', 'O2Sat', 'WBC', 'SBP', 'DBP', 'Lactate', 'Resp','Hour','Creatinine']
df = df[columns]

# Display the selected columns
print(df.head())


# In[4]:


df.tail()


# In[5]:


unique_patients_count = df['Patient_ID'].nunique()
print("Total unique patients:", unique_patients_count)


# In[6]:


from sklearn.impute import KNNImputer

# Separate numeric and non-numeric columns
numeric_cols = ['Temp', 'HR', 'O2Sat', 'WBC', 'SBP', 'DBP', 'Lactate', 'Resp','Creatinine']


# In[7]:


df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())


# In[8]:


import matplotlib.pyplot as plt
import seaborn as sns

# Select a sample patient
sample_patient = df[df['Patient_ID'] == df['Patient_ID'].iloc[0]]

# Sort values by Hour
sample_patient = sample_patient.sort_values(by="Hour")

# Plot vital signs over time
plt.figure(figsize=(12, 5))
for col in ["HR", "Temp", "Resp"]:
    if col in df.columns:
        plt.plot(sample_patient["Hour"], sample_patient[col], label=col, marker='o')

plt.title("Vital Signs Over Time for a Sample Patient")
plt.xlabel("Hour")
plt.ylabel("Values")
plt.legend()
plt.grid()
plt.show()


# In[9]:


plt.figure(figsize=(12, 6))
sns.boxplot(data=df[["HR", "Temp", "Resp", "O2Sat", "SBP", "DBP", "WBC", "Lactate", "Creatinine"]])
plt.xticks(rotation=45)
plt.title("Distribution of Vital Signs")
plt.show()


# In[10]:


plt.figure(figsize=(10, 6))
sns.heatmap(df.drop(columns=["Patient_ID"]).corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Vital Signs")
plt.show()


# In[11]:


plt.figure(figsize=(8, 5))
sns.scatterplot(x=df["Lactate"], y=df["HR"], hue=df["O2Sat"], palette="coolwarm")
plt.title("Lactate vs. Heart Rate (Sepsis Indicator)")
plt.xlabel("Lactate")
plt.ylabel("Heart Rate")
plt.colorbar(label="O2 Saturation")
plt.show()


# In[12]:


# Check for remaining missing values
print(df.isnull().sum())


# In[13]:


import pandas as pd

# Define the stricter sepsis condition function
def sepsis_condition(row):
    return (
        (row['Temp'] > 39) or (row['Temp'] < 35) or  # Stricter temperature
        (row['HR'] > 100) or  # Higher heart rate threshold
        (row['Resp'] > 25) or  # Stricter respiratory rate
        (row['WBC'] > 15) or (row['WBC'] < 3) or  # Adjusted WBC
        (row['SBP'] < 85) or  # Lower SBP
        (row['Lactate'] > 3.0) or  # Higher lactate
        (row['Creatinine'] > 1.5)  # Higher Creatinine
    )

# Apply the condition to create a new column
df['Sepsis_Condition'] = df.apply(sepsis_condition, axis=1).astype(int)  # 1 if True, 0 if False

# Count the number of times each patient meets the sepsis condition
sepsis_counts = df.groupby('Patient_ID')['Sepsis_Condition'].sum()

# Create the final label: if a patient meets the sepsis condition at least 3 times → Sepsis (1), else → Non-Sepsis (0)
df['Final_Sepsis_Label'] = df['Patient_ID'].map(lambda x: 1 if sepsis_counts[x] > 3  else 0)

# Print the distribution of Sepsis (1) and Non-Sepsis (0)
sepsis_distribution = df[['Patient_ID', 'Final_Sepsis_Label']].drop_duplicates()['Final_Sepsis_Label'].value_counts()

# Ensure both labels (0 and 1) appear in the output
sepsis_distribution = sepsis_distribution.reindex([0, 1], fill_value=0)

print(sepsis_distribution)


# In[14]:


from sklearn.preprocessing import StandardScaler

# Normalize numeric columns
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Display the normalized data
print(df.head())


# In[15]:


# Convert Patient_ID to categorical
df['Patient_ID'] = df['Patient_ID'].astype('category').cat.codes

# Display the updated DataFrame
print(df.head())


# In[16]:


# Create a time index for each patient
df['time_idx'] = df.groupby('Patient_ID').cumcount()

# Display the updated DataFrame
print(df.head())


# In[17]:


df.tail()


# In[26]:


get_ipython().system('pip install pytorch-forecasting')


# In[18]:


from pytorch_forecasting import TimeSeriesDataSet

df['Patient_ID'] = df['Patient_ID'].astype(str)

# Define the TimeSeriesDataSet
max_encoder_length = 30  # Length of historical data
max_prediction_length = 10  # Length of prediction horizon

training = TimeSeriesDataSet(
    data=df,
    time_idx="time_idx",
    target="Final_Sepsis_Label",  # Use the final sepsis classification column
    group_ids=["Patient_ID"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    static_categoricals=["Patient_ID"],
    time_varying_known_reals=numeric_cols + ['time_idx'],
    time_varying_unknown_reals=["Final_Sepsis_Label"],  # Updated target column
    target_normalizer=None,  # No normalization for binary target
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

# Create a dataloader
train_dataloader = training.to_dataloader(train=True, batch_size=128, num_workers=0)


# In[28]:


get_ipython().system('pip install torch')


# In[19]:


from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import CrossEntropy  # Use CrossEntropy for binary classification
import torch

# Define the TFT model with CrossEntropy loss
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.01,  # Reduce learning rate for better convergence
    hidden_size=128,  # Increase hidden size for better feature learning
    attention_head_size=16,  # More attention heads improve pattern detection
    dropout=0.5,  # Increase dropout to prevent overfitting
    hidden_continuous_size=64,
    output_size=2,  # Binary classification
    loss=CrossEntropy(),  # Apply class weights
    reduce_on_plateau_patience=3,  # Reduce patience for faster learning
)

# Display the model architecture
print(tft)


# In[20]:


import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import CrossEntropy

# Calculate class weights (adjust manually if needed)
class_counts = torch.tensor([14568, 38641])  # Example: [No Sepsis, Sepsis] counts
class_weights = 1.0 / class_counts  # Inverse frequency
class_weights = class_weights / class_weights.sum()  # Normalize

# Define a LightningModule wrapper for TemporalFusionTransformer
class TFTLightningModule(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(x)
        y_hat = output.prediction
        loss = self.model.loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(x)
        y_hat = output.prediction
        val_loss = self.model.loss(y_hat, y)

        # Compute recall manually
        y_pred = torch.argmax(y_hat, dim=1)
        tp = ((y_pred == 1) & (y == 1)).sum().float()
        fn = ((y_pred == 0) & (y == 1)).sum().float()
        recall = tp / (tp + fn + 1e-6)  # Avoid division by zero

        self.log("val_loss", val_loss, prog_bar=True)
        self.log("recall", recall, prog_bar=True)  # Log recall metric
        return val_loss

    def configure_optimizers(self):
        return self.model.configure_optimizers()

# Wrap the TemporalFusionTransformer in the LightningModule
tft_model = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.01,  # Reduce learning rate for better convergence
    hidden_size=128,  # Increase hidden size for better feature learning
    attention_head_size=16,  # More attention heads improve pattern detection
    dropout=0.5,  # Increase dropout to prevent overfitting
    hidden_continuous_size=64,
    output_size=2,  # Binary classification
    loss=CrossEntropy(),  # Apply class weights
    reduce_on_plateau_patience=3,  # Reduce patience for faster learning
)

tft_lightning = TFTLightningModule(tft_model)

# Define early stopping callback
early_stop_callback = EarlyStopping(
    monitor="val_loss",
    min_delta=1e-4,
    patience=15,  # Increase patience for better training
    verbose=True,
    mode="min",
)

# Initialize the trainer
trainer = pl.Trainer(
    max_epochs=5,  # Increase epochs for better learning
    accelerator="cpu",  # Use GPU for faster training
    devices=1,
    gradient_clip_val=0.2,
    callbacks=[early_stop_callback],
)

# Create a validation dataloader
validation_dataloader = training.to_dataloader(train=False, batch_size=256, num_workers=4)

# Train the model with validation
trainer.fit(tft_lightning, train_dataloader, validation_dataloader)


# In[21]:


# Check class distribution
print(df['Final_Sepsis_Label'].value_counts())


# In[23]:


# Save the model checkpoint after training
trainer.save_checkpoint("sepsis_tft_model.ckpt")


# In[24]:


# Cell 1: Load the trained model
import torch

# Load the trained model
tft_loaded = TFTLightningModule.load_from_checkpoint("sepsis_tft_model.ckpt", model=tft)
# Put model in evaluation mode
tft_loaded.eval()


# In[25]:


# Cell 2: Create test dataloader
test_dataloader = training.to_dataloader(train=False, batch_size=128, num_workers=0)


# In[33]:


import torch
import numpy as np

def get_predictions(model, dataloader):
    """
    Get model predictions and actual labels from a dataloader.

    Handles cases where the target variable `y` might be a tuple.

    Args:
        model: The trained model.
        dataloader: The dataloader to use for prediction.

    Returns:
        A tuple containing predicted labels and actual labels.
    """
    predictions = []
    actuals = []

    with torch.no_grad():
        for batch in dataloader:
            # Check if batch is a tuple and extract x and y accordingly
            if isinstance(batch, tuple) and len(batch) == 2:
                x, y = batch
                # If y is a tuple, assume the first element is the target tensor
                if isinstance(y, tuple):
                    y = y[0]
            else:
                # If batch is not a tuple, assume it's a dictionary-like object
                x = batch
                y = batch['target']

            # Get model predictions
            output = model(x)
            y_hat = output.prediction

            # Convert predictions to probabilities and then to binary labels
            probabilities = torch.softmax(y_hat, dim=1)
            pred_labels = torch.argmax(probabilities, dim=1)

            # Store predictions and actuals
            predictions.extend(pred_labels.cpu().numpy())
            actuals.extend(y.cpu().numpy())

    return np.array(predictions), np.array(actuals)


# In[36]:


# Get predictions (without patient_ids since we can't reliably extract them)
y_pred, y_true = get_predictions(tft_loaded, test_dataloader)

# Reshape or select elements from y_true to make it 1-dimensional
# Here, I'm taking the first element of each row, adjust if needed
y_true = y_true[:, 0]  # Or another appropriate selection method

# Reshape y_pred to 1-dimensional using argmax to get the predicted class labels
y_pred = np.argmax(y_pred, axis=1)  # Assuming y_pred contains probabilities

# Create a DataFrame without patient_ids
results_df = pd.DataFrame({
    'Index': range(len(y_true)),
    'Actual': y_true,
    'Predicted': y_pred
})


# In[37]:


# Cell 5: Create confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-Sepsis', 'Sepsis'],
            yticklabels=['Non-Sepsis', 'Sepsis'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()


# In[38]:


# Cell 6: Print classification report
from sklearn.metrics import classification_report, recall_score, precision_score, f1_score

# Calculate metrics
recall = recall_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# Print classification report
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=['Non-Sepsis', 'Sepsis']))

print(f"Recall (Sensitivity): {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1 Score: {f1:.4f}")


# In[43]:


# Cell 7: Create an easy-to-understand prediction vs actual chart
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Create a DataFrame for predictions vs. actuals
results_df = pd.DataFrame({
    'Patient_ID': range(len(y_true)),
    'Actual': y_true,
    'Predicted': y_pred
})

# Count the number of actual and predicted cases
sepsis_counts = {
    'Actual': [sum(results_df['Actual'] == 0), sum(results_df['Actual'] == 1)],
    'Predicted': [sum(results_df['Predicted'] == 0), sum(results_df['Predicted'] == 1)]
}

# Create a DataFrame for the counts
counts_df = pd.DataFrame(sepsis_counts, index=['Non-Sepsis', 'Sepsis'])

# Create a bar chart for the counts
plt.figure(figsize=(10, 6))
counts_df.plot(kind='bar', color=['blue', 'red'], alpha=0.7)
plt.xlabel('Sepsis Status')
plt.ylabel('Number of Patients')
plt.title('Comparison of Actual vs Predicted Sepsis Cases')
plt.legend()
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('sepsis_counts_comparison.png')
plt.show()

# Create a pie chart to show proportion of correct predictions
correct_predictions = sum(results_df['Actual'] == results_df['Predicted'])
total_predictions = len(results_df)
accuracy = correct_predictions / total_predictions

plt.figure(figsize=(8, 8))
plt.pie([correct_predictions, total_predictions - correct_predictions],
        labels=['Correct Predictions', 'Incorrect Predictions'],
        autopct='%1.1f%%',
        colors=['green', 'red'],
        explode=(0.1, 0),
        shadow=True)
plt.title(f'Model Prediction Accuracy: {accuracy:.2%}')
plt.tight_layout()
plt.savefig('prediction_accuracy_pie.png')
plt.show()


