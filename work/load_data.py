import os
import pandas as pd

# Set the correct data folder path
data_folder = 'datasets/LM-training-datasets'

# Load the datasets
train_df = pd.read_csv(os.path.join(data_folder, 'train.csv'))
validation_df = pd.read_csv(os.path.join(data_folder, 'validation.csv'))
test_df = pd.read_csv(os.path.join(data_folder, 'test.csv'))

# Print basic information about the datasets
print("\nDataset shapes:")
print(f"Train set: {train_df.shape}")
print(f"Validation set: {validation_df.shape}")
print(f"Test set: {test_df.shape}")
