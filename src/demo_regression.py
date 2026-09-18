# Imports
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pandas as pd
import kagglehub
from sklearn.preprocessing import MinMaxScaler
import argparse

from regression_task.data_prep import *
from regression_task.models_regresion import *
from regression_task.train_utils import *
from regression_task.metrics_plots import *

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=["temperature", "energy"], default="temperature")
parser.add_argument("--model", choices=["mlp", "cnn", "gru", "transformer"], default="mlp")
args = parser.parse_args()

# Load given dataset from arguments
if args.dataset == "energy":
    path = kagglehub.dataset_download("uciml/electric-power-consumption-data-set")
    FILE_NAME = "household_power_consumption.txt"
    csv_file_path = os.path.join(path, FILE_NAME)
    df = pd.read_csv(
        csv_file_path, 
        sep=';', 
        low_memory=False, 
        na_values=['?'],
        usecols=['Date', 'Time', 'Global_active_power'] 
    )
    combined_series = df['Date'] + ' ' + df['Time']
    dt_index = pd.to_datetime(
        combined_series, 
        format='%d/%m/%Y %H:%M:%S',
        errors='coerce'
    )
    df.index = dt_index 
    df.drop(columns=['Date', 'Time'], inplace=True) 
    df_target = df[['Global_active_power']].copy()
    df_target.loc[:, 'Global_active_power'] = pd.to_numeric(
        df_target['Global_active_power'], errors='coerce'
    )
    df_target.loc[:, 'Global_active_power'] = df_target['Global_active_power'].interpolate()
    df_hourly = df_target.resample('h').mean()
    values = df_hourly["Global_active_power"].values

    if np.isnan(values).any():
        mean_val = np.nanmean(values)
        values[np.isnan(values)] = mean_val
    
    PLOTS = "plots_energy"
    MODELS_DIR = "models_energy"
else:
    path = kagglehub.dataset_download("suprematism/daily-minimum-temperatures")
    FILE_NAME = "1_Daily_minimum_temps.csv"
    csv_file_path = os.path.join(path, FILE_NAME)
    df = pd.read_csv(csv_file_path)
    df['Temp'] = pd.to_numeric(df['Temp'], errors='coerce')
    df['Temp'] = df['Temp'].interpolate()
    values = df["Temp"].values
    PLOTS = "plots_temperature"
    MODELS_DIR = "models_temperature"

# Train/Val/Test Split
N = len(values)
train_split = int(N * 0.8) 
val_split = int(N * 0.9) 

train_values = values[:train_split].reshape(-1, 1) 
val_values = values[train_split:val_split].reshape(-1, 1)
test_values = values[val_split:].reshape(-1, 1)

# Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_values) 

train_scaled = scaler.transform(train_values).flatten()
val_scaled = scaler.transform(val_values).flatten()
test_scaled = scaler.transform(test_values).flatten()

WINDOW_SIZE = 30
# Create Datasets
train_dataset = RegressionSequenceDataset(train_scaled, window=WINDOW_SIZE)
val_dataset = RegressionSequenceDataset(val_scaled, window=WINDOW_SIZE)
test_dataset = RegressionSequenceDataset(test_scaled, window=WINDOW_SIZE)

# Create DataLoaders
train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, collate_fn=collate_regression
)
val_loader = DataLoader(
    val_dataset, batch_size=32, shuffle=False, collate_fn=collate_regression
)
test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, collate_fn=collate_regression
)

# Model Definitions
device = "cuda" if torch.cuda.is_available() else "cpu"

INPUT_CHANNELS = 1
HIDDEN = 64

if args.model == "mlp":
    model = MLP(input_dim=WINDOW_SIZE, hidden_dim=HIDDEN).to(device)
    model_name = "MLP"
    weights_filename = "best_MLP_Regression_-_energy.pt"

elif args.model == "cnn":
    model = CNNRegressor(input_channels=INPUT_CHANNELS, hidden_dim=HIDDEN).to(device)
    model_name = "CNN"

elif args.model == "gru":
    model = GRURegressor(input_dim=INPUT_CHANNELS, hidden_dim=HIDDEN).to(device)
    model_name = "GRU"

elif args.model == "transformer":
    model = TransformerEncoderRegressor(input_dim=INPUT_CHANNELS, hidden_dim=HIDDEN).to(device)
    model_name = "Transformer"

else:
    raise ValueError("Unknown model type")

# Test the selected model
BASE_MODEL_DIR = "regression_task"

# Create the full path to the specific model's weights
model_dir_path = os.path.join(BASE_MODEL_DIR, MODELS_DIR)

weights_files = [f for f in os.listdir(model_dir_path) if f.endswith(".pt")]
weights_filename_basename = None

for f in weights_files:
    if f.endswith(".pt") and model_name in f and args.dataset in f:
        weights_filename_basename = f
        break 

if weights_filename_basename is None:
    raise FileNotFoundError(f"No weights file found for {model_name} on {args.dataset} dataset in {MODELS_DIR}")

weights_filename = os.path.join(
    model_dir_path,
    weights_filename_basename
)

# Output Directories
PLOTS = "plots_energy"
os.makedirs(PLOTS, exist_ok=True)

# Final test set evalaution
print("\n\n" + "="*50)
print("STARTING FINAL EVALUATION ON TEST SET (Loading Best Weights)")
print("="*50)

test_final_regression_model(model_name, model, test_loader, device, weights_filename, PLOTS)
