# Imports
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pandas as pd
import kagglehub
from sklearn.preprocessing import MinMaxScaler

from data_prep import *
from models_regresion import *
from train_utils import *
from metrics_plots import *

# ---------------------------------------------------------
# 0. Kaggle Dataset Download
# ---------------------------------------------------------
# Download latest version of dataset
path = kagglehub.dataset_download("suprematism/daily-minimum-temperatures")
# Dataset filename inside the folder
FILE_NAME = "1_Daily_minimum_temps.csv"
csv_file_path = os.path.join(path, FILE_NAME)

# ---------------------------------------------------------
# 1. Load & Clean Data
# ---------------------------------------------------------
df = pd.read_csv(csv_file_path)

# Ensure temperature column is numeric (invalid entries -> NaN)
df['Temp'] = pd.to_numeric(df['Temp'], errors='coerce')

# Interpolate missing values
df['Temp'] = df['Temp'].interpolate()

# Extract temperature values as numpy array
values = df["Temp"].values

# ---------------------------------------------------------
# 2. Train/Val/Test Split
# ---------------------------------------------------------
N = len(values)
train_split = int(N * 0.8)
val_split = int(N * 0.9)

train_values = values[:train_split].reshape(-1, 1) 
val_values = values[train_split:val_split].reshape(-1, 1)
test_values = values[val_split:].reshape(-1, 1)

# ---------------------------------------------------------
# 3. Scaling
# ---------------------------------------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_values)

train_scaled = scaler.transform(train_values).flatten()
val_scaled = scaler.transform(val_values).flatten()
test_scaled = scaler.transform(test_values).flatten()

# ---------------------------------------------------------
# 4. Dataset & DataLoaders
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# 5. Model Definitions
# ---------------------------------------------------------
INPUT_CHANNELS = 1
HIDDEN_REG = 64

device = "cuda" if torch.cuda.is_available() else "cpu"

model_mlp = MLP(input_dim=WINDOW_SIZE, hidden_dim=HIDDEN_REG).to(device)
model_cnn = CNNRegressor(input_channels=INPUT_CHANNELS, hidden_dim=HIDDEN_REG).to(device)
model_gru = GRURegressor(input_dim=INPUT_CHANNELS, hidden_dim=HIDDEN_REG).to(device)
model_transformer = TransformerEncoderRegressor(input_dim=INPUT_CHANNELS, hidden_dim=HIDDEN_REG).to(device)

# ---------------------------------------------------------
# 6. Output Directories
# ---------------------------------------------------------
PLOTS = "plots_temperature"
MODELS = "models_temperature"
os.makedirs(PLOTS, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)

# ---------------------------------------------------------
# 7. Unified Training Function
# ---------------------------------------------------------
def run_experiment(model_name, model, train_loader, val_loader, plots_file, epochs=30, lr=0.001):
    print(f"\n--- Running {model_name} REGRESSION/Temperature Experiment ---")
    
    criterion = nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Paths and Tracking
    best_model_path = os.path.join(MODELS, f"best_{model_name.replace(' ', '_')}.pt")
    best_val_rmse = float('inf') 
    
    # History Lists for Plotting
    train_losses_history = []
    val_rmses_history = []

    for epoch in range(epochs):
        # Training Step
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device) 
        
        # Validation Step
        val_mse, val_rmse = eval_model(model, val_loader, device) 
        
        # Log History
        train_losses_history.append(train_loss)
        val_rmses_history.append(val_rmse)
        
        improvement = ""
        # Check for improvement and save the best model
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            # Save ONLY the model's parameters (state_dict)
            torch.save(model.state_dict(), best_model_path)
            improvement = f"(New Best! Saving to {best_model_path})" 

        print(f"Epoch {epoch+1}/{epochs}: Train MSE={train_loss:.4f}, Val RMSE={val_rmse:.4f}, Best Val RMSE={best_val_rmse:.4f} {improvement}")

    print(f"--- {model_name} Training Complete ---")
    
    # Plot the full training/validation history 
    plot_training_history(train_losses_history, val_rmses_history, model_name, plots_file)
    
    return best_model_path

# ---------------------------------------------------------
# 8. TRAIN ALL MODELS
# ---------------------------------------------------------
mlp_path = run_experiment("MLP Regression - temperatures", model_mlp, train_loader, val_loader, PLOTS)
cnn_path = run_experiment("CNN Regression - temperatures", model_cnn, train_loader, val_loader, PLOTS)
gru_path = run_experiment("GRU Regression - temperatures", model_gru, train_loader, val_loader, PLOTS)
transformer_path = run_experiment("Transformer Regression - temperatures", model_transformer, train_loader, val_loader, PLOTS)


# ---------------------------------------------------------
# 9. FINAL TEST EVALUATION
# ---------------------------------------------------------
print("\n\n" + "="*50)
print("STARTING FINAL EVALUATION ON TEST SET (Loading Best Weights)")
print("="*50 + "\n")

test_final_regression_model("MLP Regression - temperatures", model_mlp, test_loader, device, mlp_path, PLOTS)
test_final_regression_model("CNN Regression - temperatures", model_cnn, test_loader, device, cnn_path, PLOTS)
test_final_regression_model("GRU Regression - temperatures", model_gru, test_loader, device, gru_path, PLOTS)
test_final_regression_model("Transformer Regression - temperatures", model_transformer, test_loader, device, transformer_path, PLOTS)