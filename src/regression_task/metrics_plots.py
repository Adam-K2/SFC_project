import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
import os

# Collect Predictions from Regression Model
def get_regression_predictions(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs) 
            
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
            
    return np.array(all_preds), np.array(all_labels)

# Print Regression Metrics
def print_regression_metrics(labels, preds):
    mae = mean_absolute_error(labels, preds)
    r2 = r2_score(labels, preds)
    
    print("="*30)
    print(f"FINAL REGRESSION METRICS (Scaled Data)")
    print("="*30)
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R-squared Score (R2): {r2:.4f}")
    print("="*30)

# Time-Series Plot (True vs Predicted)
def plot_time_series_predictions(labels, preds, model_name, plots_file):
    fig = plt.figure(figsize=(12, 6))
    
    time_index = np.arange(len(labels))
    
    plt.plot(time_index, labels, label='True Temperature (Scaled)', color='blue')
    plt.plot(time_index, preds, label='Predicted Temperature (Scaled)', color='red', linestyle='--')
    
    plt.title(f"{model_name} - Time Series Prediction vs. True Values")
    plt.xlabel("Test Time Step (Days)")
    plt.ylabel("Scaled Temperature")
    plt.legend()
    plt.grid(True)
    
    filename = os.path.join(plots_file, f"{model_name.replace(' ', '_')}_timeseries_plot.png")
    plt.savefig(filename)
    print(f"Plot saved: {filename}")
    plt.close(fig)

# Training History Plot (MSE & RMSE)
def plot_training_history(train_losses, val_rmses, model_name, plots_file):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(train_losses, label='Training MSE', color='blue', linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training MSE", color='blue')
    ax.tick_params(axis='y', labelcolor='blue')
    
    ax2 = ax.twinx()  
    ax2.plot(val_rmses, label='Validation RMSE', color='red', linestyle='--', linewidth=2)
    ax2.set_ylabel("Validation RMSE", color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    ax.set_title(f"{model_name} - Training History")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    
    ax.grid(True)
    
    filename = os.path.join(plots_file, f"{model_name.replace(' ', '_')}_history_plot.png")
    plt.savefig(filename)
    print(f"Plot saved: {filename}")
    plt.close(fig)

# Scatter Plot (True vs Predicted)
def plot_true_vs_pred_scatter(labels, preds, model_name, plots_file):
    fig = plt.figure(figsize=(8, 8))
    
    min_val = min(labels.min(), preds.min())
    max_val = max(labels.max(), preds.max())
    
    # Scatter Plot
    plt.scatter(labels, preds, alpha=0.5, color='darkgreen', label='Predictions')
    
    # Ideal line (y=x)
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Fit (y=x)') 
    
    plt.title(f"{model_name} - True vs. Predicted Values (Test Set)")
    plt.xlabel("True Scaled Temperature")
    plt.ylabel("Predicted Scaled Temperature")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    filename = os.path.join(plots_file, f"{model_name.replace(' ', '_')}_scatter_plot.png")
    plt.savefig(filename)
    print(f"Plot saved: {filename}")
    plt.close(fig)

# Final Test Evaluation Using Best Model Weights
def test_final_regression_model(model_name, model, test_loader, device, best_model_path, plots_file):
    print(f"\n--- Running FINAL TEST for '{model_name}' ---")
    
    # Load best weights
    try:
        state_dict = torch.load(best_model_path, map_location=device)
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        print(f"ERROR: Best model path not found at {best_model_path}. Proceeding with current weights.")
    
    # Predictions 
    all_preds_scaled, all_labels_scaled = get_regression_predictions(model, test_loader, device)
    
    # Metrics
    print_regression_metrics(all_labels_scaled, all_preds_scaled)
    
    # Plots
    plot_time_series_predictions(all_labels_scaled, all_preds_scaled, model_name, plots_file)
    plot_true_vs_pred_scatter(all_labels_scaled, all_preds_scaled, model_name, plots_file)
