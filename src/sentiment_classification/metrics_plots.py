import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import os

# Collect Predictions from Classification Model
def get_predictions(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for texts, labels in loader:
            texts, labels = texts.to(device), labels.to(device)

            outputs = model(texts)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels

# Print Classification Metrics
def print_classification_report(labels, preds, class_names):
    acc = accuracy_score(labels, preds)

    # Generates P, R, F1, and support for each class
    report = classification_report(labels, preds, target_names=class_names)

    print("="*30)
    print(f"FINAL TEST METRICS")
    print("="*30)
    print(f"Overall Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(report)
    print("="*30)

# Plots training/validation loss and validation accuracy over epochs.
def plot_training_curves(history, model_name, plots_file):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Loss Curve
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title(f"{model_name} - Loss Curves")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    # Accuracy Curve
    ax2.plot(history['val_acc'], label='Validation Accuracy', color='green')
    ax2.set_title(f"{model_name} - Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    plt.tight_layout()

    # Save figure into plots_file
    filename = os.path.join(plots_file, f"{model_name.replace(' ', '_')}_training_curves.png")
    plt.savefig(filename)
    print(f"Plot saved: {filename}")
    plt.close(fig)

# Plots confusion matrix heatmap and saves the figure.
def plot_confusion_matrix(labels, preds, model_name, plots_file, class_names):
    cm = confusion_matrix(labels, preds)

    fig = plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    # Save figure into plots_file
    filename = os.path.join(plots_file, f"{model_name.replace(' ', '_')}_confusion_matrix.png")
    plt.savefig(filename)
    print(f"Plot saved: {filename}")
    plt.close(fig)

# Final Test Evaluation Using Best Model Weights
def test_final_model(model_name, model, best_model_path, test_loader, device, plots_file, cls_num):
    print(f"\n--- Loading best '{model_name}' for FINAL TEST ---")

    # Load best weights
    try:
        state_dict = torch.load(best_model_path, map_location=device)
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        print(f"ERROR: Best model path not found at {best_model_path}.")

    # Predictions
    all_preds, all_labels = get_predictions(model, test_loader, device)

    # Class labels
    if cls_num == 2:
        cls = ['Negative', 'Positive']
    else:
        cls = ['V. Negative', 'Negative', 'Neutral', 'Positive', 'V. Positive']

    # # Metrics (Accuracy, P, R, F1)
    print_classification_report(all_labels, all_preds, cls)

    # Confusion Matrix
    plot_confusion_matrix(all_labels, all_preds, model_name, plots_file, cls)