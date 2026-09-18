from torch.utils.data import DataLoader
from datasets import load_dataset
from functools import partial

from data_prep import Vocab, IMDBDataset, collate_batch, collate_batch_mlp
from models_sentiment import *
from train_utils import *
from metrics_plots import *

# =========================================================
# 1. LOAD IMDB DATASET
# =========================================================
ds = load_dataset("stanfordnlp/imdb")

# Build vocabulary (min_freq=5 to reduce vocab size)
vocab = Vocab()
vocab.build_vocab([example['text'] for example in ds['train']], min_freq=5)

# Model hyperparameters
VOCAB_SIZE = len(vocab.itos)  
EMB_DIM = 128
HIDDEN = 128  
NUM_CLASSES = 2
PAD_IDX = 0
UNK_IDX = 1
MAX_SEQ_LEN_IMDB = 2500

collate_fn_mlp = partial(collate_batch_mlp, vocab_size=VOCAB_SIZE)

# =========================================================
# 2. CREATE DATASETS
# =========================================================
test_ds = ds["test"]
splits = test_ds.train_test_split(test_size=5000, shuffle=True, seed=42)

train_dataset = IMDBDataset(ds['train'], vocab)
val_dataset = IMDBDataset(splits["train"], vocab)
test_dataset = IMDBDataset(splits["test"], vocab)

# =========================================================
# 3. DATALOADERS
# =========================================================
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_batch)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_batch)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_batch)

# For MLP (BoW)
train_loader_mlp = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn_mlp)
val_loader_mlp = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn_mlp)
test_loader_mlp = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn_mlp)

# =========================================================
# 4. MODEL DEFINITIONS
# =========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

model_mlp = MLP(vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN, num_classes=NUM_CLASSES).to(device)

model_cnn = CNNClassifier(
    vocab_size=VOCAB_SIZE,
    emb_dim=EMB_DIM,
    num_classes=NUM_CLASSES,
    pad_idx=PAD_IDX
).to(device)

model_gru = GRUClassifier(
    vocab_size=VOCAB_SIZE,
    emb_dim=EMB_DIM,
    hidden_dim=HIDDEN,
    num_classes=NUM_CLASSES,
    pad_idx=PAD_IDX
).to(device)

model_transformer = TransformerEncoderClassifier(
    vocab_size=VOCAB_SIZE,
    emb_dim=EMB_DIM,
    hidden_dim=HIDDEN,
    num_classes=NUM_CLASSES,
    pad_idx=PAD_IDX,
    max_seq_len=MAX_SEQ_LEN_IMDB
).to(device)

# =========================================================
# 5. OUTPUT DIRECTORIES
# =========================================================
MODELS = "models_imdb"
PLOTS = "plots_imdb"
os.makedirs(MODELS, exist_ok=True)
os.makedirs(PLOTS, exist_ok=True)

# =========================================================
# 6. UNIFIED TRAINING FUNCTION
# =========================================================
def run_experiment(model_name, model, train_loader, valid_loader, epochs=10, lr=0.001):
    print(f"\n--- Running {model_name} Experiment ---")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_valid_acc = -1.0
    best_model_path = os.path.join(MODELS, f"{model_name.replace(' ', '_')}_best_model.pt")

    # Dictionary to store training progress history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        valid_acc, valid_loss = eval_model(model, valid_loader, criterion, device)

        print(f"Epoch {epoch+1}/{epochs}: Train loss={train_loss:.4f}, Valid loss={valid_loss:.4f}, Valid acc={valid_acc:.4f}")

        # Logging values to history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(valid_loss)
        history['val_acc'].append(valid_acc)

        # Model Saving Logic
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best model saved to {best_model_path} (Acc: {valid_acc:.4f})")

    print(f"Training complete. Best model saved at {best_model_path}")
    # Return the path AND the history
    return best_model_path, history

# =========================================================
# 7. TRAIN ALL MODELS
# =========================================================
# MLP
mlp_path, mlp_history = run_experiment("MLP", model_mlp,
    train_loader_mlp,
    val_loader_mlp,
    epochs=10
)
test_final_model("MLP", model_mlp, mlp_path, test_loader_mlp, device, PLOTS, NUM_CLASSES)
plot_training_curves(mlp_history, "MLP", PLOTS)

#CNN
cnn_path, cnn_history = run_experiment("Kim_CNN", model_cnn,
    train_loader,
    val_loader,
    epochs=10
)
test_final_model("Kim_CNN", model_cnn, cnn_path, test_loader, device, PLOTS, NUM_CLASSES)
plot_training_curves(cnn_history, "Kim_CNN", PLOTS)

#GRU
gru_path, gru_history = run_experiment("GRU", model_gru,
    train_loader,
    val_loader,
    epochs=10
)
test_final_model("GRU", model_gru, gru_path, test_loader, device, PLOTS, NUM_CLASSES)
plot_training_curves(gru_history, "GRU", PLOTS)

#Transformer
transformer_path, transformer_history = run_experiment("Transformer",
    model_transformer,
    train_loader,
    val_loader,
    epochs=10,
    lr=0.0001
)
test_final_model("Transformer", model_transformer, transformer_path, test_loader, device, PLOTS, NUM_CLASSES)
plot_training_curves(transformer_history, "Transformer", PLOTS)