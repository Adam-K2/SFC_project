from torch.utils.data import DataLoader
from datasets import load_dataset
from functools import partial
import argparse

from sentiment_classification.data_prep import *
from sentiment_classification.models_sentiment import *
from sentiment_classification.train_utils import *
from sentiment_classification.metrics_plots import *

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=["imdb", "sst5"], default="imdb")
parser.add_argument("--model", choices=["mlp", "cnn", "gru", "transformer"], default="mlp")
args = parser.parse_args()

# Load given dataset from arguments
if args.dataset == "imdb":
    ds = load_dataset("stanfordnlp/imdb")
    NUM_CLASSES = 2
    MAX_SEQ_LEN = 2500
    vocab = Vocab()
    vocab.build_vocab([example['text'] for example in ds['train']], min_freq=5)
else:
    ds = load_dataset("SetFit/sst5")
    NUM_CLASSES = 5
    MAX_SEQ_LEN = 1000
    vocab = Vocab()
    vocab.build_vocab([example['text'] for example in ds['train']])

# Model hyperparameters
VOCAB_SIZE = len(vocab.itos)  
EMB_DIM = 128
HIDDEN = 128  
PAD_IDX = 0
UNK_IDX = 1

collate_fn_mlp = partial(collate_batch_mlp, vocab_size=VOCAB_SIZE)

# Process datasets
if (NUM_CLASSES == 2):
    test_ds = ds["test"]
    splits = test_ds.train_test_split(test_size=5000, shuffle=True, seed=42)
    train_dataset = IMDBDataset(ds['train'], vocab)
    val_dataset = IMDBDataset(splits["train"], vocab)
    test_dataset = IMDBDataset(splits["test"], vocab)
    MODELS_DIR = "models_imdb"
    PLOTS = "plots_imdb_confusion_matrix"
else:
    train_dataset = SST5Dataset(ds['train'], vocab)
    val_dataset = SST5Dataset(ds['validation'], vocab)
    test_dataset = SST5Dataset(ds['test'], vocab)
    MODELS_DIR = "models_sst5"
    PLOTS = "plots_sst5_confusion_matrix"

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_batch)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_batch)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_batch)

# For MLP (BoW)
train_loader_mlp = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn_mlp)
val_loader_mlp = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn_mlp)
test_loader_mlp = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn_mlp)

# Model definitions
device = "cuda" if torch.cuda.is_available() else "cpu"

if args.model == "mlp":
    model = MLP(vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN, num_classes=NUM_CLASSES).to(device)
    model_name_str = "MLP"
    loader_to_use = test_loader_mlp
    weight_filename = "MLP_best_model.pt"

elif args.model == "cnn":
    model = CNNClassifier(
        vocab_size=VOCAB_SIZE,
        emb_dim=EMB_DIM,
        num_classes=NUM_CLASSES,
        pad_idx=PAD_IDX
    ).to(device)
    model_name_str = "Kim_CNN"
    loader_to_use = test_loader
    weight_filename = "Kim_CNN_best_model.pt"

elif args.model == "gru":
    model = GRUClassifier(
        vocab_size=VOCAB_SIZE,
        emb_dim=EMB_DIM,
        hidden_dim=HIDDEN,
        num_classes=NUM_CLASSES,
        pad_idx=PAD_IDX
    ).to(device)
    model_name_str = "GRU"
    loader_to_use = test_loader
    weight_filename = "GRU_best_model.pt"

elif args.model == "transformer":
    model = TransformerEncoderClassifier(
        vocab_size=VOCAB_SIZE,
        emb_dim=EMB_DIM,
        hidden_dim=HIDDEN,
        num_classes=NUM_CLASSES,
        pad_idx=PAD_IDX,
        max_seq_len=MAX_SEQ_LEN
    ).to(device)
    model_name_str = "Transformer"
    loader_to_use = test_loader
    weight_filename = "Transformer_best_model.pt"

else:
    raise ValueError(f"Model '{args.model}' not recognized in logic.")

# Output director for confusion matrix
os.makedirs(PLOTS, exist_ok=True)

# Test the selected model
BASE_MODEL_DIR = "sentiment_classification"

# Create the full path to the specific model's weights
model_dir_path = os.path.join(BASE_MODEL_DIR, MODELS_DIR)
model_weight_path = os.path.join(model_dir_path, weight_filename)

print(f"--- Testing {model_name_str} on {args.dataset} dataset ---")
print(f"Loading weights from: {model_weight_path}")

# Run the test
test_final_model(
    model_name_str,
    model,
    model_weight_path,
    loader_to_use,
    device,
    PLOTS,
    NUM_CLASSES
)