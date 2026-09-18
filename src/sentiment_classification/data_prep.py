import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import re

# Constants
PAD_IDX = 0
UNK_IDX = 1

# Basic English Tokenizer
def tokenizer(text):
    text = text.lower()
    tokens = re.findall(r"\b[\w']+\b", text)
    return tokens

# Vocabulary Class
class Vocab:
    def __init__(self, specials=["<pad>", "<unk>"]):
        self.itos = list(specials)
        self.stoi = {tok: idx for idx, tok in enumerate(self.itos)}
    
    # Builds vocabulary from a list of texts.
    def build_vocab(self, texts, min_freq=1):
        freq = {}
        for text in texts:
            for token in tokenizer(text):
                freq[token] = freq.get(token, 0) + 1
        for token, count in freq.items():
            if count >= min_freq:
                self.stoi[token] = len(self.itos)
                self.itos.append(token)
    
    def __call__(self, tokens):
        return [self.stoi.get(tok, self.stoi["<unk>"]) for tok in tokens]

# Dataset Classes
class IMDBDataset(Dataset):
    def __init__(self, dataset, vocab):
        self.dataset = dataset
        self.vocab = vocab
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        text = self.dataset[idx]['text']
        label = self.dataset[idx]['label']
        text_ids = torch.tensor(self.vocab(tokenizer(text)), dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)
        return text_ids, label

class SST5Dataset(Dataset):
    def __init__(self, dataset, vocab):
        self.dataset = dataset
        self.vocab = vocab
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        text = self.dataset[idx]['text']
        label = self.dataset[idx]['label']
        text_ids = torch.tensor(self.vocab(tokenizer(text)), dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)
        return text_ids, label

# Collate Functions
def collate_batch(batch):
    texts, labels = zip(*batch)
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=PAD_IDX)
    labels = torch.stack(labels)
    return texts_padded, labels

def collate_batch_mlp(batch, vocab_size):
    texts, labels = zip(*batch)
    labels = torch.stack(labels)
    
    batch_bow = []
    for text_ids in texts:
        bow_vec = torch.bincount(text_ids, minlength=vocab_size)
        batch_bow.append(bow_vec)
        
    texts_bow = torch.stack(batch_bow).float() 
    
    return texts_bow, labels
