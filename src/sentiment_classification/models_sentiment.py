import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(vocab_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5) 
    
    def forward(self, x):
        # x shape: (batch_size, vocab_size)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_classes, pad_idx, filter_sizes=[3, 4, 5], num_filters=100, dropout=0.5):
        super().__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        
        # Convolution layers for each filter size
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=emb_dim, out_channels=num_filters, kernel_size=fs) 
            for fs in filter_sizes
        ])
        
        # Fully connected classification layer
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embedding(x)                
        emb = emb.permute(0, 2, 1)

        # Apply convolution + ReLU
        conv_outputs = [F.relu(conv(emb)) for conv in self.convs]
        
        # Global max pooling for each feature map
        pooled_outputs = [F.max_pool1d(conv_out, conv_out.shape[2]).squeeze(2) 
                          for conv_out in conv_outputs]
        
        # Concatenate all pooled features
        cat = torch.cat(pooled_outputs, dim=1)
        cat = self.dropout(cat)
        out = self.fc(cat)
        
        return out

# GRU
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_classes, pad_idx):
        super().__init__()
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        # GRU layer
        self.gru = nn.GRU(
            input_size=emb_dim, 
            hidden_size=hidden_dim, 
            num_layers=1,             
            bidirectional=False,      
            batch_first=True
        )
        
        # Classification layer
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, emb_dim)
        _, hidden = self.gru(embedded)
        
        # Use last hidden state → (batch_size, hidden_dim)
        last_hidden = hidden[-1, :, :]
        
        out = self.dropout(last_hidden)
        out = self.fc(out)
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim, max_len=1000):
        super().__init__()

        pe = torch.zeros(max_len, emb_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / emb_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0)) # (1, max_len, emb_dim)

    def forward(self, x):
        # x is (batch_size, seq_len, emb_dim)
        x = x + self.pe[:, :x.size(1), :]
        return x
    
class TransformerEncoderClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim,
        hidden_dim,
        num_classes,
        pad_idx,
        max_seq_len=1000,
        num_layers=2,
        nhead=4,
        dropout=0.5
    ):
        super().__init__()

        ff_dim = hidden_dim * 4

        # Embedding + positional encoding
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(emb_dim, max_len = max_seq_len)
        self.dropout = nn.Dropout(dropout)

        # Transformer Encoder Stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.fc = nn.Linear(emb_dim, num_classes)
        self.pad_idx = pad_idx

    def forward(self, x):
        # x: (batch_size, seq_len)

        # Embedding & positional encoding
        embedded = self.embedding(x)  # (batch_size, seq_len, emb_dim)
        embedded = self.pos_encoder(embedded)
        embedded = self.dropout(embedded)

        # Mask for padding tokens
        src_key_padding_mask = (x == self.pad_idx)

        # Transformer encoder output
        output = self.transformer_encoder(
            embedded,
            src_key_padding_mask=src_key_padding_mask
        )

        # Sequence representation: mean pooling
        cls_output = output.mean(dim=1) # (batch_size, emb_dim)

        out = self.fc(cls_output)
        return out