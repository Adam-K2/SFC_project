import torch
import torch.nn as nn
import torch.nn.functional as F

WINDOW_SIZE = 30

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1) 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x shape: (batch_size, window)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out).squeeze(1) 
        return out

class CNNRegressor(nn.Module):
    def __init__(self, input_channels, hidden_dim,
                 filter_sizes=[3, 4, 5], num_filters=100, dropout=0.5):
        super().__init__()

        self.convs = nn.ModuleList([
            nn.Conv1d(input_channels, num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])

        conv_out_dim = len(filter_sizes) * num_filters

        self.fc1 = nn.Linear(conv_out_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch, window)
        x = x.unsqueeze(1)   # → (batch, 1, window)

        conv_outputs = [F.relu(conv(x)) for conv in self.convs]
        pooled = [F.max_pool1d(c, c.size(2)).squeeze(2) for c in conv_outputs]

        # Concatenate filter outputs
        features = torch.cat(pooled, dim=1)

        # Dense → Output
        h = F.relu(self.fc1(self.dropout(features)))
        out = self.fc2(h).squeeze(1)

        return out

class GRURegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=1,
            bidirectional=False,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x: (batch_size, window) -> (batch_size, window, 1)
        x = x.unsqueeze(-1)

        _, hidden = self.gru(x)
        
        last_hidden = hidden[-1, :, :] # (batch_size, hidden_dim)
        
        out = self.dropout(last_hidden)
        out = self.fc(out).squeeze(1)
        return out
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=WINDOW_SIZE):
        super().__init__()
        self.pe = nn.Parameter(torch.empty(max_len, d_model)) 
        nn.init.uniform_(self.pe, -0.02, 0.02) # Initialize weights

    def forward(self, x):
        # x shape: (batch_size, window, hidden_dim)
        return x + self.pe[:x.size(1), :].unsqueeze(0)
    
class TransformerEncoderRegressor(nn.Module):
    def __init__(
        self, 
        input_dim,
        hidden_dim, 
        num_layers=2, 
        nhead=8, 
        dropout=0.3
    ):
        super().__init__()
        
        ff_dim = hidden_dim * 4
        
        # Projection
        self.input_projection = nn.Linear(input_dim, hidden_dim) 
        self.dropout_layer = nn.Dropout(dropout)
        
        # Positional encoder
        self.pos_encoder = PositionalEncoding(d_model=hidden_dim, max_len=WINDOW_SIZE)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=nhead, 
            dim_feedforward=ff_dim, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Regression Head
        self.fc = nn.Linear(hidden_dim, 1) 

    def forward(self, x):
        # x: (batch_size, window) -> (batch_size, window, 1)
        x = x.unsqueeze(-1) 
        
        embedded = self.input_projection(x)
        embedded = self.dropout_layer(embedded)
        embedded = self.pos_encoder(embedded) 

        output = self.transformer_encoder(embedded)

        # Average pooling over time dimension
        final_output = output.mean(dim=1) 

        out = self.fc(final_output).squeeze(1)
        return out