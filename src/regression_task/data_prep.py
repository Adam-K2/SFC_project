import torch
from torch.utils.data import Dataset

# Class to process regression dataset
class RegressionSequenceDataset(Dataset):
    def __init__(self, values, window=30):
        self.values = torch.tensor(values, dtype=torch.float32)
        self.window = window

    def __len__(self):
        return len(self.values) - self.window

    def __getitem__(self, idx):
        x = self.values[idx : idx + self.window]
        y = self.values[idx + self.window]
        return x, y

# Collate function
def collate_regression(batch):
    xs, ys = zip(*batch)
    xs = torch.stack(xs)
    ys = torch.stack(ys)
    return xs, ys