import torch
import torch.nn.functional as F

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs) 
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_model(model, loader, device):
    model.eval()
    mse_total = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # Calculate MSE for the batch
            mse_batch = F.mse_loss(outputs, targets, reduction='sum').item()
            mse_total += mse_batch
            total_samples += targets.size(0)
            
    final_mse = mse_total / total_samples
    final_rmse = final_mse**0.5
    return final_mse, final_rmse