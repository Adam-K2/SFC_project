import torch

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for texts, labels in loader:
        texts, labels = texts.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_model(model, loader, criterion, device):
    model.eval()
    correct, total = 0, 0
    total_loss = 0.0
    with torch.no_grad():
        for texts, labels in loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)

            # Calculate validation loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Calculate validation accuracy
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    # Return average accuracy AND average loss
    return correct / total, total_loss / len(loader)