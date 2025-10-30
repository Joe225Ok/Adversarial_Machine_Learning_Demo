import torch
import torch.nn as nn
import torch.optim as optim
import os

def train_model(model, train_loader, val_loader, epochs=3, lr=0.001, device="cpu", ckpt_path=None):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Check if checkpoint exists
    if ckpt_path and os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"[train] Loaded checkpoint from {ckpt_path}, skipping training")
        return model

    print("[train] Starting training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"[train] Epoch {epoch+1}/{epochs} - Loss: {running_loss / len(train_loader):.4f}")

        # Optional: validation accuracy after each epoch
        val_acc = evaluate(model, val_loader, device)
        print(f"[train] Validation Accuracy: {val_acc:.2f}%")

    if ckpt_path:
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
        print(f"[train] Model checkpoint saved at {ckpt_path}")

    return model

def evaluate(model, dataloader, device="cpu"):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total
