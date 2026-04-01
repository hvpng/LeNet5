import torch
import torch.nn as nn
import torch.optim as optim

def train(model, train_loader, config, device, save_path="best_model.pth"):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    history = {"train_loss": [], "train_acc": []}

    for epoch in range(config["epochs"]):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out  = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X.size(0)
            correct    += (out.argmax(1) == y).sum().item()
            total      += X.size(0)

        train_loss = total_loss / total
        train_acc  = correct   / total

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        print(f"Epoch [{epoch+1:02d}/{config['epochs']}] | "
              f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc*100:.2f}%")

        scheduler.step()

    torch.save(model.state_dict(), save_path)
    print(f"\n Model saved: {save_path}")
    return history