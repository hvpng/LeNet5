# train.py

import torch
import torch.optim as optim
from model.lenet5 import MAPLoss_RBF, MSELoss_RBF

def train(model, train_loader, config, device, save_path="best_model.pth"):

    use_map   = config.get("use_map_loss", True)
    criterion = MAPLoss_RBF(j=0.0) if use_map else MSELoss_RBF()

    opt_name = config.get("optimizer", "sgd")

    if opt_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=config["learning_rate"],
            momentum=0.0
        )

    lr_schedule = config.get("lr_schedule", {})
    history = {"train_loss": [], "train_acc": []}

    for epoch in range(config["epochs"]):

        # LR schedule
        if lr_schedule and epoch in lr_schedule:
            new_lr = lr_schedule[epoch]
            for g in optimizer.param_groups:
                g['lr'] = new_lr
            print(f"  → LR updated: {new_lr}")

        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            rbf_out = model(X)
            loss    = criterion(rbf_out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X.size(0)
            preds   = rbf_out.argmin(dim=1)
            correct += (preds == y).sum().item()
            total   += X.size(0)

        train_loss = total_loss / total
        train_acc  = correct   / total
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        print(f"Epoch [{epoch+1:02d}/{config['epochs']}] | "
              f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc*100:.2f}%")

    torch.save(model.state_dict(), save_path)
    print(f"\n Model saved: {save_path}")
    return history