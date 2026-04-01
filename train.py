import torch
import torch.optim as optim
from model.lenet5 import MAPLoss_RBF, MSELoss_RBF
import os

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

    lr_schedule  = config.get("lr_schedule", {})
    
    # ── Checkpoint settings ───────────────────────────────
    ckpt_dir     = config.get("checkpoint_dir", "/content/drive/MyDrive/LeNet5/checkpoints")
    ckpt_every   = config.get("checkpoint_every", 1)   # lưu mỗi N epochs
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Resume từ checkpoint nếu có ───────────────────────
    start_epoch = 0
    best_acc    = 0.0
    history     = {"train_loss": [], "train_acc": []}

    resume_path = os.path.join(ckpt_dir, f"latest_{os.path.basename(save_path)}")
    if os.path.exists(resume_path):
        print(f"📂 Resume từ checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        history     = ckpt["history"]
        best_acc    = ckpt.get("best_acc", 0.0)
        print(f"   → Tiếp tục từ epoch {start_epoch + 1}")

    # ── Training loop ─────────────────────────────────────
    for epoch in range(start_epoch, config["epochs"]):

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

        # ── Lưu checkpoint định kỳ lên Drive ──────────────
        if (epoch + 1) % ckpt_every == 0:
            ckpt_data = {
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "history":         history,
                "best_acc":        best_acc,
                "config":          config,
            }
            # Latest checkpoint (ghi đè)
            torch.save(ckpt_data, resume_path)
            # Epoch checkpoint (giữ lại)
            epoch_path = os.path.join(
                ckpt_dir,
                f"epoch{epoch+1:02d}_{os.path.basename(save_path)}"
            )
            torch.save(ckpt_data, epoch_path)
            print(f"  💾 Checkpoint saved: epoch {epoch+1}")

        # ── Lưu best model ─────────────────────────────────
        if train_acc > best_acc:
            best_acc = train_acc
            best_path = os.path.join(ckpt_dir, f"best_{os.path.basename(save_path)}")
            torch.save(model.state_dict(), best_path)
            print(f"  ⭐ Best model updated: {train_acc*100:.2f}%")

    # Lưu model cuối lên Drive
    final_path = os.path.join(ckpt_dir, os.path.basename(save_path))
    torch.save(model.state_dict(), final_path)
    print(f"\n✅ Model saved: {final_path}")
    return history