import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def evaluate(model, test_loader, device, class_names, dataset_name):
    """
    RBF output: argmin (distance nhỏ nhất = closest class).
    Khác với softmax (argmax).
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X, y in test_loader:
            X       = X.to(device)
            rbf_out = model(X)
            preds   = rbf_out.argmin(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = (all_preds == all_labels).mean()
    print(f"\n✅ Test Accuracy [{dataset_name}]: {acc*100:.2f}%")
    print(classification_report(all_labels, all_preds,
                                target_names=class_names, digits=4))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix — {dataset_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{dataset_name}.png", dpi=150)
    plt.show()
    return all_preds, all_labels


def plot_history(history, dataset_name):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history["train_loss"], label="Train Loss (MAP)", marker="o")
    axes[0].set_title(f"Loss — {dataset_name}")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MAP Loss")
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot([a*100 for a in history["train_acc"]], label="Train Acc", marker="o")
    axes[1].set_title(f"Accuracy — {dataset_name}")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend(); axes[1].grid(True)

    plt.suptitle(dataset_name, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"training_curve_{dataset_name}.png", dpi=150)
    plt.show()