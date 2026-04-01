import torch, os, sys
sys.path.insert(0, os.path.dirname(__file__))

from config.config_mnist    import CONFIG_MNIST
from config.config_fashion  import CONFIG_FASHION
from config.config_medical  import CONFIG_MEDICAL
from config.config_mnist_v2   import CONFIG_MNIST_V2
from config.config_fashion_v2 import CONFIG_FASHION_V2
from config.config_medical_v2 import CONFIG_MEDICAL_V2
from model.lenet5    import LeNet5      # v1: paper gốc
from model.lenet5_v2 import LeNet5V2   # v2: cải tiến
from utils                  import get_dataloader
from train                  import train
from evaluate               import evaluate, plot_history

CLASS_NAMES = {
    "mnist":         [str(i) for i in range(10)],
    "fashion_mnist": ["T-shirt","Trouser","Pullover","Dress","Coat",
                      "Sandal","Shirt","Sneaker","Bag","Ankle boot"],
    "medical_mnist": ["AbdomenCT","BreastMRI","ChestCT",
                      "CXR","Hand","HeadCT"],
}

def run(config, version="v1"):
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset     = config["dataset"]
    class_names = CLASS_NAMES[dataset]

    print(f"\n{'='*60}")
    print(f"  Dataset : {dataset} [{version}]  |  Device: {device}")
    print(f"{'='*60}")

    train_loader, test_loader = get_dataloader(config)

    if version == "v1":
        model = LeNet5(     
            num_classes    = config["num_classes"],
            input_channels = config["input_channels"],
        ).to(device)
    else:
        model = LeNet5V2(
            num_classes    = config["num_classes"],
            input_channels = config["input_channels"],
            dropout        = config.get("dropout", 0.0),
        ).to(device)

    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {total_params:,}\n")

    ckpt_dir  = config.get("checkpoint_dir", "/content/drive/MyDrive/CV_NN/checkpoints")
    save_path = os.path.join(ckpt_dir, f"best_{dataset}_{version}.pth")

    history   = train(model, train_loader, config, device, save_path)

    model.load_state_dict(torch.load(save_path, map_location=device))
    evaluate(model, test_loader, device, class_names, f"{dataset}_{version}", save_dir=ckpt_dir)
    plot_history(history, f"{dataset}_{version}", save_dir=ckpt_dir)

if __name__ == "__main__":
    # ── Baseline ──────────────────────────
    # run(CONFIG_MNIST,   "v1")
    # run(CONFIG_FASHION, "v1")
    # run(CONFIG_MEDICAL, "v1")

    # ── Cải tiến ──────────────────────────
    run(CONFIG_MNIST_V2,   "v2")
    run(CONFIG_FASHION_V2, "v2")
    run(CONFIG_MEDICAL_V2, "v2")