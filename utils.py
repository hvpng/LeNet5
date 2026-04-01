import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

# ── Normalize chuẩn cho từng dataset ──────────────────────
NORMALIZE = {
    "mnist":         transforms.Normalize((0.1307,), (0.3081,)),
    "fashion_mnist": transforms.Normalize((0.2860,), (0.3530,)),
    "medical_mnist": transforms.Normalize((0.3583,), (0.2822,)),
}

def get_dataloader(config):
    dataset_name = config["dataset"]
    data_path    = config["data_path"]
    batch_size   = config["batch_size"]
    augmentation = config.get("augmentation", False)
    aug_type     = config.get("aug_type", "general")

    norm = NORMALIZE[dataset_name]

    # ── Base transform — dùng cho test, KHÔNG augment ──────
    base_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        norm
    ])

    # ── MNIST augmentation ─────────────────────────────────
    mnist_aug_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((32, 32)),
        transforms.RandomRotation(10),
        transforms.RandomApply([
            transforms.ElasticTransform(alpha=20.0, sigma=3.0)
        ], p=0.7),
        transforms.ToTensor(),
        norm
    ])

    # ── Fashion augmentation ───────────────────────────────
    fashion_aug_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((32, 32)),
        transforms.RandomRotation(degrees=8),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.08, 0.08),
            scale=(0.95, 1.05),
        ),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3)
        ], p=0.3),
        transforms.ToTensor(),
        norm
    ])

    # ── Medical augmentation ───────────────────────────────
    # Flip ngang OK vì ảnh y tế có tính đối xứng
    medical_aug_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.08, 0.08),
            scale=(0.95, 1.05),
        ),
        transforms.ToTensor(),
        norm
    ])

    # ── Chọn transform cho train ───────────────────────────
    if not augmentation:
        train_transform = base_transform
    elif aug_type == "mnist":
        train_transform = mnist_aug_transform
    elif aug_type == "fashion":
        train_transform = fashion_aug_transform
    elif aug_type == "medical":
        train_transform = medical_aug_transform
    else:
        train_transform = base_transform

    # ── Load dataset ───────────────────────────────────────
    if dataset_name == "mnist":
        train_data = datasets.MNIST(root=data_path, train=True,
                                    download=False, transform=train_transform)
        test_data  = datasets.MNIST(root=data_path, train=False,
                                    download=False, transform=base_transform)

    elif dataset_name == "fashion_mnist":
        train_data = datasets.FashionMNIST(root=data_path, train=True,
                                           download=False, transform=train_transform)
        test_data  = datasets.FashionMNIST(root=data_path, train=False,
                                           download=False, transform=base_transform)

    elif dataset_name == "medical_mnist":
        full_train = ImageFolder(root=data_path, transform=train_transform)
        full_test  = ImageFolder(root=data_path, transform=base_transform)

        total         = len(full_train)
        train_size    = int(0.8 * total)
        test_size     = total - train_size
        generator     = torch.Generator().manual_seed(42)

        indices       = torch.randperm(total, generator=generator).tolist()
        train_indices = indices[:train_size]
        test_indices  = indices[train_size:]

        train_data = Subset(full_train, train_indices)
        test_data  = Subset(full_test,  test_indices)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_data,  batch_size=batch_size, shuffle=False)

    return train_loader, test_loader