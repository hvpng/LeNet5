CONFIG_MEDICAL_V2 = {
    "dataset": "medical_mnist",
    "data_path": "/content/Data/MNIST Medical",
    "input_channels": 1,
    "num_classes": 6,
    "dropout": 0.3,                   # thêm Dropout để tránh overfit
    "activation": "relu",
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 20,
    "optimizer": "adam",
    "batchnorm": True,                # thêm BatchNorm
    "augmentation": "medical",             # augment ảnh y tế
}