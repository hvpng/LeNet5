CONFIG_FASHION = {
    "dataset": "fashion_mnist",
    "data_path": "/content/Data",        # chứa folder FashionMNIST/raw bên trong
    "input_channels": 1,
    "num_classes": 10,
    "dropout": 0.0,               # baseline: không dùng dropout
    "activation": "tanh",
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 20,
    "optimizer": "adam",
}