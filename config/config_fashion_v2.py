CONFIG_FASHION_V2 = {
    "dataset": "fashion_mnist",
    "data_path": "/content/Data",
    "input_channels": 1,
    "num_classes": 10,
    "dropout": 0.4,                   # Dropout cao hơn vì overfit nặng
    "activation": "relu",
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 30,                     # tăng epoch vì dataset khó hơn
    "optimizer": "adam",
    "batchnorm": True,                # thêm BatchNorm
    "augmentation": "fashion",             # thêm augmentation cho Fashion
}