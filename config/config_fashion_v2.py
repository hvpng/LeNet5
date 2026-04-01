CONFIG_FASHION_V2 = {
    "dataset": "fashion_mnist",
    "data_path": "/content/Data",
    "input_channels": 1,
    "num_classes": 10,
    "dropout": 0.4,                   # Dropout cao hơn vì overfit nặng
    "learning_rate": 0.001,
    "lr_schedule": {
        2:  0.0002,
        5:  0.0001,
        8:  0.00005,
        12: 0.00001,
    },
    "batch_size": 64,
    "epochs": 30,                     # tăng epoch vì dataset khó hơn
    "optimizer": "adam",
    "augmentation": True,
    "aug_type":   "fashion",             # thêm augmentation cho Fashion
}