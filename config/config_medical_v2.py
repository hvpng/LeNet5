CONFIG_MEDICAL_V2 = {
    "dataset": "medical_mnist",
    "data_path": "/content/Data/MNIST Medical",
    "input_channels": 1,
    "num_classes": 6,
    "dropout": 0.3,                  
    "learning_rate": 0.001,
    "lr_schedule": {
        2:  0.0002,
        5:  0.0001,
        8:  0.00005,
        12: 0.00001,
    },
    "batch_size": 64,
    "epochs": 20,
    "optimizer": "adam",
    "augmentation": True,             # augment ảnh y tế
    "aug_type": "medical",            
}