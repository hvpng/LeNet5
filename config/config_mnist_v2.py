CONFIG_MNIST_V2 = {
    "dataset":        "mnist",
    "data_path":      "/content/Data",
    "num_classes":    10,
    "input_channels": 1,
    "dropout":        0.3,
    "optimizer":      "adam",   # cải tiến
    "learning_rate":  0.001,
    "lr_schedule": {
        2:  0.0002,
        5:  0.0001,
        8:  0.00005,
        12: 0.00001,
    },
    "batch_size":     64,
    "epochs":         20,
    "augmentation":   True,
    "aug_type":       "mnist",
}