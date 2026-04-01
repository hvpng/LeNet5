CONFIG_MNIST_V2 = {
    "dataset":        "mnist",
    "data_path":      "/content/Data",
    "num_classes":    10,
    "input_channels": 1,
    "batchnorm":      True,
    "dropout":        0.3,
    "use_map_loss":   True,
    "optimizer":      "adam",   # cải tiến
    "learning_rate":  0.001,
    "batch_size":     64,
    "epochs":         15,
    "augmentation":   True,
    "aug_type":       "mnist",
}