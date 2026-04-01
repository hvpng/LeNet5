CONFIG_MNIST = {
    "dataset":        "mnist",
    "data_path":      "/content/Data",
    "num_classes":    10,
    "input_channels": 1,
    "batchnorm":      False,
    "dropout":        0.0,
    # Loss: MAP (Eq. 9)
    "use_map_loss":   True,
    # Optimizer: Stochastic Diagonal LM (Appendix C)
    "optimizer":      "sgd",
    "learning_rate":  0.0005,   # η ban đầu (Section III.B)
    # LR schedule theo Section III.B:
    # epoch 0-1: 0.0005, 2-4: 0.0002, 5-7: 0.0001, 8-11: 0.00005, 12+: 0.00001
    "lr_schedule": {
        2:  0.0002,
        5:  0.0001,
        8:  0.00005,
        12: 0.00001,
    },
    "batch_size":     64,    
    "epochs":         20,       # paper: "20 iterations"
    "checkpoint_dir":   "/content/drive/MyDrive/LeNet5/checkpoints",
    "checkpoint_every": 1,
}
