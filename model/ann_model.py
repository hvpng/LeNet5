import torch
import torch.nn as nn

class ANN(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes,
                 dropout=0.0, activation="relu", batchnorm=False):
        super(ANN, self).__init__()

        act_fn = nn.ReLU() if activation == "relu" else nn.Tanh()

        layers = []
        in_dim = input_size
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))  # BatchNorm sau Linear
            layers.append(act_fn)
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))         # Dropout sau Activation
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)