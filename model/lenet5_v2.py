import torch.nn as nn


class LeNet5V2(nn.Module):
    """
    LeNet-5 cải tiến hiện đại:
    - ReLU thay ScaledTanh
    - AvgPool thay Subsampling trainable
    - Full Conv2d thay C3 Partial Connection
    - Linear + CrossEntropy thay RBF + MAP Loss
    - BatchNorm + Dropout
    """
    def __init__(
        self,
        num_classes: int    = 10,
        input_channels: int = 1,
        dropout: float      = 0.0,
    ):
        super().__init__()

        self.features = nn.Sequential(
            # C1
            nn.Conv2d(input_channels, 6, kernel_size=5),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            # S2
            nn.AvgPool2d(kernel_size=2, stride=2),
            # C3 — full connection
            nn.Conv2d(6, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # S4
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            # C5
            nn.Linear(16 * 5 * 5, 120),
            nn.BatchNorm1d(120),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            # F6
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            # Output
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x