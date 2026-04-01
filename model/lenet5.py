import torch.nn as nn

class LeNet5(nn.Module):
    """
    LeNet-5 chuẩn (LeCun 1998) — input 1×32×32, output num_classes.
    Hỗ trợ tuỳ chọn: activation, batchnorm, dropout để dễ so sánh v1/v2.
    """
    def __init__(
        self,
        num_classes: int = 10,
        input_channels: int = 1,
        activation: str = "tanh",   # "tanh" | "relu"
        batchnorm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        act = nn.Tanh() if activation == "tanh" else nn.ReLU(inplace=True)

        # ── FEATURE EXTRACTOR ─────────────────────────────
        # C1: 1×32×32 → 6×28×28
        layers_feat = [nn.Conv2d(input_channels, 6, kernel_size=5)]
        if batchnorm: layers_feat.append(nn.BatchNorm2d(6))
        layers_feat.append(act)
        # S2: 6×28×28 → 6×14×14
        layers_feat.append(nn.AvgPool2d(kernel_size=2, stride=2))

        # C3: 6×14×14 → 16×10×10
        layers_feat.append(nn.Conv2d(6, 16, kernel_size=5))
        if batchnorm: layers_feat.append(nn.BatchNorm2d(16))
        layers_feat.append(act)
        # S4: 16×10×10 → 16×5×5
        layers_feat.append(nn.AvgPool2d(kernel_size=2, stride=2))

        self.features = nn.Sequential(*layers_feat)

        # ── CLASSIFIER ────────────────────────────────────
        # C5: 400 → 120
        layers_cls = [nn.Linear(16 * 5 * 5, 120)]
        if batchnorm: layers_cls.append(nn.BatchNorm1d(120))
        layers_cls.append(act)
        if dropout > 0: layers_cls.append(nn.Dropout(dropout))

        # F6: 120 → 84
        layers_cls.append(nn.Linear(120, 84))
        if batchnorm: layers_cls.append(nn.BatchNorm1d(84))
        layers_cls.append(act)
        if dropout > 0: layers_cls.append(nn.Dropout(dropout))

        # Output: 84 → num_classes
        layers_cls.append(nn.Linear(84, num_classes))

        self.classifier = nn.Sequential(*layers_cls)

    def forward(self, x):
        x = self.features(x)          # (B, 16, 5, 5)
        x = x.view(x.size(0), -1)     # (B, 400)
        x = self.classifier(x)        # (B, num_classes)
        return x