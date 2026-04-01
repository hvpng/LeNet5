import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ─────────────────────────────────────────────────────────────
# Appendix A — Scaled Tanh: f(a) = A*tanh(S*a), A=1.7159, S=2/3
# "overall gain ~1 in normal operating, f(1)=1, f(-1)=-1"
# ─────────────────────────────────────────────────────────────
class ScaledTanh(nn.Module):
    A = 1.7159
    S = 2.0 / 3.0
    def forward(self, x):
        return self.A * torch.tanh(self.S * x)


# ─────────────────────────────────────────────────────────────
# Section II.B — Subsampling S2/S4
# "four inputs added, × trainable coeff, + trainable bias → sigmoid"
# S2: 6 maps, S4: 16 maps — mỗi map 1 weight + 1 bias (trainable)
# ─────────────────────────────────────────────────────────────
class SubsamplingLayer(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        # 1 trainable weight và bias PER CHANNEL
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias   = nn.Parameter(torch.zeros(num_channels))
        self.act    = ScaledTanh()

    def forward(self, x):
        # Sum (not average) → multiply by 4
        x = F.avg_pool2d(x, kernel_size=2, stride=2) * 4.0
        w = self.weight.view(1, -1, 1, 1)
        b = self.bias.view(1, -1, 1, 1)
        return self.act(w * x + b)


# ─────────────────────────────────────────────────────────────
# Section II.B — C3 Partial Connection, Table I trang 8
# Columns = 16 C3 feature maps, Rows = 6 S2 feature maps
# X = connected
# ─────────────────────────────────────────────────────────────
#        0  1  2  3  4  5  6  7  8  9 10  11 12 13 14 15
# row 0: X        X  X     X  X  X     X  X  X  X     X
# row 1: X  X        X  X     X  X  X     X  X     X  X
# row 2: X  X  X        X  X     X  X  X     X  X  X  X 
# Bảng chính xác từ Table I paper:
C3_TABLE = [
    (0,1,2),        # col 0
    (1,2,3),        # col 1
    (2,3,4),        # col 2
    (3,4,5),        # col 3
    (0,4,5),        # col 4
    (0,1,5),        # col 5
    (0,1,2,3),      # col 6
    (1,2,3,4),      # col 7
    (2,3,4,5),      # col 8
    (0,3,4,5),      # col 9
    (0,1,4,5),      # col 10
    (0,1,2,5),      # col 11
    (0,1,3,4),      # col 12
    (1,2,4,5),      # col 13
    (0,2,3,5),      # col 14
    (0,1,2,3,4,5),  # col 15
]
# Kiểm tra params: 6*(3*25+1) + 6*(4*25+1) + 3*(4*25+1) + 1*(6*25+1) = 1516 

class C3PartialConnection(nn.Module):
    """Table I, Section II.B — 1516 trainable params (đúng paper)."""
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=len(inp), out_channels=1,
                      kernel_size=5, bias=True)
            for inp in C3_TABLE
        ])
        self.act = ScaledTanh()

    def forward(self, x):
        # x: (B, 6, 14, 14)
        out = [conv(x[:, list(idx), :, :]) for conv, idx in
               zip(self.convs, C3_TABLE)]
        return self.act(torch.cat(out, dim=1))  # (B, 16, 10, 10)


# ─────────────────────────────────────────────────────────────
# Section II.B — RBF Output Layer (Eq. 7, trang 8)
# y_i = Σ_j (x_j - w_ij)²
# w_ij: fixed bitmap -1/+1, 7×12 = 84 bits per class
# "chosen by hand and kept fixed (at least initially)"
# ─────────────────────────────────────────────────────────────
class RBFOutput(nn.Module):
    """
    Eq. 7: y_i = Σ_j (x_j - w_ij)²
    Weights = fixed bitmap -1/+1 (không trainable).
    Predicted class = argmin(y_i).
    """
    def __init__(self, num_classes: int = 10, in_features: int = 84):
        super().__init__()
        # Fixed -1/+1 bitmap. Paper: "designed to represent stylized image"
        # Dùng fixed random -1/+1 (đủ cho digit recognition)
        torch.manual_seed(1998)   # year of paper
        w = torch.bernoulli(torch.full((num_classes, in_features), 0.5)) * 2 - 1
        self.register_buffer('weight', w)   # NOT trainable

    def forward(self, x):
        # x: (B, 84) — output của F6
        # y_i = ||x - w_i||² = Σ(x_j - w_ij)²
        diff = x.unsqueeze(1) - self.weight.unsqueeze(0)  # (B, C, 84)
        return (diff ** 2).sum(dim=2)                      # (B, C) distances


# ─────────────────────────────────────────────────────────────
# Section II.C — Loss Functions
# MSE (Eq. 8): E = (1/P) Σ y_Dp
# MAP (Eq. 9): E = (1/P) Σ [ y_Dp + log(e^{-j} + Σ_i e^{-y_i}) ]
# ─────────────────────────────────────────────────────────────
class MSELoss_RBF(nn.Module):
    """Eq. 8 — Simplest loss, minimize distance of correct class."""
    def forward(self, rbf_out: torch.Tensor, labels: torch.Tensor):
        B = rbf_out.size(0)
        y_p = rbf_out[torch.arange(B), labels]   # correct class distance
        return y_p.mean()


class MAPLoss_RBF(nn.Module):
    """
    Eq. 9 — MAP criterion (dùng trong paper thực tế).
    E = y_Dp + log(e^{-j} + Σ_i e^{-y_i})
    j > 0: prevents pushing up already-large penalties.
    Paper không nêu giá trị j cụ thể → dùng j=0 (safe default).
    """
    def __init__(self, j: float = 0.0):
        super().__init__()
        self.j = j

    def forward(self, rbf_out: torch.Tensor, labels: torch.Tensor):
        B = rbf_out.size(0)
        # Term 1: y_Dp — distance of correct class (want small)
        y_p = rbf_out[torch.arange(B), labels]

        # Term 2: log(e^{-j} + Σ_i e^{-y_i}) — competitive term
        # Σ_i e^{-y_i}: sum over ALL classes
        log_sum = torch.log(
            math.exp(-self.j) +
            torch.exp(-rbf_out).sum(dim=1)
        )

        return (y_p + log_sum).mean()


# ─────────────────────────────────────────────────────────────
# LeNet-5 — LeCun et al. 1998
# ─────────────────────────────────────────────────────────────
class LeNet5(nn.Module):
    """
    LeNet-5 đúng trong paper gốc.

    Layers (Section II.B):
        Input : 1×32×32
        C1    : Conv2d(1→6, k=5) + ScaledTanh  → 6×28×28   [156 params]
        S2    : Subsampling(6)   + ScaledTanh  → 6×14×14   [12 params]
        C3    : PartialConv(Table I) + ScaledTanh → 16×10×10 [1516 params]
        S4    : Subsampling(16)  + ScaledTanh  → 16×5×5    [32 params]
        C5    : Conv2d(16→120,k=5)+ScaledTanh → 120×1×1   [48120 params]
        F6    : Linear(120→84)  + ScaledTanh  → 84         [10164 params]
        Out   : RBF(84→10) fixed                → 10        [840 fixed]
    Total trainable: ~60,000 (paper: 60,000)
    """
    def __init__(
        self,
        num_classes: int    = 10,
        input_channels: int = 1,
        # v2 options
        batchnorm: bool     = False,
        dropout: float      = 0.0,
    ):
        super().__init__()
        self.act = ScaledTanh()

        # C1
        self.c1    = nn.Conv2d(input_channels, 6, kernel_size=5)
        self.bn_c1 = nn.BatchNorm2d(6) if batchnorm else nn.Identity()

        # S2
        self.s2 = SubsamplingLayer(num_channels=6)

        # C3 (partial connection)
        self.c3 = C3PartialConnection()

        # S4
        self.s4 = SubsamplingLayer(num_channels=16)

        # C5: Conv2d 16→120, kernel 5×5 (input 5×5 → output 1×1)
        # Section II.B: "labeled as conv layer, not FC, for scalability"
        self.c5    = nn.Conv2d(16, 120, kernel_size=5)
        self.bn_c5 = nn.BatchNorm1d(120) if batchnorm else nn.Identity()

        # F6: fully connected 120→84
        self.f6    = nn.Linear(120, 84)
        self.bn_f6 = nn.BatchNorm1d(84) if batchnorm else nn.Identity()

        # Dropout (v2 only)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Output: RBF (fixed weights)
        self.output = RBFOutput(num_classes=num_classes, in_features=84)

        # Appendix A: weight init Uniform(-2.4/Fi, +2.4/Fi)
        self._init_weights()

    def _init_weights(self):
        """Appendix A: Uniform(-2.4/F_i, +2.4/F_i), F_i = fan-in."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_in = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
                bound = 2.4 / fan_in
                nn.init.uniform_(m.weight, -bound, bound)
                if m.bias is not None:
                    nn.init.uniform_(m.bias, -bound, bound)
            elif isinstance(m, nn.Linear):
                fan_in = m.in_features
                bound = 2.4 / fan_in
                nn.init.uniform_(m.weight, -bound, bound)
                nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x):
        # C1: Conv → ScaledTanh
        x = self.act(self.bn_c1(self.c1(x)))   # Conv → BN (v2) → ScaledTanh
        # S2: sum×w+b → ScaledTanh (bên trong SubsamplingLayer)
        x = self.s2(x)                          # (B, 6, 14, 14)

        # C3: partial conv → ScaledTanh (bên trong C3PartialConnection)
        x = self.c3(x)
        # S4
        x = self.s4(x)                          # (B, 16, 5, 5)

        # C5: Conv → ScaledTanh
        x = self.act(self.c5(x))               # (B, 120, 1, 1)
        x = x.view(x.size(0), -1)              # (B, 120)
        x = self.drop(self.bn_c5(x))

        # F6: Linear → ScaledTanh
        x = self.act(self.bn_f6(self.f6(x)))   # (B, 84)
        x = self.drop(x)

        # Output RBF
        return self.output(x)                   # (B, num_classes)