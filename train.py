import torch
import torch.optim as optim
import os
from model.lenet5 import MAPLoss_RBF, MSELoss_RBF


class StochasticDiagLM(optim.Optimizer):
    """
    Appendix C — Stochastic Diagonal Levenberg-Marquardt.
    ε_k = η / (μ + h_kk)
    h_kk: diagonal Gauss-Newton Hessian approximation.
    Recomputed on 500 samples trước mỗi epoch.

    PyTorch không có LM built-in.
    Approximation thực tế: SGD với per-param lr scaling từ h_kk.
    Cho v1 (paper gốc): SGD + manual hessian est.
    Cho v2 (cải tiến):  Adam.
    """
    def __init__(self, params, eta=0.001, mu=0.02):
        # μ = 0.02 theo paper (Section III.B)
        defaults = dict(eta=eta, mu=mu)
        super().__init__(params, defaults)
        # h_kk: diagonal Hessian estimate per param
        self._h = {}

    def update_hessian(self, h_estimates: dict):
        """Nhận dict {param_id: h_kk_value} từ hessian estimation."""
        self._h = h_estimates

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            eta = group['eta']
            mu  = group['mu']
            for p in group['params']:
                if p.grad is None:
                    continue
                pid = id(p)
                h   = self._h.get(pid, 1.0)   # default h=1 → ε=η/μ+1
                eps = eta / (mu + h)
                p.add_(p.grad, alpha=-eps)

        return loss


def estimate_hessian_diag(model, loader, criterion, device, n_samples=500):
    """
    Appendix C — Gauss-Newton diagonal Hessian approximation.
    h_kk = (1/P) Σ_p (∂²E^p/∂w_k²)
    Approximation: backward pass với squared gradients.
    Chạy trên n_samples patterns trước mỗi epoch.
    """
    model.train()
    h_accum = {id(p): torch.zeros_like(p) for p in model.parameters()
               if p.requires_grad}
    count = 0

    for X, y in loader:
        if count >= n_samples:
            break
        X, y = X.to(device), y.to(device)
        batch = min(X.size(0), n_samples - count)
        X, y = X[:batch], y[:batch]

        model.zero_grad()
        out  = model(X)
        loss = criterion(out, y)
        loss.backward()

        # Gauss-Newton approx: h_kk ≈ (∂E/∂w_k)²
        for p in model.parameters():
            if p.grad is not None:
                h_accum[id(p)] += p.grad.data ** 2

        count += batch

    # Average và trả về scalar per param
    h_dict = {}
    for p in model.parameters():
        if p.requires_grad:
            h_dict[id(p)] = (h_accum[id(p)] / count).mean().item()

    return h_dict


def train(model, train_loader, config, device, save_path="best_model.pth"):
    """
    Training theo đúng paper:
    - Loss    : MAP criterion (Eq. 9) cho v1; có thể dùng MSE (Eq. 8)
    - Optimizer: Stochastic Diagonal LM (Appendix C) cho v1; Adam cho v2
    - LR schedule: giảm theo paper Section III.B cho v1
    - Epochs  : 20 (paper: "20 iterations through training set")
    """
    use_map  = config.get("use_map_loss", True)
    criterion = MAPLoss_RBF(j=0.0) if use_map else MSELoss_RBF()

    opt_name = config.get("optimizer", "sdlm")

    if opt_name == "adam":
        # v2: modern optimizer
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
        scheduler = None
        use_hessian = False
    else:
        # v1: Stochastic Diagonal LM (paper gốc)
        # Section III.B: η bắt đầu 0.0005, giảm dần
        optimizer = StochasticDiagLM(
            model.parameters(),
            eta=config["learning_rate"],   # η
            mu=0.02                        # μ (paper Section III.B)
        )
        # LR schedule theo paper Section III.B:
        # 0.0005 (2 passes) → 0.0002 (3) → 0.0001 (3) → 0.00005 (4) → 0.00001
        lr_schedule = config.get("lr_schedule", None)
        scheduler   = None
        use_hessian = True

    # ── Checkpoint settings ───────────────────────────────
    ckpt_dir   = config.get("checkpoint_dir", "/content/drive/MyDrive/LeNet5/checkpoints")
    ckpt_every = config.get("checkpoint_every", 1)  # lưu mỗi N epochs
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Resume từ checkpoint nếu có ───────────────────────
    start_epoch = 0
    best_acc = 0.0
    history = {"train_loss": [], "train_acc": []}

    resume_path = os.path.join(ckpt_dir, f"latest_{os.path.basename(save_path)}")
    if os.path.exists(resume_path):
        print(f" Resume từ checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        history = ckpt["history"]
        best_acc = ckpt.get("best_acc", 0.0)
        print(f"   -> Tiếp tục từ epoch {start_epoch + 1}")

    for epoch in range(start_epoch, config["epochs"]):

        # ── LR schedule từ paper Section III.B ──────────────
        if use_hessian and lr_schedule:
            new_lr = lr_schedule.get(epoch, None)
            if new_lr:
                for g in optimizer.param_groups:
                    g['eta'] = new_lr

        # ── Estimate diagonal Hessian (Appendix C) ──────────
        # "reestimate h_kk on 500 samples before each pass"
        if use_hessian:
            h_dict = estimate_hessian_diag(
                model, train_loader, criterion, device, n_samples=500
            )
            optimizer.update_hessian(h_dict)

        # ── Training loop ────────────────────────────────────
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            rbf_out = model(X)              # (B, C) — RBF distances
            loss    = criterion(rbf_out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X.size(0)
            # argmin: class với distance NHỎ NHẤT = predicted class
            preds   = rbf_out.argmin(dim=1)
            correct += (preds == y).sum().item()
            total   += X.size(0)

        if scheduler:
            scheduler.step()

        train_loss = total_loss / total
        train_acc  = correct   / total
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        best_acc = max(best_acc, train_acc)

        # ── Lưu checkpoint định kỳ lên Drive ──────────────
        if (epoch + 1) % ckpt_every == 0:
            ckpt_data = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "history": history,
                "best_acc": best_acc,
                "config": config,
            }
            # Latest checkpoint (ghi đè)
            torch.save(ckpt_data, resume_path)
            # Epoch checkpoint (giữ lại)
            epoch_path = os.path.join(
                ckpt_dir,
                f"epoch{epoch+1:02d}_{os.path.basename(save_path)}"
            )
            torch.save(ckpt_data, epoch_path)

        print(f"Epoch [{epoch+1:02d}/{config['epochs']}] | "
              f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc*100:.2f}%")

    torch.save(model.state_dict(), save_path)
    print(f"\n Model saved: {save_path}")
    return history