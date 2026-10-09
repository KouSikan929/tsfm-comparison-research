"""Continuous horizon-conditioned gate with four jointly trained experts."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from fusion.data import FEATURES
from fusion.experts import build_experts


class FeatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("center", torch.zeros(len(FEATURES)))
        self.register_buffer("scale", torch.ones(len(FEATURES)))

    def fit(self, features):
        x = np.where(np.isfinite(features), features, np.nan)
        # All-missing columns retain zero imputation and an explicit mask.
        for i in range(x.shape[1]):
            valid = x[:, i][np.isfinite(x[:, i])]
            if len(valid):
                self.center[i] = float(np.median(valid))
                self.scale[i] = max(float(np.std(valid)), 1e-5)

    def forward(self, x):
        valid = torch.isfinite(x)
        z = (torch.where(valid, x, self.center) - self.center) / self.scale
        return torch.cat((z.clamp(-10, 10), (~valid).to(z.dtype)), dim=-1)


class JointFusion(nn.Module):
    def __init__(self, config, initialize=True):
        super().__init__()
        self.config = config
        self.experts = build_experts(config, initialize=initialize)
        self.neural_names = list(self.experts)
        self.names = self.neural_names + (["persistence", "drift", "moving_average"]
                                         if config.get("baselines", True) else [])
        self.scaler = FeatureScaler()
        # All 14 features + missing masks + all expert point/width pairs +
        # disagreement + normalized horizon. Native hidden states feed the
        # expert heads; the gate receives their predictions and widths.
        n = len(self.names)
        self.gate = nn.Sequential(nn.Linear(2*len(FEATURES) + 2*n + 2, 64),
                                  nn.SiLU(), nn.Linear(64, n))
        nn.init.normal_(self.gate[-1].weight, std=.01)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, batch):
        qs = [expert(batch) for expert in self.experts.values()]
        close = batch["context"][:, :, 3]
        last = close[:, -1:]
        horizon = self.config["horizon"]
        steps = torch.arange(1, horizon+1, device=close.device, dtype=close.dtype)[None]
        if self.config.get("baselines", True):
            base_points = [last.expand(-1, horizon),
                           last + (last-close[:, :1]) / (close.shape[1]-1) * steps,
                           close[:, -20:].mean(1, keepdim=True).expand(-1, horizon)]
            # Historical random-walk width is only a gate input/initial interval
            # heuristic, not a claim of calibrated 80% coverage.
            width = close.diff(dim=1).std(dim=1, keepdim=True, unbiased=False) * steps.sqrt() * 1.28155
            qs.extend(torch.stack((p-width, p, p+width), dim=-1) for p in base_points)
        quantiles = torch.stack(qs, dim=2)  # B, H, experts, [q10, point, q90]
        point = quantiles[..., 1]
        scale = close.std(1, keepdim=True, unbiased=False).clamp_min(1e-5)
        relative = (point-last.unsqueeze(-1)) / scale.unsqueeze(-1)
        widths = (quantiles[..., 2]-quantiles[..., 0]) / scale.unsqueeze(-1)
        state = self.scaler(batch["features"])[:, None].expand(-1, horizon, -1)
        disagreement = relative.std(-1, keepdim=True, unbiased=False)
        h = (steps / horizon).unsqueeze(-1).expand(close.shape[0], -1, -1)
        inputs = torch.cat((state, relative.clamp(-50, 50), widths.clamp(0, 100),
                            disagreement.clamp(0, 100), h), dim=-1)
        weights = self.gate(inputs).softmax(-1)
        fused = (quantiles * weights.unsqueeze(-1)).sum(2)
        return {"prediction": fused[..., 1], "quantiles": fused,
                "weights": weights, "expert_quantiles": quantiles}


def pinball(q, y, scale):
    levels = q.new_tensor([.1, .5, .9])
    err = (y.unsqueeze(-1)-q) / scale.unsqueeze(-1)
    return torch.maximum(levels*err, (levels-1)*err).mean()


def objective(output, batch, expert_weight=.2, quantile_weight=.1, gate_penalty=1e-4):
    target = batch["target"]
    scale = batch["context"][:, -1, 3:4].clamp_min(1e-5)
    # Percentage-like error, denominator observable at prediction time.
    fused = ((output["prediction"]-target)/scale).abs().mean()
    expert_q = output["expert_quantiles"][:, :, :4]
    expert = ((expert_q[..., 1]-target.unsqueeze(-1))/scale.unsqueeze(-1)).abs().mean()
    qloss = pinball(output["quantiles"], target, scale)
    qloss = qloss + pinball(expert_q, target.unsqueeze(-1), scale.unsqueeze(-1))
    weights = output["weights"]
    # Positive KL to uniform softly discourages premature expert collapse.
    reg = (weights * (weights.clamp_min(1e-8).log() + np.log(weights.shape[-1]))).sum(-1).mean()
    total = fused + expert_weight*expert + quantile_weight*qloss + gate_penalty*reg
    return total, {"fused_relative_mae": float(fused.detach()),
                   "expert_relative_mae": float(expert.detach()),
                   "quantile_loss": float(qloss.detach())}


def gradient_audit(model, output, batch):
    """Test ONLY final point-loss gradients, excluding expert auxiliary losses."""
    loss = ((output["prediction"]-batch["target"])/batch["context"][:, -1, 3:4]).abs().mean()
    selected = []
    for name, expert in model.experts.items():
        # Probe an actual native backbone, not just its new prediction head.
        key, parameter = next((k, p) for k, p in expert.backbone.named_parameters()
                              if p.requires_grad and p.ndim >= 2)
        selected.append((f"{name}.backbone.{key}", parameter))
    selected.append(("gate.0.weight", model.gate[0].weight))
    gradients = torch.autograd.grad(loss, [p for _, p in selected], retain_graph=True, allow_unused=True)
    report = {}
    for (name, _), grad in zip(selected, gradients):
        if grad is None or not torch.isfinite(grad).all() or grad.abs().sum() <= 0:
            raise RuntimeError(f"Fusion loss does not reach {name}")
        report[name] = float(grad.norm())
    return report
