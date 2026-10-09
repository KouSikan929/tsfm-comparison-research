"""Real model adapters; tiny mode uses the SAME native classes at reduced size.

Kronos and Lag-Llama use direct multi-horizon heads on context representations.
Their original autoregressive samplers are deliberately not the training graph.
Kronos's discrete tokenizer is fixed. Unused original output heads are fixed.
"""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]


def native_classes():
    # Kronos imports model.module. Load iTransformer by file to avoid the two
    # vendored repositories' identically named top-level `model` packages.
    for sub in ("Kronos", "lag-llama", "iTransformer"):
        path = str(ROOT / "models" / sub)
        if path not in sys.path:
            sys.path.append(path)
    from model.kronos import Kronos, KronosTokenizer
    from lag_llama.model.module import LagLlamaModel
    spec = importlib.util.spec_from_file_location(
        "fusion_itransformer_native", ROOT / "models/iTransformer/model/iTransformer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from transformers import TimesFm2_5Config, TimesFm2_5ModelForPrediction
    return Kronos, KronosTokenizer, LagLlamaModel, module.Model, TimesFm2_5Config, TimesFm2_5ModelForPrediction


def normalize(x):
    mean = x.mean(1, keepdim=True)
    std = x.std(1, keepdim=True, unbiased=False).clamp_min(1e-5)
    return (x - mean) / std, mean, std


class DirectHead(nn.Module):
    def __init__(self, width, horizon):
        super().__init__()
        self.horizon = horizon
        self.net = nn.Linear(width, horizon * 3)
        nn.init.normal_(self.net.weight, std=.01)
        nn.init.zeros_(self.net.bias)

    def forward(self, representation, last):
        delta, lower, upper = self.net(representation).reshape(-1, self.horizon, 3).unbind(-1)
        point = last + delta
        return torch.stack((point - F.softplus(lower), point,
                            point + F.softplus(upper)), dim=-1)


class KronosExpert(nn.Module):
    def __init__(self, backbone, tokenizer, horizon):
        super().__init__()
        self.backbone, self.tokenizer = backbone, tokenizer
        tokenizer.requires_grad_(False)
        backbone.head.requires_grad_(False)
        backbone.dep_layer.requires_grad_(False)
        self.readout = DirectHead(backbone.d_model, horizon)

    def forward(self, batch):
        x, mean, std = normalize(batch["context"])
        self.tokenizer.eval()
        with torch.no_grad():
            tokens = self.tokenizer.encode(x.clamp(-5, 5), half=True)
        b = self.backbone
        z = b.token_drop(b.embedding(tokens) + b.time_emb(batch["stamps"]))
        for layer in b.transformer:
            z = layer(z)
        z = b.norm(z)[:, -1]
        q = self.readout(z, x[:, -1, 3:4])
        return q * std[:, :, 3:4] + mean[:, :, 3:4]


class LagLlamaExpert(nn.Module):
    def __init__(self, backbone, horizon, time_feat=True):
        super().__init__()
        self.backbone, self.time_feat = backbone, time_feat
        backbone.param_proj.requires_grad_(False)
        self.readout = DirectHead(backbone.transformer.wte.out_features, horizon)

    def forward(self, batch):
        b = self.backbone
        close = batch["context"][:, :, 3]
        lag = max(b.lags_seq)
        # Exactly context observations. Unavailable earlier lags are padded and
        # marked unobserved, never fetched from before the 400-observation input.
        past = F.pad(close, (lag, 0))
        observed = F.pad(torch.ones_like(close), (lag, 0))
        time = F.pad(batch["time_features"], (0, 0, lag, 0)) if self.time_feat else None
        # Empty future features bypass a native prepare_input conditional bug;
        # no future target or future observed covariate enters this graph.
        future_time = time[:, :0] if time is not None else None
        inputs, loc, scale = b.prepare_input(past, observed, time, future_time)
        z = b.transformer.wte(inputs)
        for block in b.transformer.h:
            z = block(z, use_kv_cache=False)
        z = b.transformer.ln_f(z)[:, -1]
        q = self.readout(z, (close[:, -1:] - loc) / scale.clamp_min(1e-5))
        return q * scale.unsqueeze(-1) + loc.unsqueeze(-1)


class TimesFMExpert(nn.Module):
    def __init__(self, backbone, horizon, context):
        super().__init__()
        self.backbone, self.horizon = backbone, horizon
        if context > backbone.config.context_length or horizon > backbone.config.horizon_length:
            raise ValueError("Requested window exceeds TimesFM native config")
        patch = backbone.config.patch_length
        self.aligned_context = ((context + patch - 1) // patch) * patch

    def forward(self, batch):
        x, mean, std = normalize(batch["context"][:, :, 3])
        out = self.backbone(past_values=list(x.unbind(0)),
                            forecast_context_len=self.aligned_context,
                            truncate_negative=False)
        q = out.full_predictions[:, :self.horizon]
        quantiles = list(self.backbone.config.quantiles)
        lo, hi = quantiles.index(.1) + 1, quantiles.index(.9) + 1
        point = out.mean_predictions[:, :self.horizon]
        # Enforce order without assuming the native quantile head never crosses.
        q = torch.stack((torch.minimum(q[:, :, lo], point), point,
                         torch.maximum(q[:, :, hi], point)), dim=-1)
        return q * std.unsqueeze(-1) + mean.unsqueeze(-1)


class ITransformerExpert(nn.Module):
    def __init__(self, backbone, horizon):
        super().__init__()
        self.backbone = backbone
        self.width = nn.Linear(horizon, 2 * horizon)
        self.horizon = horizon

    def forward(self, batch):
        x, mean, std = normalize(batch["context"])
        point = self.backbone(x, None, None, None)[:, :, 3]
        widths = F.softplus(self.width(point).reshape(-1, self.horizon, 2))
        q = torch.stack((point - widths[:, :, 0], point, point + widths[:, :, 1]), dim=-1)
        return q * std[:, :, 3:4] + mean[:, :, 3:4]


def build_experts(config, initialize=True):
    Kronos, Tokenizer, LagLlama, ITransformer, TFConfig, TimesFM = native_classes()
    from gluonts.torch.distributions import StudentTOutput
    tiny = config["tiny"]
    horizon, context = config["horizon"], config["context"]
    if tiny:
        tokenizer = Tokenizer(6, 32, 4, 64, 2, 2, 0., 0., 0., 4, 4,
                              .25, 1., 1., 1., 4)
        kronos = Kronos(4, 4, 1, 32, 4, 64, 0., 0., 0., 0., False)
        lag_kwargs = dict(context_length=context, max_context_length=max(512, context),
                          scaling="std", input_size=1, n_layer=1, n_embd_per_head=16,
                          n_head=2, lags_seq=[0, 1, 2, 7], time_feat=True)
        timesfm = TimesFM(TFConfig(context_length=max(512, context), horizon_length=32,
                                   patch_length=32, num_hidden_layers=1, hidden_size=64,
                                   intermediate_size=64, head_dim=16, num_attention_heads=4,
                                   num_key_value_heads=4, output_quantile_len=32,
                                   force_flip_invariance=False))
    elif initialize:
        tokenizer = Tokenizer.from_pretrained(config["tokenizer_id"])
        kronos = Kronos.from_pretrained(config["kronos_id"])
        # Only trusted local/public official Lag-Llama checkpoints; format
        # contains GluonTS objects and requires weights_only=False.
        path = Path(config["lag_checkpoint"])
        if not path.is_file():
            raise FileNotFoundError(f"Lag-Llama checkpoint missing: {path}")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        lag_kwargs = dict(ckpt["hyper_parameters"]["model_kwargs"])
        native_context = lag_kwargs["context_length"]
        lag_kwargs.update(context_length=context,
                          rope_scaling={"type": "linear", "factor": max(1., (context+horizon)/native_context)})
        lag_kwargs.pop("distr_output", None)
        # Preserve native lags/architecture/scaler; avoid Lightning inference.
        timesfm = TimesFM.from_pretrained(config["timesfm_id"], torch_dtype=torch.float32)
    else:
        # Self-contained reconstruction: no network or original weight paths.
        tokenizer = Tokenizer(**config["architecture"]["tokenizer"])
        kronos = Kronos(**config["architecture"]["kronos"])
        lag_kwargs = dict(config["architecture"]["lag"])
        timesfm = TimesFM(TFConfig.from_dict(config["architecture"]["timesfm"]))
    lag = LagLlama(**lag_kwargs, distr_output=StudentTOutput())
    if not tiny and initialize:
        state = {k.removeprefix("model."): v for k, v in ckpt["state_dict"].items()
                 if k.startswith("model.")}
        lag.load_state_dict(state, strict=True)
    it_config = dict(seq_len=context, pred_len=horizon, output_attention=False,
                     use_norm=True, d_model=32 if tiny else 128, embed="timeF", freq="d",
                     dropout=0., class_strategy="projection", factor=1, n_heads=4 if tiny else 8,
                     d_ff=64 if tiny else 128, activation="gelu", e_layers=1 if tiny else 2)
    itransformer = ITransformer(SimpleNamespace(**it_config))
    # Persist constructors, not pickled model objects, to allow offline reload.
    import inspect
    def constructor_attrs(obj, cls):
        return {k: getattr(obj, k) for k in inspect.signature(cls).parameters}
    tokenizer_config = dict(d_in=tokenizer.d_in, d_model=tokenizer.d_model,
        n_heads=tokenizer.n_heads, ff_dim=tokenizer.ff_dim,
        n_enc_layers=tokenizer.enc_layers, n_dec_layers=tokenizer.dec_layers,
        ffn_dropout_p=tokenizer.ffn_dropout_p, attn_dropout_p=tokenizer.attn_dropout_p,
        resid_dropout_p=tokenizer.resid_dropout_p, s1_bits=tokenizer.s1_bits,
        s2_bits=tokenizer.s2_bits,
        **{k: getattr(tokenizer.tokenizer.bsq, k)
           for k in ("beta", "gamma0", "gamma", "zeta", "group_size")})
    config["architecture"] = {"tokenizer": tokenizer_config,
        "kronos": constructor_attrs(kronos, Kronos), "lag": lag_kwargs,
        "timesfm": timesfm.config.to_dict(), "itransformer": it_config}
    return nn.ModuleDict({"kronos": KronosExpert(kronos, tokenizer, horizon),
        "lag_llama": LagLlamaExpert(lag, horizon, lag_kwargs.get("time_feat", True)),
        "timesfm": TimesFMExpert(timesfm, horizon, context),
        "itransformer": ITransformerExpert(itransformer, horizon)})
