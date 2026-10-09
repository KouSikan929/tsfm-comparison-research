"""Native four-expert integration tests; no mocked forecast adapters."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from fusion.data import FEATURES, WindowDataset
from fusion.model import FeatureScaler, JointFusion, gradient_audit, objective
from train_joint_fusion import load_model


@pytest.fixture(scope="module")
def sample():
    torch.set_num_threads(2)
    ds = WindowDataset(ROOT / "data/raw", ["SP500"], "train", limit_per_market=2)
    return ds, {k: v.unsqueeze(0) for k, v in ds[0].items()}


def tiny_config():
    return dict(tiny=True, context=400, horizon=20, baselines=True)


def test_four_real_backbones_receive_fused_loss_and_update(sample, tmp_path):
    ds, batch = sample
    torch.manual_seed(7)
    model = JointFusion(tiny_config())
    model.scaler.fit(ds.features)
    model.train()
    before = {k: p.detach().clone() for k, p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    out = model(batch)
    assert out["prediction"].shape == (1, 20)
    assert out["weights"].shape == (1, 20, 7)
    torch.testing.assert_close(out["weights"].sum(-1), torch.ones(1, 20))
    assert (out["weights"] > 0).all() and (out["weights"] < 1).all()
    assert (out["quantiles"][..., 0] <= out["prediction"]).all()
    assert (out["quantiles"][..., 2] >= out["prediction"]).all()
    audit = gradient_audit(model, out, batch)
    assert len(audit) == 5
    loss, _ = objective(out, batch)
    loss.backward()
    assert all(p.grad is None for p in model.experts["kronos"].tokenizer.parameters())
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), name
    optimizer.step()
    for expert in model.experts:
        assert any(not torch.equal(before[k], p) for k, p in model.named_parameters()
                   if k in before and k.startswith(f"experts.{expert}.backbone.")), expert
    model.eval()
    with torch.no_grad():
        expected = model(batch)
        # No hidden teacher forcing: target changes must not change forward.
        changed = dict(batch, target=batch["target"]*100)
        torch.testing.assert_close(model(changed)["prediction"], expected["prediction"])
    checkpoint = tmp_path / "fusion.pt"
    torch.save(dict(format_version=1, config=model.config, features=list(FEATURES),
                    model=model.state_dict()), checkpoint)
    restored, _ = load_model(checkpoint)
    restored.eval()
    with torch.no_grad():
        actual = restored(batch)
    torch.testing.assert_close(actual["prediction"], expected["prediction"], rtol=0, atol=0)
    torch.testing.assert_close(actual["weights"], expected["weights"], rtol=0, atol=0)


def test_date_splits_features_do_not_see_future(tmp_path):
    dates = pd.date_range("2024-01-01", "2025-08-30")
    rng = np.random.default_rng(9)
    close = 100*np.exp(np.cumsum(rng.normal(0, .01, len(dates))))
    df = pd.DataFrame(dict(timestamps=dates, open=close, high=close*1.01,
                           low=close*.99, close=close, volume=np.ones(len(close))*100))
    path = tmp_path / "EX_daily.csv"
    df.to_csv(path, index=False)
    kw = dict(raw_dir=tmp_path, markets=["EX"], context=40, horizon=20, limit_per_market=2)
    train = WindowDataset(split="train", **kw)
    val = WindowDataset(split="val", **kw)
    test = WindowDataset(split="test", **kw)
    assert train.manifest()["EX"]["last_target"] <= "2024-06-30"
    assert val.manifest()["EX"]["first_target"] > "2024-06-30"
    assert val.manifest()["EX"]["last_target"] <= "2025-06-30"
    assert test.manifest()["EX"]["first_target"] >= "2025-07-01"
    df.loc[df.timestamps >= "2025-07-01", ["open", "high", "low", "close"]] *= 10
    df.to_csv(path, index=False)
    again = WindowDataset(split="test", **kw)
    np.testing.assert_allclose(test.features[0], again.features[0], equal_nan=True)
    np.testing.assert_allclose(train.features, WindowDataset(split="train", **kw).features, equal_nan=True)
    np.testing.assert_allclose(val.features, WindowDataset(split="val", **kw).features, equal_nan=True)
    assert not torch.equal(test[0]["target"], again[0]["target"])


def test_scaler_missing_and_constant_features():
    x = np.ones((3, len(FEATURES)), dtype=np.float32)
    x[:, 0] = np.nan
    x[1, 1] = np.inf
    scaler = FeatureScaler()
    scaler.fit(x)
    out = scaler(torch.from_numpy(x))
    assert out.shape == (3, 28)
    assert torch.isfinite(out).all()
    assert (out[:, 14] == 1).all()
    center = scaler.center.clone()
    scaler(torch.full((1, 14), 1e8))
    torch.testing.assert_close(scaler.center, center)


def test_refuse_leaking_split():
    with pytest.raises(ValueError, match="final test"):
        WindowDataset(ROOT / "data/raw", ["SP500"], "train", val_end="2025-07-01")


def test_evaluation_window_ids_match_existing_schedule():
    ds = WindowDataset(ROOT / "data/raw", ["SP500", "BTC"], "test", limit_per_market=2)
    for index, (market_idx, start, _) in enumerate(ds.windows):
        name = ds.markets[market_idx]
        existing = pd.read_csv(ROOT / "data" / f"eval_windows_{name}.csv")
        match = existing[existing.target_start_idx == start]
        assert len(match) == 1
        assert int(match.iloc[0].window_id) == ds.window_ids[index]
