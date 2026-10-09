"""Joint soft-gated training. No cached predictions or future labels enter forward()."""
import argparse
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from fusion.data import FEATURES, WindowDataset
from fusion.model import JointFusion, gradient_audit, objective

ROOT = Path(__file__).resolve().parents[1]
MARKETS = "SP500 SSE SZSE Nikkei225 CSI300 TOPIX KO BTC NVDA GLD".split()


def move(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def save_checkpoint(path, payload):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_model(path, device="cpu"):
    # New fusion checkpoints contain tensors and primitive configuration only.
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if ckpt.get("format_version") != 1 or ckpt["features"] != list(FEATURES):
        raise ValueError("Incompatible checkpoint format / feature order")
    model = JointFusion(ckpt["config"], initialize=False)
    model.load_state_dict(ckpt["model"], strict=True)
    return model.to(device), ckpt


@torch.no_grad()
def evaluate(model, dataset, device, batch_size=1, output_path=None):
    import pandas as pd
    model.eval()
    sums, counts, rows = {}, {}, []
    for batch in DataLoader(dataset, batch_size=batch_size):
        b = move(batch, device)
        out = model(b)
        err = ((out["prediction"]-b["target"]) / b["context"][:, -1, 3:4]).abs().mean(1)
        for i, market_id in enumerate(batch["market"].tolist()):
            sums[market_id] = sums.get(market_id, 0.) + err[i].item()
            counts[market_id] = counts.get(market_id, 0) + 1
            if output_path:
                index = batch["index"][i].item()
                _, start, _ = dataset.windows[index]
                dates = dataset.series[market_id][3]
                for h in range(dataset.horizon):
                    row = dict(model="joint_fusion", dataset=dataset.markets[market_id],
                               window_id=dataset.window_ids[index], target_date=str(dates[start+h].date()),
                               step_ahead=h+1, actual_close=b["target"][i, h].item(),
                               pred_close=out["prediction"][i, h].item(),
                               pred_q10=out["quantiles"][i, h, 0].item(),
                               pred_q90=out["quantiles"][i, h, 2].item())
                    row.update({f"weight_{name}": out["weights"][i, h, j].item()
                                for j, name in enumerate(model.names)})
                    rows.append(row)
    metrics = {dataset.markets[m]: sums[m]/counts[m] for m in sums}
    metrics["macro_relative_mae"] = float(np.mean(list(metrics.values())))
    if not np.isfinite(metrics["macro_relative_mae"]):
        raise RuntimeError("Nonfinite validation / evaluation result")
    if output_path:
        pd.DataFrame(rows).to_csv(output_path, index=False)
    return metrics


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", default=MARKETS)
    p.add_argument("--raw-dir", default=str(ROOT / "data/raw"))
    p.add_argument("--output", default=str(ROOT / "data/fusion_runs/joint"))
    p.add_argument("--context", type=int, default=400)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--train-end", default="2024-06-30")
    p.add_argument("--val-end", default="2025-06-30")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--accumulation", type=int, default=8)
    p.add_argument("--samples-per-epoch", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--head-lr", type=float, default=1e-4)
    p.add_argument("--expert-weight", type=float, default=.2)
    p.add_argument("--quantile-weight", type=float, default=.1)
    p.add_argument("--gate-penalty", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--tiny", action="store_true", help="random small REAL native models; smoke only")
    p.add_argument("--limit-per-market", type=int, default=0, help="smoke subset, 0 means all")
    p.add_argument("--no-baselines", dest="baselines", action="store_false", default=True)
    p.add_argument("--kronos-id", default="NeoQuasar/Kronos-small")
    p.add_argument("--tokenizer-id", default="NeoQuasar/Kronos-Tokenizer-base")
    p.add_argument("--timesfm-id", default="google/timesfm-2.5-200m-transformers")
    p.add_argument("--lag-checkpoint", default=str(ROOT / "data/checkpoints/lag-llama/lag-llama.ckpt"))
    p.add_argument("--resume", help="resume at epoch boundary; retains saved data and optimizer settings")
    p.add_argument("--evaluate", help="fusion checkpoint to evaluate (no training)")
    p.add_argument("--split", choices=["val", "test"], default="val")
    return p


def make_dataset(run, split):
    return WindowDataset(run["raw_dir"], run["datasets"], split,
                         context=run["context"], horizon=run["horizon"], stride=run["stride"],
                         train_end=run["train_end"], val_end=run["val_end"],
                         limit_per_market=run["limit_per_market"])


def main():
    args = parser().parse_args()
    if args.resume and args.evaluate:
        raise ValueError("--resume and --evaluate are mutually exclusive")
    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; pass --device cpu for local smoke testing")
    device = torch.device(args.device)
    outdir = Path(args.output)
    # A fresh run must not silently overwrite an earlier experiment.
    if not args.resume and not args.evaluate and outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError(f"Nonempty output directory: {outdir}; use --resume or a new --output")
    outdir.mkdir(parents=True, exist_ok=True)
    if args.evaluate:
        model, ckpt = load_model(args.evaluate, device)
        run = dict(ckpt["config"]["run"])
        run["raw_dir"] = args.raw_dir
        ds = make_dataset(run, args.split)
        prediction_path = outdir / f"{args.split}_predictions.csv"
        if prediction_path.exists():
            raise FileExistsError(prediction_path)
        metrics = evaluate(model, ds, device, args.batch_size, prediction_path)
        (outdir / f"{args.split}_metrics.json").write_text(json.dumps(metrics, indent=2))
        print(json.dumps(metrics, indent=2), flush=True)
        return
    run = vars(args).copy()
    ckpt = None
    if args.resume:
        model, ckpt = load_model(args.resume, device)
        run = dict(ckpt["config"]["run"])
        run.update(epochs=args.epochs, raw_dir=args.raw_dir)
    for key in ("epochs", "batch_size", "accumulation", "samples_per_epoch", "patience"):
        if run[key] < 1:
            raise ValueError(f"{key} must be positive")
    for key in ("lr", "head_lr"):
        if run[key] <= 0:
            raise ValueError(f"{key} must be positive")
    for key in ("expert_weight", "quantile_weight", "gate_penalty", "limit_per_market"):
        if run[key] < 0:
            raise ValueError(f"{key} cannot be negative")
    random.seed(run["seed"])
    np.random.seed(run["seed"])
    torch.manual_seed(run["seed"])
    print("Preparing context-only train/validation features...", flush=True)
    train, val = make_dataset(run, "train"), make_dataset(run, "val")
    print(f"Windows train={len(train)}, val={len(val)}; loading four experts...", flush=True)
    if ckpt is None:
        model = JointFusion({k: run[k] for k in (
            "tiny", "context", "horizon", "baselines", "kronos_id", "tokenizer_id",
            "timesfm_id", "lag_checkpoint")}).to(device)
        model.scaler.fit(train.features)
    model.config["run"] = run
    model.config["features"] = list(FEATURES)
    model.config["trainable_policy"] = "native backbones + gate + heads; fixed tokenizer and unused native heads"
    params = dict(model.named_parameters())
    backbone = [p for name, p in params.items() if p.requires_grad and ".backbone." in name]
    heads = [p for name, p in params.items() if p.requires_grad and ".backbone." not in name]
    optimizer = torch.optim.AdamW([{"params": backbone, "lr": run["lr"]},
                                  {"params": heads, "lr": run["head_lr"]}], weight_decay=.01)
    start, best, stale = 0, float("inf"), 0
    if ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
        start, best, stale = ckpt["epoch"], ckpt["best"], ckpt["stale"]
        torch.set_rng_state(ckpt["rng"].cpu())
        if device.type == "cuda" and ckpt["cuda_rng"]:
            torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
    (outdir / "config.json").write_text(json.dumps(model.config, indent=2), encoding="utf-8")
    (outdir / "data_manifest.json").write_text(json.dumps(
        {"train": train.manifest(), "val": val.manifest()}, indent=2), encoding="utf-8")
    counts = {name: sum(p.numel() for p in expert.parameters() if p.requires_grad)
              for name, expert in model.experts.items()}
    print(f"Trainable expert parameters: {counts}", flush=True)
    audited = False
    for epoch in range(start, run["epochs"]):
        generator = torch.Generator().manual_seed(run["seed"] + epoch)
        sampler = WeightedRandomSampler(train.sampling_weights(), run["samples_per_epoch"],
                                        replacement=True, generator=generator)
        loader = DataLoader(train, batch_size=run["batch_size"], sampler=sampler)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total, samples = 0., 0
        for step, batch in enumerate(loader):
            b = move(batch, device)
            output = model(b)
            if not audited:
                audit = gradient_audit(model, output, b)
                (outdir / "gradient_audit.json").write_text(json.dumps(audit, indent=2))
                print(f"Fusion-only gradient audit passed: {audit}", flush=True)
                audited = True
            loss, parts = objective(output, b, run["expert_weight"],
                                    run["quantile_weight"], run["gate_penalty"])
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at epoch={epoch+1}, batch={step}")
            # Weight partial batches and the final partial accumulation group
            # by their actual sample counts, not the requested batch size.
            group_first = (step // run["accumulation"]) * run["accumulation"]
            group_samples = min(run["accumulation"] * run["batch_size"],
                                run["samples_per_epoch"] - group_first*run["batch_size"])
            n = len(batch["target"])
            (loss*n/group_samples).backward()
            if (step+1) % run["accumulation"] == 0 or step+1 == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            total += loss.item()*n
            samples += n
            if step == 0 or (step+1) % 25 == 0:
                print(f"epoch={epoch+1} batch={step+1}/{len(loader)} loss={loss.item():.6f}", flush=True)
        metrics = evaluate(model, val, device, run["batch_size"])
        improved = metrics["macro_relative_mae"] < best
        best, stale = (metrics["macro_relative_mae"], 0) if improved else (best, stale+1)
        record = dict(epoch=epoch+1, train_loss=total/samples, validation=metrics)
        with (outdir / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record)+"\n")
        payload = dict(format_version=1, config=model.config, features=list(FEATURES),
                       model=model.state_dict(), optimizer=optimizer.state_dict(),
                       epoch=epoch+1, best=best, stale=stale,
                       rng=torch.get_rng_state(),
                       cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else [])
        save_checkpoint(outdir / "last.pt", payload)
        if improved:
            save_checkpoint(outdir / "best.pt", payload)
        print(json.dumps(record), flush=True)
        if stale >= run["patience"]:
            break
    print(f"Completed. Checkpoints and audits: {outdir}", flush=True)


if __name__ == "__main__":
    main()
