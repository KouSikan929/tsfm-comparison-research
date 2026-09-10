"""
Fine-tunes Lag-Llama on one dataset's leakage-safe train+val period
(data/finetune_data/<dataset>_finetune.csv -- see prep_finetune_data.py,
run automatically if missing).

Reuses scripts/eval_lagllama.py's own approach to building a LagLlamaEstimator
(same architecture hyperparameters read from the pretrained checkpoint, same
RoPE context-scaling logic) -- the only difference from zero-shot eval is
that this script actually calls estimator.train() instead of just building a
predictor directly from the untouched pretrained weights. This is standard
GluonTS usage: passing ckpt_path warm-starts the Lightning module from the
pretrained checkpoint, then .train() continues training (fine-tunes) via
PyTorch Lightning.

Output: a serialized GluonTS predictor directory (via predictor.serialize()),
loadable with `gluonts.torch.model.predictor.PyTorchPredictor.deserialize(Path(...))`
-- pass that path to eval_lagllama.py --checkpoint to evaluate it with the
exact same harness used for the zero-shot results.

All the parameters likely to change between the local smoke test and the A40
production run are CLI flags -- see --help. Defaults below are the LOCAL
SMOKE TEST defaults (1 epoch, few batches/epoch); see
docs/finetuning_server_guide.md for recommended A40 production values.

Usage (from the lag-llama conda env):
    python finetune_lagllama.py --dataset SP500 --epochs 1
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# See run_lagllama_smoke_test.py / eval_lagllama.py for why these two
# patches are needed (pkg_resources / torch.load weights_only=True default).
_orig_torch_load = torch.load


def _torch_load_weights_only_false(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_weights_only_false

from huggingface_hub import hf_hub_download  # noqa: E402
from gluonts.dataset.pandas import PandasDataset  # noqa: E402

LAG_LLAMA_ROOT = os.path.join(os.path.dirname(__file__), "..", "models", "lag-llama")
sys.path.append(os.path.abspath(LAG_LLAMA_ROOT))
from lag_llama.gluon.estimator import LagLlamaEstimator  # noqa: E402

SCRIPT_DIR = os.path.dirname(__file__)
FINETUNE_DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data", "finetune_data")
DEFAULT_OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "..", "data", "finetuned_checkpoints", "lag-llama")
PRETRAINED_CKPT_DIR = os.path.join(SCRIPT_DIR, "..", "data", "checkpoints", "lag-llama")
# NOTE: the internal train/val split for fine-tuning is a configurable ratio
# (--val_ratio) applied to the whole leakage-safe finetune CSV, NOT the
# project's fixed TRAIN_END/VAL_END boundary -- that boundary only leaves
# ~250 rows for "val" on most datasets, fewer than context_len+pred_len
# (420 rows by default), which would give GluonTS's validation windowing
# nothing to work with. Same fix applied in finetune_kronos.py, for the
# same underlying reason -- see its --train_ratio/--val_ratio docstring.


def ensure_finetune_data(dataset: str) -> str:
    path = os.path.abspath(os.path.join(FINETUNE_DATA_DIR, f"{dataset}_finetune.csv"))
    if not os.path.exists(path):
        print(f"[prep] {dataset}_finetune.csv not found, generating it first...")
        subprocess.run(
            [sys.executable, os.path.join(SCRIPT_DIR, "prep_finetune_data.py"), "--dataset", dataset],
            check=True,
        )
    return path


def to_gluonts_dataset(close_prices: np.ndarray) -> PandasDataset:
    series = pd.Series(close_prices.astype("float32"))
    series.index = pd.period_range(start="2000-01-01", periods=len(series), freq="D")
    return PandasDataset(series.to_frame(name="close"), target="close")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="e.g. SP500, SSE, SZSE, Nikkei225")
    parser.add_argument("--epochs", type=int, default=1, help="max_epochs (smoke test default: 1)")
    parser.add_argument("--batch_size", type=int, default=8, help="smoke test default: 8; A40 production: 32-64")
    parser.add_argument("--num_batches_per_epoch", type=int, default=10,
                         help="smoke test default: 10 (fast); A40 production: 50-100 (GluonTS default 50)")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context_len", type=int, default=400, help="matches scripts/common/windowing.py's CONTEXT_LEN")
    parser.add_argument("--pred_len", type=int, default=20, help="matches scripts/common/windowing.py's PRED_LEN")
    parser.add_argument("--val_ratio", type=float, default=0.2,
                         help="fraction of the leakage-safe finetune CSV (chronological tail) held out as "
                              "validation; needs val_ratio * total_rows >= context_len+pred_len+1")
    parser.add_argument("--output_dir", default=None, help="default: data/finetuned_checkpoints/lag-llama/<dataset>")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    finetune_csv_path = ensure_finetune_data(args.dataset)
    output_dir = Path(os.path.abspath(args.output_dir or os.path.join(DEFAULT_OUTPUT_ROOT, args.dataset)))
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"Device: {device}")

    print("1. Downloading Lag-Llama pretrained checkpoint (cached after first run)...")
    os.makedirs(PRETRAINED_CKPT_DIR, exist_ok=True)
    ckpt_path = hf_hub_download(
        repo_id="time-series-foundation-models/Lag-Llama",
        filename="lag-llama.ckpt",
        local_dir=PRETRAINED_CKPT_DIR,
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

    print(f"2. Loading {args.dataset} fine-tune data (train+val, leakage-safe)...")
    df = pd.read_csv(finetune_csv_path)
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    split_idx = int(len(df) * (1 - args.val_ratio))
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    print(f"   train: {len(train_df)} rows, val: {len(val_df)} rows")
    min_needed = args.context_len + args.pred_len + 1
    if len(val_df) < min_needed:
        print(f"   WARNING: val split ({len(val_df)} rows) is smaller than context_len+pred_len+1 "
              f"({min_needed}) -- validation windows may be empty. Consider a smaller --context_len "
              f"or a larger --val_ratio for small datasets (e.g. CSI300).")

    train_ds = to_gluonts_dataset(train_df["close"].values)
    val_ds = to_gluonts_dataset(val_df["close"].values)

    rope_scaling_arguments = {
        "type": "linear",
        "factor": max(1.0, (args.context_len + args.pred_len) / estimator_args["context_length"]),
    }

    print(f"3. Building estimator (warm-started from pretrained checkpoint) and fine-tuning "
          f"(epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr})...")
    estimator = LagLlamaEstimator(
        ckpt_path=ckpt_path,  # warm-starts the Lightning module from the pretrained weights
        prediction_length=args.pred_len,
        context_length=args.context_len,
        input_size=estimator_args["input_size"],
        n_layer=estimator_args["n_layer"],
        n_embd_per_head=estimator_args["n_embd_per_head"],
        n_head=estimator_args["n_head"],
        scaling=estimator_args["scaling"],
        time_feat=estimator_args["time_feat"],
        rope_scaling=rope_scaling_arguments,
        batch_size=args.batch_size,
        num_batches_per_epoch=args.num_batches_per_epoch,
        lr=args.lr,
        device=device,
        trainer_kwargs={
            "max_epochs": args.epochs,
            "accelerator": "gpu" if device.type == "cuda" else "cpu",
            "devices": 1,
            "default_root_dir": str(output_dir / "lightning_logs"),
            "enable_progress_bar": True,
            "logger": True,  # writes a CSV/TensorBoard-style log under default_root_dir for resuming/inspection
        },
    )

    predictor = estimator.train(training_data=train_ds, validation_data=val_ds, cache_data=True)

    checkpoint_path = output_dir / "predictor"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    predictor.serialize(checkpoint_path)
    print(f"\nDone. Fine-tuned predictor: {checkpoint_path}")
    print(f"Evaluate it with:\n  python eval_lagllama.py --dataset {args.dataset} "
          f"--checkpoint \"{checkpoint_path}\" --output_model_name lag-llama_finetuned")


if __name__ == "__main__":
    main()
