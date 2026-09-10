"""
Fine-tunes TimesFM 2.5 (LoRA) on one dataset's leakage-safe train+val period
(data/finetune_data/<dataset>_finetune.csv -- see prep_finetune_data.py, run
automatically if missing).

IMPORTANT: this uses a DIFFERENT checkpoint family than the zero-shot
pipeline. eval_timesfm.py loads google/timesfm-2.5-200m-pytorch via the raw
torch module in models/timesfm/src/timesfm (forecast()/decode() are
inference-only, no training path -- confirmed in
docs/methodology_experimental_setup.md section 6). Fine-tuning instead uses
google/timesfm-2.5-200m-transformers via HuggingFace `transformers` +
`peft` LoRA -- the same underlying weights, repackaged for a trainable API.
Both checkpoints are Google's own official releases of the same model.

Reuses the vendored repo's own dataset classes
(models/timesfm/timesfm-forecasting/examples/finetuning/finetune_lora.py's
TimeSeriesRandomWindowDataset / TimeSeriesLastWindowDataset) via import,
rather than reimplementing them -- adapted here only in how the train/val
series are split (the vendored script's own train() function passes the
SAME full series list to both the random-window train set and the
last-window val set, since it was designed for many independent store
series where overlap between one store's random window and another's last
window isn't a concern; for our single-series case we split the series
into disjoint train/val portions first, so no training window can overlap
the validation window).

Output: a PEFT LoRA adapter directory (via model.save_pretrained()),
loadable with `peft.PeftModel.from_pretrained(base_model, <dir>)` -- pass
that path to eval_timesfm_finetuned.py --checkpoint to evaluate it.

All the parameters likely to change between the local smoke test and the A40
production run are CLI flags -- see --help. Defaults below are the LOCAL
SMOKE TEST defaults (1 epoch, few random windows); see
docs/finetuning_server_guide.md for recommended A40 production values.

Usage (from the timesfm conda env, after `pip install transformers peft accelerate`):
    python finetune_timesfm.py --dataset SP500 --epochs 1
"""
import argparse
import logging
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(__file__)
FINETUNE_DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data", "finetune_data")
DEFAULT_OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "..", "data", "finetuned_checkpoints", "timesfm")
FINETUNE_LORA_DIR = os.path.join(SCRIPT_DIR, "..", "models", "timesfm", "timesfm-forecasting", "examples", "finetuning")
sys.path.append(os.path.abspath(FINETUNE_LORA_DIR))
from finetune_lora import TimeSeriesRandomWindowDataset, TimeSeriesLastWindowDataset  # noqa: E402


def ensure_finetune_data(dataset: str) -> str:
    path = os.path.abspath(os.path.join(FINETUNE_DATA_DIR, f"{dataset}_finetune.csv"))
    if not os.path.exists(path):
        print(f"[prep] {dataset}_finetune.csv not found, generating it first...")
        subprocess.run(
            [sys.executable, os.path.join(SCRIPT_DIR, "prep_finetune_data.py"), "--dataset", dataset],
            check=True,
        )
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="e.g. SP500, SSE, SZSE, Nikkei225")
    parser.add_argument("--model_id", default="google/timesfm-2.5-200m-transformers",
                         help="HF checkpoint for the trainable (transformers+peft) path -- "
                              "NOT the same repo as eval_timesfm.py's zero-shot google/timesfm-2.5-200m-pytorch")
    parser.add_argument("--epochs", type=int, default=1, help="smoke test default: 1")
    parser.add_argument("--batch_size", type=int, default=4, help="smoke test default: 4; A40 production: 16-32")
    parser.add_argument("--num_samples", type=int, default=40,
                         help="number of random training windows drawn per 'epoch' (there's no fixed epoch "
                              "size for a single series, unlike the other two models -- this stands in for "
                              "it). Smoke test default: 40 (~10 batches at batch_size=4); A40 production: "
                              "2000-5000")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora_r", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=8)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context_len", type=int, default=400, help="matches scripts/common/windowing.py's CONTEXT_LEN")
    parser.add_argument("--pred_len", type=int, default=20, help="matches scripts/common/windowing.py's PRED_LEN (horizon_len)")
    parser.add_argument("--val_ratio", type=float, default=0.2,
                         help="fraction of the leakage-safe finetune CSV (chronological tail) held out as "
                              "validation -- kept disjoint from training windows, see module docstring")
    parser.add_argument("--output_dir", default=None, help="default: data/finetuned_checkpoints/timesfm/<dataset>")
    args = parser.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import TimesFm2_5ModelForPrediction

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    finetune_csv_path = ensure_finetune_data(args.dataset)
    output_dir = os.path.abspath(args.output_dir or os.path.join(DEFAULT_OUTPUT_ROOT, args.dataset))
    os.makedirs(output_dir, exist_ok=True)

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    logger.info("Using device: %s", device)

    logger.info("Loading %s data...", args.dataset)
    df = pd.read_csv(finetune_csv_path)
    close = df["close"].values.astype(np.float32)
    split_idx = int(len(close) * (1 - args.val_ratio))
    train_series, val_series = close[:split_idx], close[split_idx:]
    logger.info("train: %d rows, val: %d rows", len(train_series), len(val_series))
    min_needed = args.context_len + args.pred_len
    if len(val_series) < min_needed:
        logger.warning("val split (%d rows) is smaller than context_len+pred_len (%d) -- the validation "
                        "set will be empty. Consider a smaller --context_len or a larger --val_ratio for "
                        "small datasets (e.g. CSI300).", len(val_series), min_needed)

    logger.info("Loading model: %s", args.model_id)
    model = TimesFm2_5ModelForPrediction.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map=device,
    )
    # TimesFM's patch size is 32 -- the HF transformers forward pass requires
    # past_values to reshape evenly into patches (view(batch, -1, 32)) and,
    # unlike the zero-shot src/timesfm path (which auto-rounds internally in
    # compile()), does NOT round this for you. Round up to the nearest
    # multiple of 32 here so --context_len 400 (this project's default)
    # doesn't crash with a reshape error.
    patch_len = 32
    context_len = min(args.context_len, model.config.context_length)
    if context_len % patch_len != 0:
        rounded = ((context_len // patch_len) + 1) * patch_len
        logger.info("context_len=%d is not a multiple of the patch size (%d); rounding up to %d",
                    context_len, patch_len, rounded)
        context_len = rounded

    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, target_modules="all-linear",
        lora_dropout=args.lora_dropout, bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_ds = TimeSeriesRandomWindowDataset(
        [train_series], context_len, args.pred_len, num_samples=args.num_samples, seed=args.seed,
    )
    val_ds = TimeSeriesLastWindowDataset([val_series], context_len, args.pred_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    logger.info("Train samples: %d (%d batches) | Val samples: %d", len(train_ds), len(train_loader), len(val_ds))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs * len(train_loader)))

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for context, target_vals in train_loader:
            context, target_vals = context.to(device), target_vals.to(device)
            outputs = model(past_values=context, future_values=target_vals, forecast_context_len=context_len)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_train_loss = epoch_loss / max(n_batches, 1)

        model.eval()
        val_loss, val_batches = 0.0, 0
        with torch.no_grad():
            for context, target_vals in val_loader:
                context, target_vals = context.to(device), target_vals.to(device)
                outputs = model(past_values=context, future_values=target_vals, forecast_context_len=context_len)
                val_loss += outputs.loss.item()
                val_batches += 1
        avg_val_loss = val_loss / max(val_batches, 1) if val_batches > 0 else float("nan")

        logger.info("Epoch %d/%d (%d steps) -- train loss: %.4f, val loss: %.4f",
                     epoch, args.epochs, n_batches, avg_train_loss, avg_val_loss)

        if val_batches == 0 or avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(output_dir)
            logger.info("  saved adapter -> %s", output_dir)

    print(f"\nDone. Fine-tuned LoRA adapter: {output_dir}")
    print(f"Evaluate it with:\n  python eval_timesfm_finetuned.py --dataset {args.dataset} "
          f"--checkpoint \"{output_dir}\" --output_model_name timesfm_finetuned")


if __name__ == "__main__":
    main()
