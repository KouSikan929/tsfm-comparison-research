"""
Fine-tunes Kronos-small on one dataset's leakage-safe train+val period
(data/finetune_data/<dataset>_finetune.csv -- see prep_finetune_data.py,
run automatically if missing).

Reuses the vendored repo's own working fine-tuning pipeline
(models/Kronos/finetune_csv/finetune_base_model.py) as-is -- this script
only generates the YAML config it expects and invokes it, rather than
reimplementing training. The tokenizer is kept frozen (pretrained, not
fine-tuned) by default; only the predictor ("basemodel") is fine-tuned,
warm-started from NeoQuasar/Kronos-small.

Output: a checkpoint directory loadable via
`Kronos.from_pretrained(<checkpoint_dir>)`, at
<output_dir>/basemodel/best_model -- pass that path to
eval_kronos.py --checkpoint to evaluate it with the exact same harness
used for the zero-shot results.

All the parameters likely to change between the local smoke test and the
A40 production run are CLI flags -- see --help. Defaults below are the
LOCAL SMOKE TEST defaults (1 epoch, small batch); see
docs/finetuning_server_guide.md for recommended A40 production values.

Usage (from the kronos conda env):
    python finetune_kronos.py --dataset SP500 --epochs 1
"""
import argparse
import os
import subprocess
import sys

import yaml

SCRIPT_DIR = os.path.dirname(__file__)
FINETUNE_DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data", "finetune_data")
DEFAULT_OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "..", "data", "finetuned_checkpoints", "kronos")
KRONOS_FINETUNE_CSV_DIR = os.path.join(SCRIPT_DIR, "..", "models", "Kronos", "finetune_csv")


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
    parser.add_argument("--epochs", type=int, default=1, help="basemodel_epochs (smoke test default: 1)")
    parser.add_argument("--batch_size", type=int, default=8, help="smoke test default: 8; A40 production: 32-160")
    parser.add_argument("--lr", type=float, default=1e-6, help="predictor_learning_rate")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--device_id", type=int, default=0, help="GPU index when --device cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context_len", type=int, default=400, help="lookback_window -- matches scripts/common/windowing.py's CONTEXT_LEN")
    parser.add_argument("--pred_len", type=int, default=20, help="predict_window -- matches scripts/common/windowing.py's PRED_LEN")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                         help="split within the leakage-safe finetune CSV (never touches TEST_START+ data). "
                              "NOTE: val split must contain at least context_len+pred_len+1 rows or the val "
                              "DataLoader will crash (ValueError: __len__() should return >= 0) -- with the "
                              "default 400/20 window that's 421 rows; CSI300 (~1043 pre-test rows, the smallest "
                              "dataset) may need a smaller --context_len or a larger --val_ratio.")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=2, help="keep low on Windows -- see the num_workers/paging-file gotcha in SERVER_SETUP.md")
    parser.add_argument("--pretrained_tokenizer", default="NeoQuasar/Kronos-Tokenizer-base")
    parser.add_argument("--pretrained_predictor", default="NeoQuasar/Kronos-small")
    parser.add_argument("--output_dir", default=None, help="default: data/finetuned_checkpoints/kronos/<dataset>")
    args = parser.parse_args()

    finetune_csv_path = ensure_finetune_data(args.dataset)
    output_dir = os.path.abspath(args.output_dir or os.path.join(DEFAULT_OUTPUT_ROOT, args.dataset))
    os.makedirs(output_dir, exist_ok=True)

    config = {
        "data": {
            "data_path": finetune_csv_path,
            "lookback_window": args.context_len,
            "predict_window": args.pred_len,
            "max_context": 512,
            "clip": 5.0,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": 0.0,
        },
        "training": {
            "basemodel_epochs": args.epochs,
            "batch_size": args.batch_size,
            "log_interval": 10,
            "num_workers": args.num_workers,
            "seed": args.seed,
            "predictor_learning_rate": args.lr,
            "tokenizer_learning_rate": 2e-4,  # unused (tokenizer frozen) but required by the config schema
            "adam_beta1": 0.9,
            "adam_beta2": 0.95,
            "adam_weight_decay": 0.1,
            "accumulation_steps": 1,
        },
        "model_paths": {
            "pretrained_tokenizer": args.pretrained_tokenizer,
            "pretrained_predictor": args.pretrained_predictor,
            "exp_name": args.dataset,
            "base_path": os.path.dirname(output_dir),
            "base_save_path": output_dir,
            # tokenizer is frozen (train_tokenizer: false below), so point
            # straight at the pretrained tokenizer rather than an
            # auto-generated "finetuned/best_model" path that would never exist.
            "finetuned_tokenizer": args.pretrained_tokenizer,
            "tokenizer_save_name": "tokenizer",
            "basemodel_save_name": "basemodel",
        },
        "experiment": {
            "name": f"kronos_finetune_{args.dataset}",
            "description": f"Fine-tune Kronos-small on {args.dataset} (leakage-safe train+val period)",
            "use_comet": False,
            "train_tokenizer": False,
            "train_basemodel": True,
            "skip_existing": False,
            "pre_trained_tokenizer": True,
            "pre_trained_predictor": True,
        },
        "device": {
            "use_cuda": args.device == "cuda",
            "device_id": args.device_id,
        },
    }

    config_path = os.path.join(output_dir, "config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
    print(f"Generated config: {config_path}")

    print(f"\nStarting Kronos fine-tuning on {args.dataset} "
          f"(epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr})...\n")
    result = subprocess.run(
        [sys.executable, "finetune_base_model.py", "--config", config_path],
        cwd=os.path.abspath(KRONOS_FINETUNE_CSV_DIR),
    )
    if result.returncode != 0:
        print(f"\nFine-tuning FAILED (exit code {result.returncode})")
        sys.exit(result.returncode)

    checkpoint_path = os.path.join(output_dir, "basemodel", "best_model")
    print(f"\nDone. Fine-tuned checkpoint: {checkpoint_path}")
    print(f"Evaluate it with:\n  python eval_kronos.py --dataset {args.dataset} "
          f"--checkpoint \"{checkpoint_path}\" --output_model_name kronos_finetuned")


if __name__ == "__main__":
    main()
