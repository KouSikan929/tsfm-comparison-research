"""
Phase 1 rolling-window evaluation for Lag-Llama on SP500 data.

Runs Lag-Llama over every window in data/eval_windows.csv (shared across all
four models), with genuine per-sample quantile forecasts (num_samples draws,
not internally averaged away like Kronos's predict()), and writes a
long-format results CSV via scripts/common/results_io.py.

The predictor is compiled once (context_length/prediction_length are fixed
across all windows by construction -- see scripts/common/windowing.py) and
reused for every window, which is much cheaper than rebuilding it 27 times.

Usage (from the lag-llama conda env):
    python eval_lagllama.py --dataset SP500
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

# See run_lagllama_smoke_test.py for why these two patches are needed:
# (1) newer setuptools dropped pkg_resources, which pytorch_lightning still
#     imports (fixed by pinning setuptools<81 in the lag-llama env, not here);
# (2) PyTorch 2.6 defaults torch.load to weights_only=True, which rejects this
#     checkpoint's embedded gluonts class object.
_orig_torch_load = torch.load


def _torch_load_weights_only_false(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_weights_only_false

from pathlib import Path  # noqa: E402

from huggingface_hub import hf_hub_download  # noqa: E402
from gluonts.dataset.pandas import PandasDataset  # noqa: E402
from gluonts.evaluation import make_evaluation_predictions  # noqa: E402
from gluonts.torch.model.predictor import PyTorchPredictor  # noqa: E402

LAG_LLAMA_ROOT = os.path.join(os.path.dirname(__file__), "..", "models", "lag-llama")
sys.path.append(os.path.abspath(LAG_LLAMA_ROOT))
from lag_llama.gluon.estimator import LagLlamaEstimator  # noqa: E402

COMMON_ROOT = os.path.join(os.path.dirname(__file__), "common")
sys.path.append(os.path.abspath(COMMON_ROOT))
from windowing import load_windows, data_path  # noqa: E402
from results_io import save_results, compute_metrics  # noqa: E402

CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "checkpoints", "lag-llama")
NUM_SAMPLES = 100


def build_predictor(ckpt_path, prediction_length, context_length, device, num_samples):
    ckpt = torch.load(ckpt_path, map_location=device)
    estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

    rope_scaling_arguments = {
        "type": "linear",
        "factor": max(1.0, (context_length + prediction_length) / estimator_args["context_length"]),
    }

    estimator = LagLlamaEstimator(
        ckpt_path=ckpt_path,
        prediction_length=prediction_length,
        context_length=context_length,
        input_size=estimator_args["input_size"],
        n_layer=estimator_args["n_layer"],
        n_embd_per_head=estimator_args["n_embd_per_head"],
        n_head=estimator_args["n_head"],
        scaling=estimator_args["scaling"],
        time_feat=estimator_args["time_feat"],
        rope_scaling=rope_scaling_arguments,
        batch_size=1,
        num_parallel_samples=num_samples,
        device=device,
    )

    lightning_module = estimator.create_lightning_module()
    transformation = estimator.create_transformation()
    predictor = estimator.create_predictor(transformation, lightning_module)
    return predictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="e.g. SP500, SSE, SZSE, Nikkei225")
    parser.add_argument("--seed", type=int, default=None,
                         help="Optional seed for the num_samples predictive-distribution draws. "
                              "Unset by default -- prior runs in this project were unseeded, so results "
                              "won't be bit-identical to earlier runs even with a seed now.")
    parser.add_argument("--checkpoint", default=None,
                         help="Path to a fine-tuned predictor directory (from finetune_lagllama.py, "
                              "<output_dir>/predictor) to evaluate instead of the zero-shot pretrained "
                              "checkpoint. Loaded via PyTorchPredictor.deserialize() -- bypasses "
                              "build_predictor() entirely, so context_len/pred_len are whatever the "
                              "checkpoint was fine-tuned with (read from the serialized predictor, not "
                              "from this script's own windows).")
    parser.add_argument("--output_model_name", default="lag-llama",
                         help="Model name results are saved under (data/results/<dataset>/<name>_results.csv). "
                              "Use e.g. lag-llama_finetuned when evaluating a fine-tuned checkpoint.")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"2. Loading {args.dataset} data + shared windows...")
    df = pd.read_csv(data_path(args.dataset))
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    windows = load_windows(args.dataset)
    context_len = int(windows.iloc[0].context_end_idx - windows.iloc[0].context_start_idx + 1)
    pred_len = int(windows.iloc[0].target_end_idx - windows.iloc[0].target_start_idx + 1)
    print(f"   {len(windows)} windows to evaluate, context_len={context_len}, pred_len={pred_len}, num_samples={NUM_SAMPLES}")

    if args.checkpoint:
        print(f"1/3. Loading fine-tuned predictor from {args.checkpoint} ...")
        predictor = PyTorchPredictor.deserialize(Path(args.checkpoint))
    else:
        print("1. Downloading Lag-Llama checkpoint (cached after first run)...")
        os.makedirs(CKPT_DIR, exist_ok=True)
        ckpt_path = hf_hub_download(
            repo_id="time-series-foundation-models/Lag-Llama",
            filename="lag-llama.ckpt",
            local_dir=CKPT_DIR,
        )
        print("3. Building predictor (once, reused across all windows)...")
        predictor = build_predictor(ckpt_path, pred_len, context_len, device, NUM_SAMPLES)

    all_rows = []
    t0 = time.time()
    for i, w in windows.iterrows():
        full_slice = df.loc[w.context_start_idx:w.target_end_idx]
        target_slice = df.loc[w.target_start_idx:w.target_end_idx]

        full_series = full_slice["close"].astype("float32").reset_index(drop=True)
        full_series.index = pd.period_range(start="2000-01-01", periods=len(full_series), freq="D")
        ds = PandasDataset(full_series.to_frame(name="close"), target="close")

        forecast_it, _ = make_evaluation_predictions(dataset=ds, predictor=predictor, num_samples=NUM_SAMPLES)
        forecast = list(forecast_it)[0]

        pred_mean = forecast.mean
        q10 = np.quantile(forecast.samples, 0.1, axis=0)
        q90 = np.quantile(forecast.samples, 0.9, axis=0)

        for step, (_, row) in enumerate(target_slice.iterrows(), start=1):
            all_rows.append({
                "model": args.output_model_name,
                "window_id": w.window_id,
                "target_date": row["timestamps"].date().isoformat(),
                "step_ahead": step,
                "actual_close": row["close"],
                "pred_close": pred_mean[step - 1],
                "pred_q10": q10[step - 1],
                "pred_q90": q90[step - 1],
            })

        elapsed = time.time() - t0
        print(f"   window {int(w.window_id) + 1}/{len(windows)} done ({elapsed:.1f}s elapsed)")

    out_path = save_results(all_rows, args.output_model_name, args.dataset)
    print(f"\nSaved {len(all_rows)} rows to {out_path}")

    metrics = compute_metrics(pd.DataFrame(all_rows))
    print("\nAggregate metrics (all windows):")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
