"""
Phase evaluation for a LoRA-fine-tuned TimesFM 2.5 checkpoint (from
finetune_timesfm.py), on the exact same shared windows
(data/eval_windows_<dataset>.csv) and long-format results schema
(scripts/common/results_io.py) used by every other model's eval script --
so its output is directly comparable via aggregate_results.py.

Separate script from eval_timesfm.py (not a --checkpoint flag on it)
because the model-loading stack is genuinely different: eval_timesfm.py
uses the zero-shot-only google/timesfm-2.5-200m-pytorch checkpoint via the
raw torch module in models/timesfm/src/timesfm; this script uses
google/timesfm-2.5-200m-transformers via HuggingFace `transformers` + a
`peft` LoRA adapter -- see finetune_timesfm.py's module docstring for why.
Everything that CAN be shared (windowing, results schema, metrics) is
shared; only the model-loading/inference calls differ.

Usage (from the timesfm conda env, after `pip install transformers peft accelerate`):
    python eval_timesfm_finetuned.py --dataset SP500 --checkpoint <path from finetune_timesfm.py> --output_model_name timesfm_finetuned
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

COMMON_ROOT = os.path.join(os.path.dirname(__file__), "common")
sys.path.append(os.path.abspath(COMMON_ROOT))
from windowing import load_windows, data_path  # noqa: E402
from results_io import save_results, compute_metrics  # noqa: E402

# full_predictions[..., 0] = point/decode channel, [..., 1..9] = quantiles
# 0.1..0.9 -- same convention verified for the zero-shot src/timesfm path in
# eval_timesfm.py, confirmed identical here via a dummy forward pass.
Q10_IDX = 1
Q90_IDX = 9
PATCH_LEN = 32


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="e.g. SP500, SSE, SZSE, Nikkei225")
    parser.add_argument("--checkpoint", required=True, help="Path to the LoRA adapter dir from finetune_timesfm.py")
    parser.add_argument("--model_id", default="google/timesfm-2.5-200m-transformers",
                         help="Base checkpoint the adapter was fine-tuned from -- must match finetune_timesfm.py's --model_id")
    parser.add_argument("--output_model_name", default="timesfm_finetuned")
    args = parser.parse_args()

    from peft import PeftModel
    from transformers import TimesFm2_5ModelForPrediction

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"1. Loading {args.dataset} data + shared windows...")
    df = pd.read_csv(data_path(args.dataset))
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    windows = load_windows(args.dataset)
    context_len = int(windows.iloc[0].context_end_idx - windows.iloc[0].context_start_idx + 1)
    pred_len = int(windows.iloc[0].target_end_idx - windows.iloc[0].target_start_idx + 1)
    if context_len % PATCH_LEN != 0:
        context_len = ((context_len // PATCH_LEN) + 1) * PATCH_LEN
    print(f"   {len(windows)} windows to evaluate, context_len={context_len} (patch-rounded), pred_len={pred_len}")

    print(f"2. Loading base model ({args.model_id}) + LoRA adapter ({args.checkpoint})...")
    base_model = TimesFm2_5ModelForPrediction.from_pretrained(args.model_id, torch_dtype=torch.bfloat16, device_map=device)
    model = PeftModel.from_pretrained(base_model, args.checkpoint)
    model.eval()

    all_rows = []
    t0 = time.time()
    for i, w in windows.iterrows():
        context_slice = df.loc[w.context_start_idx:w.context_end_idx]
        target_slice = df.loc[w.target_start_idx:w.target_end_idx]
        context_values = context_slice["close"].to_numpy(dtype=np.float32)
        context_tensor = torch.tensor(context_values, dtype=torch.bfloat16, device=device).unsqueeze(0)

        with torch.no_grad():
            out = model(past_values=context_tensor, forecast_context_len=context_len)

        pred_mean = out.mean_predictions[0, :pred_len].float().cpu().numpy()
        q10 = out.full_predictions[0, :pred_len, Q10_IDX].float().cpu().numpy()
        q90 = out.full_predictions[0, :pred_len, Q90_IDX].float().cpu().numpy()

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
