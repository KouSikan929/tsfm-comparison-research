#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Override defaults by appending CLI flags, e.g. --output data/fusion_runs/run02.
python scripts/train_joint_fusion.py \
  --device cuda --batch-size 1 --accumulation 8 \
  --epochs 10 --samples-per-epoch 2000 \
  --context 400 --horizon 20 --stride 10 \
  --output data/fusion_runs/a40_joint "$@"
