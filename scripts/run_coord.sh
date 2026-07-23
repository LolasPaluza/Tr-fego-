#!/usr/bin/env bash
# Fase 3 — treino coordenado do grid (5 seeds), retomável.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON=python3
mkdir -p results/logs
"$PYTHON" - <<'PY' 2>&1 | tee -a results/logs/coord_train.log
from traffic_rl.grid_config import load_grid_config
from traffic_rl.training.grid_train import train_grid_all_seeds
cfg = load_grid_config("configs/grid_coord.yaml")
train_grid_all_seeds(cfg)
print("COORDENADO — TREINO COMPLETO")
PY
