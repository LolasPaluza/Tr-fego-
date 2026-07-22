#!/usr/bin/env bash
# Treino generalista (multi-cenário) — 5 seeds × 300k, mesmo orçamento do
# especialista, para a comparação especialista × generalista ser justa.
# Retomável por checkpoint. Uso: bash scripts/run_generalist.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON=python3
mkdir -p results/logs
"$PYTHON" - <<'PY' 2>&1 | tee -a results/logs/generalist_train.log
from traffic_rl.config import load_config
from traffic_rl.training.generalist import train_generalist_all_seeds
cfg = load_config("configs/default.yaml")
train_generalist_all_seeds(cfg)
print("GENERALISTA — TREINO COMPLETO")
PY
