#!/usr/bin/env bash
# Fase 2 completa na máquina local: treino multi-seed do grid + comparação.
# Edite configs/grid.yaml (ou passe GRID=meu_grid.yaml) para descrever SUAS
# ruas. Retomável por checkpoint, como o treino da Fase 1.
#
# Uso:       bash scripts/run_grid_full.sh
#            GRID=configs/meu_bairro.yaml bash scripts/run_grid_full.sh
# Monitorar: tail -f results/logs/grid_train.log
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON=python3
GRID="${GRID:-configs/grid.yaml}"
mkdir -p results/logs

echo "== TrafficRL Fase 2: treino do grid ($GRID) =="
"$PYTHON" -m traffic_rl.cli grid-train --grid "$GRID" 2>&1 | tee results/logs/grid_train.log

echo "== TrafficRL Fase 2: comparação no grid =="
"$PYTHON" -m traffic_rl.cli grid-compare --grid "$GRID" 2>&1 | tee results/logs/grid_compare.log
echo "== Relatório: results/grid/REPORT.md =="
