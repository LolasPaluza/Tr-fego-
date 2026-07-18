#!/usr/bin/env bash
# As duas ablações completas (spec 1.3 e 1.4):
#   observação: obs_minimal vs obs_rica       (2 variantes × 5 seeds × 300k)
#   recompensa: r_espera vs r_fila vs r_pressao (3 variantes × 5 seeds × 300k)
# A variante default (obs_minimal+r_espera) é treinada do zero sob o tag da
# ablação — runs independentes do treino principal, comparação limpa.
# RETOMÁVEL como o treino principal (checkpoints por seed).
#
# Uso:      bash scripts/run_ablations.sh
# Monitorar: tail -f results/logs/ablations.log
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON=python3
mkdir -p results/logs

echo "== TrafficRL ablações (obs + recompensa) — log: results/logs/ablations.log =="
"$PYTHON" -m traffic_rl.cli ablation --type all 2>&1 | tee results/logs/ablations.log
echo "== Relatórios: results/ablations/obs/REPORT.md e results/ablations/reward/REPORT.md =="
