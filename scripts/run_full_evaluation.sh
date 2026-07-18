#!/usr/bin/env bash
# Protocolo completo de avaliação: (4 baselines + DQN best-per-seed)
# × 4 cenários × 20 episódios + REPORT.md final em results/.
#
# Pré-requisito: run_full_training.sh concluído (best_model.zip por seed).
# Uso:      bash scripts/run_full_evaluation.sh
# Monitorar: tail -f results/logs/evaluation.log
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON=python3
mkdir -p results/logs

echo "== TrafficRL avaliação completa (log: results/logs/evaluation.log) =="
"$PYTHON" -m traffic_rl.cli compare 2>&1 | tee results/logs/evaluation.log
echo "== Relatório final: results/REPORT.md =="
