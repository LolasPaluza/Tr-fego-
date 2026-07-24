#!/usr/bin/env bash
# Fase 3 — treino coordenado PARALELO (PAR seeds ao mesmo tempo), retomável.
# Otimiza o tempo de relógio usando os vários núcleos. Cada seed retoma do
# checkpoint; seeds já completas retornam na hora.
#
# Uso:
#   bash scripts/run_coord_parallel.sh
#   PAR=3 SEEDS_OVERRIDE="42 123 7" CONFIG=configs/grid_coord.yaml \
#     bash scripts/run_coord_parallel.sh
set -uo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON=python3
CONFIG="${CONFIG:-configs/grid_coord.yaml}"
PAR="${PAR:-2}"
read -r -a SEEDS <<< "${SEEDS_OVERRIDE:-42 123 7 2024 777}"
mkdir -p results/logs

train_one() {
    local s="$1"
    OMP_NUM_THREADS=1 "$PYTHON" -c "
from traffic_rl.grid_config import load_grid_config
from traffic_rl.training.grid_train import train_grid_seed
train_grid_seed(load_grid_config('$CONFIG'), seed=$s)
print('seed $s COMPLETO')
" >> "results/logs/coord_seed_${s}.log" 2>&1
}

echo "== treino coordenado paralelo: seeds ${SEEDS[*]} | $PAR por vez =="
running=0
for s in "${SEEDS[@]}"; do
    echo "-> seed $s (log: results/logs/coord_seed_${s}.log)"
    train_one "$s" &
    running=$((running + 1))
    if [ "$running" -ge "$PAR" ]; then
        wait -n 2>/dev/null || wait
        running=$((running - 1))
    fi
done
wait
echo "== COORD PARALELO COMPLETO =="
