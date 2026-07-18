#!/usr/bin/env bash
# Treino completo: 5 seeds × 300k timesteps no pico_assimetrico.
# RETOMÁVEL: se interrompido, rode de novo — cada seed continua do último
# checkpoint (results/runs/<tag>/seed_<s>/checkpoints/).
#
# Uso:
#   bash scripts/run_full_training.sh            # paralelo se houver núcleos
#   PARALLEL=0 bash scripts/run_full_training.sh # força sequencial
#
# Monitorar:
#   tail -f results/logs/train_seed_42.log
#   tensorboard --logdir results/runs
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON=python3
SEEDS=(42 123 7 2024 777)
LOGDIR=results/logs
mkdir -p "$LOGDIR"

NCORES=$(nproc 2>/dev/null || echo 4)
# cada treino usa ~1.5 núcleos (SUMO + torch); paralelo total se couberem todos
PARALLEL="${PARALLEL:-$([ "$NCORES" -ge 8 ] && echo 1 || echo 0)}"

echo "== TrafficRL treino completo: ${#SEEDS[@]} seeds × 300k passos =="
echo "   núcleos: $NCORES | paralelo: $PARALLEL | logs: $LOGDIR/"

if [ "$PARALLEL" = "1" ]; then
    pids=()
    for seed in "${SEEDS[@]}"; do
        echo "-> seed $seed (background, log: $LOGDIR/train_seed_${seed}.log)"
        nohup "$PYTHON" -m traffic_rl.cli train --seeds "$seed" \
            > "$LOGDIR/train_seed_${seed}.log" 2>&1 &
        pids+=($!)
    done
    fail=0
    for i in "${!pids[@]}"; do
        wait "${pids[$i]}" || { echo "!! seed ${SEEDS[$i]} falhou"; fail=1; }
    done
    [ "$fail" = "0" ] || exit 1
else
    for seed in "${SEEDS[@]}"; do
        echo "-> seed $seed (sequencial, log: $LOGDIR/train_seed_${seed}.log)"
        "$PYTHON" -m traffic_rl.cli train --seeds "$seed" \
            2>&1 | tee "$LOGDIR/train_seed_${seed}.log"
    done
fi

echo "== Treino completo. Próximo: bash scripts/run_full_evaluation.sh =="
