#!/usr/bin/env bash
# Roda a avaliação incremental em laço até completar. Cada execução avança o
# que conseguir antes de um eventual reinício; o progresso por episódio fica
# salvo em results/<out>/parciais/.
set -u
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON=python3
ARGS="${*:-}"
for i in $(seq 1 60); do
    out=$("$PYTHON" scripts/eval_grid_incremental.py $ARGS 2>&1)
    echo "$out" | tail -n 25
    if echo "$out" | grep -q "AVALIACAO COMPLETA"; then
        echo "== concluída na tentativa $i =="
        exit 0
    fi
    sleep 3
done
echo "== esgotou tentativas =="
