#!/usr/bin/env bash
# Guardião da sessão cloud: a cada N segundos, copia best models e curvas
# para artifacts/ e pusha — proteção contra reciclagem do sandbox.
# Uso: bash scripts/cloud_snapshot_loop.sh [intervalo_s]   (default 600)
set -u
cd "$(dirname "$0")/.."
INTERVAL="${1:-600}"

while true; do
    sleep "$INTERVAL"
    changed=0
    for d in results/runs/*/seed_*/; do
        [ -d "$d" ] || continue
        tag=$(basename "$(dirname "$d")")
        seed=$(basename "$d")
        dest="artifacts/$tag/$seed"
        mkdir -p "$dest"
        for f in best_model.zip evaluations.npz eval_log.csv config.yaml git_hash.txt; do
            if [ -f "$d/$f" ] && ! cmp -s "$d/$f" "$dest/$f" 2>/dev/null; then
                cp "$d/$f" "$dest/$f"
                changed=1
            fi
        done
    done
    if [ "$changed" = "1" ]; then
        git add artifacts >/dev/null 2>&1
        git -c user.name="Claude" -c user.email="noreply@anthropic.com" \
            commit -q -m "artifacts: snapshot automático $(date -u +%H:%M) UTC" \
            >/dev/null 2>&1 && git push -q >/dev/null 2>&1
    fi
done
