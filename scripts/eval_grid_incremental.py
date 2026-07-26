#!/usr/bin/env python3
"""Avaliação incremental do grid — resiliente a reinícios do ambiente.

Avalia método por método e salva um CSV parcial por método. Se o processo
morrer (container reiniciado), basta rodar de novo: os métodos já avaliados
são pulados. Junta tudo no final e gera o relatório.

Uso:
  python scripts/eval_grid_incremental.py --grid configs/grid_coord.yaml
  python scripts/eval_grid_incremental.py --grid configs/grid_coord.yaml --evento
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--evento", action="store_true",
                    help="aplica o evento imprevisto no meio do episódio")
    args = ap.parse_args()

    import pandas as pd

    from traffic_rl.analysis.report import generate_report, load_eval_curves
    from traffic_rl.evaluation.grid_runner import evaluate_grid_controller
    from traffic_rl.grid_config import load_grid_config
    from traffic_rl.paths import configs_dir, results_dir, runs_dir
    from traffic_rl.training.grid_train import find_grid_best_models, grid_run_tag
    from traffic_rl.training.train import _git_hash

    grid_path = args.grid or (configs_dir() / "grid_coord.yaml")
    base = load_grid_config(grid_path)
    overrides: dict = {}
    if args.evento:
        overrides["grid"] = {
            **base.grid.model_dump(),
            "event": {"at_s": 1800, "avenue_factor": 0.4, "local_factor": 2.5},
        }
    if args.episodes:
        overrides["eval"] = {**base.eval.model_dump(), "n_episodes": args.episodes}
    cfg = load_grid_config(grid_path, overrides or None)

    tag = grid_run_tag(cfg)
    models = find_grid_best_models(tag)
    out_dir = args.out or (results_dir() / ("grid_evento" if args.evento else "grid_coord"))
    parts = out_dir / "parciais"
    parts.mkdir(parents=True, exist_ok=True)
    print(f"grid: {grid_path.name} | evento: {args.evento} | seeds DQN: {sorted(models)}",
          flush=True)

    jobs: list[tuple[str, str, Path | None]] = [
        (m, m, None) for m in ("fixo_igual", "fixo_proporcional", "atuado_gap", "max_pressure")
    ]
    jobs += [(f"dqn_seed{s}", "dqn", p) for s, p in sorted(models.items())]

    frames = []
    for job_id, method, model_path in jobs:
        part = parts / f"{job_id}.csv"
        if part.exists():
            print(f"[pulando] {job_id} (já avaliado)", flush=True)
            frames.append(pd.read_csv(part))
            continue
        print(f"[avaliando] {job_id}", flush=True)
        df = evaluate_grid_controller(cfg, method, model_path, cfg.eval.n_episodes)
        df["dqn_seed"] = job_id.removeprefix("dqn_seed") if method == "dqn" else pd.NA
        df.to_csv(part, index=False)  # salva ANTES de seguir: resiliente
        frames.append(df)

    full = pd.concat(frames, ignore_index=True)
    full.to_csv(out_dir / "metrics.csv", index=False)
    curves = load_eval_curves(runs_dir() / tag)
    suffix = " — grid COORDENADO + evento imprevisto" if args.evento else " — grid COORDENADO"
    rep = generate_report(full, out_dir, eval_curves=curves or None,
                          git_hash=_git_hash(), title_suffix=suffix)
    print("\n" + full.groupby("method")[
        ["espera_media_s", "espera_p95_s", "throughput"]
    ].mean().round(1).to_string())
    print(f"\nrelatório: {rep}")
    print("AVALIACAO COMPLETA")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
