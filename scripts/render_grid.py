#!/usr/bin/env python3
"""Renderiza o mapa do grid e (opcional) o GIF de congestionamento.

Uso:
  python scripts/render_grid.py                          # mapa + GIF (max_pressure)
  python scripts/render_grid.py --method fixo_igual
  python scripts/render_grid.py --method dqn --model results/runs/<tag>/seed_42/best_model.zip
  python scripts/render_grid.py --grid meu_bairro.yaml --map-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, default=None)
    parser.add_argument("--method", default="max_pressure",
                        choices=["fixo_igual", "fixo_proporcional", "atuado_gap",
                                 "max_pressure", "dqn"])
    parser.add_argument("--model", type=Path, help="best_model.zip (para --method dqn)")
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--map-only", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    from traffic_rl.analysis.grid_viz import render_congestion_gif, render_grid_map
    from traffic_rl.grid_config import load_grid_config
    from traffic_rl.paths import configs_dir, results_dir

    cfg = load_grid_config(args.grid or (configs_dir() / "grid.yaml"))
    out_dir = args.out_dir or (results_dir() / "grid" / "viz")
    mapa = render_grid_map(cfg, out_dir / "mapa_grid.png")
    print("mapa:", mapa)
    if not args.map_only:
        gif = render_congestion_gif(
            cfg, args.method, out_dir / f"congestionamento_{args.method}.gif",
            model_path=args.model, traffic_seed=args.seed,
        )
        print("gif:", gif)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
