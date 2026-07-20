#!/usr/bin/env python3
"""Fase 4 — prepara um recorte OSM e valida o pipeline de ponta a ponta.

Uso:
  python scripts/osm_prepare.py --osm data/osm/pinheiros.osm
  python scripts/osm_prepare.py --osm data/osm/pinheiros.osm --vph 3000

Faz: importa a malha (netconvert), lista os cruzamentos controláveis pelo
modelo de 4 estágios, gera demanda de teste (randomTrips) e roda um episódio
de sanidade com o controlador max-pressure em cada cruzamento controlável.
Como obter o recorte real de SP: docs/PENDENTE_LOCAL.md §Fase 4.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--osm", type=Path, required=True, help="arquivo .osm do recorte")
    parser.add_argument("--vph", type=float, default=2000.0,
                        help="viagens/hora no recorte (demanda provisória)")
    parser.add_argument("--episode", type=float, default=300.0,
                        help="duração do episódio de sanidade (s)")
    args = parser.parse_args()

    from traffic_rl.controllers.max_pressure import MaxPressureController
    from traffic_rl.envs.osm_env import OsmTrafficEnv
    from traffic_rl.envs.osm_network import discover_controllable_tls, import_osm
    from traffic_rl.osm_config import OsmProjectConfig
    from traffic_rl.paths import generated_dir

    print(f"== importando {args.osm} ==")
    net = import_osm(args.osm, generated_dir())
    print(f"rede: {net}")
    tables = discover_controllable_tls(net)
    print(f"cruzamentos semaforizados controláveis (4 aproximações): {len(tables)}")
    for tls_id, table in sorted(tables.items()):
        print(f"  - {tls_id}: {table.n_links} links")
    if not tables:
        print("nenhum cruzamento controlável — verifique o recorte (precisa de "
              "cruzamentos em cruz com semáforo).", file=sys.stderr)
        return 1

    cfg = OsmProjectConfig(
        osm_file=str(args.osm),
        env={"episode_length_s": args.episode},
        demand={"total_vph": args.vph},
    )
    env = OsmTrafficEnv(cfg)
    try:
        controllers = [MaxPressureController(cfg.env.signal) for _ in range(env.n_tls)]
        observations, info = env.reset(traffic_seed=1)
        done = False
        while not done:
            actions = [c.act(o) for c, o in zip(controllers, observations, strict=True)]
            observations, _r, done, info = env.step(actions)
        print(f"episódio de sanidade ok: {info['sim_time']:.0f}s simulados, "
              f"{info['arrived_cum']} veículos concluíram, "
              f"fila máx {info['episode_max_queue']:.0f}")
    finally:
        env.close()
    print("== pipeline Fase 4 validado para este recorte ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
