"""Fase 4 — demanda provisória sobre a malha OSM.

Enquanto a matriz Origem-Destino calibrada não entra (etapa da máquina
local — ver ADR-019), a demanda usa o `randomTrips.py` do próprio SUMO:
viagens com origem/destino aleatórios na malha, roteadas pelo duarouter,
com volume total configurável e determinísticas na seed. Serve para treinar
e comparar métodos NA GEOMETRIA REAL; a distribuição espacial realista das
viagens vem com a OD.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from traffic_rl.sumo_home import ensure_sumo_home


def build_osm_routes(
    net_file: str | Path,
    out_dir: Path,
    traffic_seed: int,
    episode_length_s: float,
    total_vph: float,
) -> Path:
    """Gera o .rou.xml do episódio via randomTrips (determinístico na seed)."""
    home = ensure_sumo_home()
    random_trips = home / "tools" / "randomTrips.py"
    if not random_trips.exists():
        raise FileNotFoundError(f"randomTrips.py não encontrado em {random_trips}")
    out_dir.mkdir(parents=True, exist_ok=True)
    route_file = out_dir / f"osm_seed{traffic_seed}.rou.xml"
    trips_file = out_dir / f"osm_seed{traffic_seed}.trips.xml"
    period = 3600.0 / max(total_vph, 1.0)  # segundos entre inserções
    cmd = [
        sys.executable, str(random_trips),
        "-n", str(net_file),
        "-o", str(trips_file),
        "-r", str(route_file),  # roteia com duarouter e valida conectividade
        "--seed", str(traffic_seed),
        "--begin", "0",
        "--end", str(int(episode_length_s)),
        "--period", f"{period:.4f}",
        "--fringe-factor", "5",  # mais viagens atravessando o recorte
        "--validate",
        "--vehicle-class", "passenger",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not route_file.exists():
        raise RuntimeError(f"randomTrips falhou:\n{result.stderr[-2000:]}")
    return route_file
