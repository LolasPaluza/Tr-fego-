"""Demanda do grid: roteador próprio por caminhada aleatória.

Cada rua declarada no grid.yaml injeta veículos pelos DOIS extremos, com
chegadas Bernoulli por segundo (taxa = flow_vph/3600 — o mesmo modelo
binomial da Fase 1, aqui materializado em veículos explícitos porque cada
veículo precisa de uma rota completa). Em cada cruzamento o veículo segue em
frente/direita/esquerda pelas frações de `turn_shares`, até sair do grid.

O arquivo é gerado POR EPISÓDIO (determinístico na seed de tráfego): rotas e
horários de partida saem de `numpy.default_rng(seed)`, então o mesmo par
(grid.yaml, seed) reproduz exatamente o mesmo tráfego em qualquer máquina.

Id do veículo codifica classe e rua de entrada para a análise de justiça:
  {av|loc}_{slug_da_rua}_{tipo}.{n}   ex.: av_avpaulista_carro.17
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from traffic_rl.envs.demand import VTYPES_XML
from traffic_rl.envs.grid_network import GridTopology
from traffic_rl.grid_config import GridSpec

_CLASS_PREFIX = {"avenida": "av", "local": "loc"}


def grid_road_class_of_vehicle(veh_id: str) -> str:
    """Classe viária a partir do id (`av_...`/`loc_...`)."""
    return "avenida" if veh_id.startswith("av_") else "local"


def _sample_route(topo: GridTopology, entry_edge: str, rng: np.random.Generator) -> list[str]:
    shares = topo.spec.turn_shares
    probs = np.array([shares.straight, shares.right, shares.left])
    route = [entry_edge]
    edge = entry_edge
    guard = 0
    while (moves := topo.next_moves(edge)) is not None:
        movement = rng.choice(["s", "r", "l"], p=probs)
        edge = moves[movement]
        route.append(edge)
        guard += 1
        if guard > 10 * (topo.R + topo.C) * (topo.R + topo.C):
            # caminhada aleatória pode circular; corta seguindo em frente
            while (m2 := topo.next_moves(edge)) is not None:
                edge = m2["s"]
                route.append(edge)
            break
    return route


def build_grid_routes(
    spec: GridSpec,
    out_dir: Path,
    traffic_seed: int,
    episode_length_s: float,
) -> Path:
    """Gera o .rou.xml do episódio (determinístico em traffic_seed)."""
    topo = GridTopology(spec)
    rng = np.random.default_rng(traffic_seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    route_file = out_dir / f"grid_seed{traffic_seed}.rou.xml"

    vehicles: list[tuple[float, str, str, str]] = []  # (depart, id, vtype, edges)
    counters: dict[str, int] = {}
    horizon = int(episode_length_s)
    for entry in topo.entry_points():
        street = entry.street
        p = street.flow_vph / 3600.0
        truck_share = (
            spec.truck_share_avenida if street.street_class == "avenida"
            else spec.truck_share_local
        )
        # chegadas Bernoulli(p) por segundo do horizonte
        arrivals = np.flatnonzero(rng.random(horizon) < p)
        for t in arrivals:
            vtype = "caminhao" if rng.random() < truck_share else "carro"
            prefix = f"{_CLASS_PREFIX[street.street_class]}_{street.slug}_{vtype}"
            n = counters.get(prefix, 0)
            counters[prefix] = n + 1
            route = _sample_route(topo, entry.entry_edge, rng)
            vehicles.append((float(t), f"{prefix}.{n}", vtype, " ".join(route)))

    vehicles.sort(key=lambda v: v[0])
    lines = ["<routes>", VTYPES_XML.rstrip("\n")]
    for depart, veh_id, vtype, edges in vehicles:
        lines.append(
            f'    <vehicle id="{veh_id}" type="{vtype}" depart="{depart:.1f}" '
            f'departLane="best" departSpeed="max">'
            f'<route edges="{edges}"/></vehicle>'
        )
    lines.append("</routes>")
    route_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return route_file
