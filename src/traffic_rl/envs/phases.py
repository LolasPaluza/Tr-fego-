"""Construção do programa semafórico de 4 estágios verdes.

Estágios (ordem canônica, ver config.STAGE_NAMES):
0. av_frente_dir    — avenida: seguir + direita (E e W simultâneos)
1. av_esquerda      — avenida: esquerda protegida (faixas dedicadas)
2. local_frente_dir — via local: seguir + direita
3. local_esquerda   — via local: esquerda protegida

Os estados são strings RYG sobre os links controlados do semáforo, derivadas
DINAMICAMENTE do .net.xml via sumolib (nada de índices mágicos): para cada
link controlado identificamos (aproximação, movimento) pelo par de arestas e
consultamos a tabela de movimentos do estágio. Transições de segurança
(amarelo 3s + vermelho total 2s) são estados derivados: links que perdem o
verde ficam 'y', depois tudo 'r'. Quem aplica os tempos é o ambiente
(wrapper), nunca o agente.
"""

from __future__ import annotations

from dataclasses import dataclass

import sumolib

from traffic_rl.config import N_STAGES
from traffic_rl.envs.network import TLS_ID, TURN_MAP

# Movimentos servidos por estágio: conjunto de (aproximação, movimento).
STAGE_MOVEMENTS: tuple[frozenset[tuple[str, str]], ...] = (
    frozenset({("E", "s"), ("E", "r"), ("W", "s"), ("W", "r")}),
    frozenset({("E", "l"), ("W", "l")}),
    frozenset({("N", "s"), ("N", "r"), ("S", "s"), ("S", "r")}),
    frozenset({("N", "l"), ("S", "l")}),
)


@dataclass(frozen=True)
class LinkInfo:
    """Um link controlado do semáforo (índice na string de estado)."""

    index: int
    approach: str  # E/W/N/S
    movement: str  # s/r/l
    from_lane: str  # ex.: "in_E_2"
    to_edge: str  # ex.: "out_S"


@dataclass(frozen=True)
class PhaseTable:
    links: tuple[LinkInfo, ...]
    green_states: tuple[str, ...]  # um estado por estágio
    yellow_states: tuple[str, ...]  # transição saindo de cada estágio
    all_red_state: str

    @property
    def n_links(self) -> int:
        return len(self.links)


def _movement_of(from_edge: str, to_edge: str) -> tuple[str, str]:
    approach = from_edge.removeprefix("in_")
    dest = to_edge.removeprefix("out_")
    for (appr, mov), d in TURN_MAP.items():
        if appr == approach and d == dest:
            return approach, mov
    raise ValueError(f"conexão não mapeada: {from_edge} -> {to_edge}")


def build_phase_table(net_file: str) -> PhaseTable:
    """Lê o .net.xml e monta a tabela de fases dos 4 estágios."""
    net = sumolib.net.readNet(net_file)
    tls = net.getTLS(TLS_ID)
    # getConnections(): lista de (inLane, outLane, linkIndex)
    raw = tls.getConnections()
    links: list[LinkInfo] = []
    for in_lane, out_lane, link_index in raw:
        from_edge = in_lane.getEdge().getID()
        to_edge = out_lane.getEdge().getID()
        approach, movement = _movement_of(from_edge, to_edge)
        links.append(
            LinkInfo(
                index=link_index,
                approach=approach,
                movement=movement,
                from_lane=in_lane.getID(),
                to_edge=to_edge,
            )
        )
    links.sort(key=lambda li: li.index)
    n = len(links)
    if n == 0:
        raise RuntimeError(f"semáforo {TLS_ID} sem links controlados em {net_file}")
    indices = [li.index for li in links]
    if indices != list(range(n)):
        raise RuntimeError(f"índices de link não contíguos: {indices}")

    greens, yellows = [], []
    for stage in range(N_STAGES):
        movements = STAGE_MOVEMENTS[stage]
        green = "".join(
            "G" if (li.approach, li.movement) in movements else "r" for li in links
        )
        yellow = "".join(
            "y" if (li.approach, li.movement) in movements else "r" for li in links
        )
        greens.append(green)
        yellows.append(yellow)
    return PhaseTable(
        links=tuple(links),
        green_states=tuple(greens),
        yellow_states=tuple(yellows),
        all_red_state="r" * n,
    )
