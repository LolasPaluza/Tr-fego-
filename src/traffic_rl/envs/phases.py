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
from traffic_rl.envs.network import TLS_ID

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


def _heading(edge) -> tuple[float, float]:
    """Vetor unitário do sentido de circulação da aresta (nó origem → destino)."""
    x0, y0 = edge.getFromNode().getCoord()
    x1, y1 = edge.getToNode().getCoord()
    dx, dy = x1 - x0, y1 - y0
    norm = (dx * dx + dy * dy) ** 0.5
    if norm == 0:
        raise ValueError(f"aresta degenerada: {edge.getID()}")
    return dx / norm, dy / norm


def _movement_of(from_edge, to_edge) -> tuple[str, str]:
    """(aproximação, movimento) por GEOMETRIA — vale para qualquer cruzamento
    em cruz (Fase 1 ou grid): a aproximação é o lado de onde o veículo vem
    (rumo leste ⇒ veio do oeste) e o movimento sai do produto vetorial entre
    os rumos de entrada e saída (negativo = direita, positivo = esquerda)."""
    hx, hy = _heading(from_edge)
    if abs(hx) >= abs(hy):
        approach = "W" if hx > 0 else "E"
    else:
        approach = "S" if hy > 0 else "N"
    ox, oy = _heading(to_edge)
    dot = hx * ox + hy * oy
    cross = hx * oy - hy * ox
    if dot > 0.7:
        movement = "s"
    elif cross < 0:
        movement = "r"
    else:
        movement = "l"
    return approach, movement


def build_phase_table(net_file: str, tls_id: str = TLS_ID) -> PhaseTable:
    """Lê o .net.xml e monta a tabela de fases dos 4 estágios do semáforo
    `tls_id` (default: o cruzamento único da Fase 1)."""
    net = sumolib.net.readNet(net_file)
    tls = net.getTLS(tls_id)
    # getConnections(): lista de (inLane, outLane, linkIndex)
    raw = tls.getConnections()
    links: list[LinkInfo] = []
    for in_lane, out_lane, link_index in raw:
        to_edge = out_lane.getEdge().getID()
        approach, movement = _movement_of(in_lane.getEdge(), out_lane.getEdge())
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
        raise RuntimeError(f"semáforo {tls_id} sem links controlados em {net_file}")
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
