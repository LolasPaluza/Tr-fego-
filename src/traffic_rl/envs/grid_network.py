"""Gerador do grid NxM (Fase 2) + topologia navegável para o roteador.

Geometria: ruas horizontais (rows, leste-oeste) × verticais (cols,
norte-sul); cada cruzamento interno é semaforizado com id `n_{i}_{j}`
(i = índice da row, j = índice da col). Stubs de borda são as entradas e
saídas de veículos.

Convenção de ids de aresta (a topologia inteira é derivável do id):
  h_{i}_{k}_E  — rua horizontal i, segmento k (0..C), sentido leste
  h_{i}_{k}_W  — idem, sentido oeste
  v_{j}_{k}_S  — rua vertical j, segmento k (0..R), sentido sul
  v_{j}_{k}_N  — idem, sentido norte
Segmento k liga a posição k à k+1 na lista de nós da rua
(stub — cruzamentos — stub).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from traffic_rl.grid_config import GridSpec, StreetSpec
from traffic_rl.sumo_home import netconvert_binary

MOVEMENTS = ("s", "r", "l")


@dataclass(frozen=True)
class EntryPoint:
    street: StreetSpec
    heading: str  # E/W/S/N — sentido de circulação
    entry_edge: str


class GridTopology:
    """Regras de navegação do grid — usada pelo gerador e pelo roteador."""

    def __init__(self, spec: GridSpec) -> None:
        self.spec = spec
        self.R = len(spec.rows)
        self.C = len(spec.cols)

    # ---------------------------------------------------------------- nós

    def node_xy(self, i: int, j: int) -> tuple[float, float]:
        return j * self.spec.block_length_m, -i * self.spec.block_length_m

    def tls_ids(self) -> list[str]:
        return self.spec.tls_ids()

    # ------------------------------------------------------------- arestas

    def street_of_edge(self, edge_id: str) -> StreetSpec:
        kind, idx, _k, _d = self._parse(edge_id)
        return self.spec.rows[idx] if kind == "h" else self.spec.cols[idx]

    @staticmethod
    def _parse(edge_id: str) -> tuple[str, int, int, str]:
        kind, idx, k, d = edge_id.split("_")
        return kind, int(idx), int(k), d

    def entry_points(self) -> list[EntryPoint]:
        pts: list[EntryPoint] = []
        for i, row in enumerate(self.spec.rows):
            pts.append(EntryPoint(row, "E", f"h_{i}_0_E"))
            pts.append(EntryPoint(row, "W", f"h_{i}_{self.C}_W"))
        for j, col in enumerate(self.spec.cols):
            pts.append(EntryPoint(col, "S", f"v_{j}_0_S"))
            pts.append(EntryPoint(col, "N", f"v_{j}_{self.R}_N"))
        return pts

    def next_moves(self, edge_id: str) -> dict[str, str] | None:
        """Movimentos possíveis ao fim da aresta: {s|r|l: próxima aresta}.
        Retorna None se a aresta termina num stub de borda (saída do grid)."""
        kind, idx, k, d = self._parse(edge_id)
        R, C = self.R, self.C
        if kind == "h" and d == "E":
            if k >= C:
                return None
            i, j = idx, k
            return {"s": f"h_{i}_{j + 1}_E", "r": f"v_{j}_{i + 1}_S", "l": f"v_{j}_{i}_N"}
        if kind == "h" and d == "W":
            if k <= 0:
                return None
            i, j = idx, k - 1
            return {"s": f"h_{i}_{j}_W", "r": f"v_{j}_{i}_N", "l": f"v_{j}_{i + 1}_S"}
        if kind == "v" and d == "S":
            if k >= R:
                return None
            i, j = k, idx
            return {"s": f"v_{j}_{i + 1}_S", "r": f"h_{i}_{j}_W", "l": f"h_{i}_{j + 1}_E"}
        if kind == "v" and d == "N":
            if k <= 0:
                return None
            i, j = k - 1, idx
            return {"s": f"v_{j}_{i}_N", "r": f"h_{i}_{j + 1}_E", "l": f"h_{i}_{j}_W"}
        raise ValueError(f"aresta desconhecida: {edge_id}")

    def incoming_edges(self, i: int, j: int) -> dict[str, str]:
        """Arestas que CHEGAM ao cruzamento n_i_j, por aproximação (E/W/N/S).
        Aproximação = lado de onde o veículo vem (W = vindo do oeste)."""
        return {
            "W": f"h_{i}_{j}_E",
            "E": f"h_{i}_{j + 1}_W",
            "N": f"v_{j}_{i}_S",
            "S": f"v_{j}_{i + 1}_N",
        }


def _nodes_xml(topo: GridTopology) -> str:
    spec = topo.spec
    lines = ["<nodes>"]
    for i in range(topo.R):
        for j in range(topo.C):
            x, y = topo.node_xy(i, j)
            lines.append(
                f'    <node id="n_{i}_{j}" x="{x}" y="{y}" '
                f'type="traffic_light" tl="n_{i}_{j}"/>'
            )
    b = spec.boundary_length_m
    for i in range(topo.R):
        _, y = topo.node_xy(i, 0)
        x_last, _ = topo.node_xy(i, topo.C - 1)
        lines.append(f'    <node id="rb_{i}_W" x="{-b}" y="{y}" type="priority"/>')
        lines.append(f'    <node id="rb_{i}_E" x="{x_last + b}" y="{y}" type="priority"/>')
    for j in range(topo.C):
        x, _ = topo.node_xy(0, j)
        _, y_last = topo.node_xy(topo.R - 1, j)
        lines.append(f'    <node id="cb_{j}_N" x="{x}" y="{b}" type="priority"/>')
        lines.append(f'    <node id="cb_{j}_S" x="{x}" y="{y_last - b}" type="priority"/>')
    lines.append("</nodes>")
    return "\n".join(lines) + "\n"


def _edges_xml(topo: GridTopology) -> str:
    lines = ["<edges>"]

    def edge(eid: str, frm: str, to: str, street: StreetSpec) -> None:
        lines.append(
            f'    <edge id="{eid}" from="{frm}" to="{to}" '
            f'numLanes="{street.lanes}" speed="{street.speed_ms:.2f}"/>'
        )

    for i, row in enumerate(topo.spec.rows):
        positions = [f"rb_{i}_W"] + [f"n_{i}_{j}" for j in range(topo.C)] + [f"rb_{i}_E"]
        for k in range(len(positions) - 1):
            edge(f"h_{i}_{k}_E", positions[k], positions[k + 1], row)
            edge(f"h_{i}_{k}_W", positions[k + 1], positions[k], row)
    for j, col in enumerate(topo.spec.cols):
        positions = [f"cb_{j}_N"] + [f"n_{i}_{j}" for i in range(topo.R)] + [f"cb_{j}_S"]
        for k in range(len(positions) - 1):
            edge(f"v_{j}_{k}_S", positions[k], positions[k + 1], col)
            edge(f"v_{j}_{k}_N", positions[k + 1], positions[k], col)
    lines.append("</edges>")
    return "\n".join(lines) + "\n"


def _connections_xml(topo: GridTopology) -> str:
    """Conexões faixa-a-faixa em cada cruzamento (mesma política da Fase 1:
    faixa 0 = direita+frente, intermediárias = frente, última = só esquerda
    quando há 3 faixas; com 1 faixa, todos os movimentos)."""
    lines = ["<connections>"]
    for i in range(topo.R):
        for j in range(topo.C):
            incoming = topo.incoming_edges(i, j)
            for _approach, in_edge in incoming.items():
                moves = topo.next_moves(in_edge)
                if moves is None:  # não ocorre em nós internos
                    continue
                in_street = topo.street_of_edge(in_edge)
                n = in_street.lanes
                for mov, out_edge in moves.items():
                    out_lanes = topo.street_of_edge(out_edge).lanes
                    if mov == "r":
                        pairs = [(0, 0)]
                    elif mov == "s":
                        pairs = [(lane, lane) for lane in range(max(n - 1, 1))]
                    else:  # esquerda: última faixa -> faixa mais à esquerda
                        pairs = [(n - 1, out_lanes - 1)]
                    for fl, tl in pairs:
                        lines.append(
                            f'    <connection from="{in_edge}" to="{out_edge}" '
                            f'fromLane="{fl}" toLane="{tl}"/>'
                        )
    lines.append("</connections>")
    return "\n".join(lines) + "\n"


def build_grid_network(spec: GridSpec, out_dir: Path) -> Path:
    """Escreve nod/edg/con do grid e roda netconvert. Retorna o .net.xml."""
    topo = GridTopology(spec)
    out_dir.mkdir(parents=True, exist_ok=True)
    nod = out_dir / "grid.nod.xml"
    edg = out_dir / "grid.edg.xml"
    con = out_dir / "grid.con.xml"
    net_file = out_dir / "grid.net.xml"
    nod.write_text(_nodes_xml(topo), encoding="utf-8")
    edg.write_text(_edges_xml(topo), encoding="utf-8")
    con.write_text(_connections_xml(topo), encoding="utf-8")
    cmd = [
        netconvert_binary(),
        "--node-files", str(nod),
        "--edge-files", str(edg),
        "--connection-files", str(con),
        "--no-turnarounds", "true",
        "--tls.yellow.time", "3",
        "--output-file", str(net_file),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"netconvert (grid) falhou:\n{result.stderr}")
    return net_file
