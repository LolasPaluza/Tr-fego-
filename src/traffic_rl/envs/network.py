"""Geração paramétrica da rede viária (.net.xml) a partir da config.

Um único cruzamento em cruz: avenida leste-oeste (3 faixas/sentido, 60 km/h,
faixa da esquerda dedicada à conversão) × via local norte-sul (1 faixa/sentido,
40 km/h). A construção é uma função de `NetworkConfig` → na Fase 2 este módulo
vira um gerador de grid NxM reutilizando as mesmas primitivas.

Convenções de nomes (usadas em todo o projeto):
- nós: C (centro, semaforizado), N/S/E/W (extremidades)
- arestas: in_X (aproximação de X para C) e out_X (saída de C para X)
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from traffic_rl.config import NetworkConfig
from traffic_rl.sumo_home import netconvert_binary

TLS_ID = "C"

# (aproximação de origem, movimento) -> aproximação de destino.
# Movimentos: s=seguir, r=direita, l=esquerda. Sentidos de tráfego:
# in_W anda para leste, in_E para oeste, in_N para sul, in_S para norte.
TURN_MAP: dict[tuple[str, str], str] = {
    ("W", "s"): "E", ("W", "r"): "S", ("W", "l"): "N",
    ("E", "s"): "W", ("E", "r"): "N", ("E", "l"): "S",
    ("N", "s"): "S", ("N", "r"): "W", ("N", "l"): "E",
    ("S", "s"): "N", ("S", "r"): "E", ("S", "l"): "W",
}

AVENUE_APPROACHES = ("E", "W")
LOCAL_APPROACHES = ("N", "S")


def _node_xml(net: NetworkConfig) -> str:
    d = net.approach_length_m
    return f"""<nodes>
    <node id="C" x="0" y="0" type="traffic_light" tl="{TLS_ID}"/>
    <node id="N" x="0" y="{d}" type="priority"/>
    <node id="S" x="0" y="-{d}" type="priority"/>
    <node id="E" x="{d}" y="0" type="priority"/>
    <node id="W" x="-{d}" y="0" type="priority"/>
</nodes>
"""


def _edge_xml(net: NetworkConfig) -> str:
    rows = []
    for approach in AVENUE_APPROACHES + LOCAL_APPROACHES:
        if approach in AVENUE_APPROACHES:
            lanes, speed = net.avenue_lanes, net.avenue_speed_ms
        else:
            lanes, speed = net.local_lanes, net.local_speed_ms
        rows.append(
            f'    <edge id="in_{approach}" from="{approach}" to="C" '
            f'numLanes="{lanes}" speed="{speed:.2f}"/>'
        )
        rows.append(
            f'    <edge id="out_{approach}" from="C" to="{approach}" '
            f'numLanes="{lanes}" speed="{speed:.2f}"/>'
        )
    return "<edges>\n" + "\n".join(rows) + "\n</edges>\n"


def _connection_xml(net: NetworkConfig) -> str:
    """Conexões faixa-a-faixa explícitas.

    Avenida (3 faixas, índice 0 = direita): faixa 0 = direita+frente,
    faixa 1 = frente, faixa 2 = SÓ esquerda (dedicada — cria o conflito de
    fase protegida que motiva o estágio de conversão).
    Via local (1 faixa): todos os movimentos da faixa 0.
    """
    rows = []
    for approach in AVENUE_APPROACHES:
        to_s = TURN_MAP[(approach, "s")]
        to_r = TURN_MAP[(approach, "r")]
        to_l = TURN_MAP[(approach, "l")]
        rows += [
            f'    <connection from="in_{approach}" to="out_{to_r}" fromLane="0" toLane="0"/>',
            f'    <connection from="in_{approach}" to="out_{to_s}" fromLane="0" toLane="0"/>',
            f'    <connection from="in_{approach}" to="out_{to_s}" fromLane="1" toLane="1"/>',
            f'    <connection from="in_{approach}" to="out_{to_l}" fromLane="2" toLane="0"/>',
        ]
    for approach in LOCAL_APPROACHES:
        to_s = TURN_MAP[(approach, "s")]
        to_r = TURN_MAP[(approach, "r")]
        to_l = TURN_MAP[(approach, "l")]
        # Esquerda a partir da via local entra na faixa mais à esquerda da avenida.
        left_target_lane = net.avenue_lanes - 1
        rows += [
            f'    <connection from="in_{approach}" to="out_{to_r}" fromLane="0" toLane="0"/>',
            f'    <connection from="in_{approach}" to="out_{to_s}" fromLane="0" toLane="0"/>',
            f'    <connection from="in_{approach}" to="out_{to_l}" fromLane="0" '
            f'toLane="{left_target_lane}"/>',
        ]
    return "<connections>\n" + "\n".join(rows) + "\n</connections>\n"


def build_network(net: NetworkConfig, out_dir: Path) -> Path:
    """Escreve nod/edg/con e roda netconvert. Retorna o caminho do .net.xml."""
    out_dir.mkdir(parents=True, exist_ok=True)
    nod = out_dir / "intersection.nod.xml"
    edg = out_dir / "intersection.edg.xml"
    con = out_dir / "intersection.con.xml"
    net_file = out_dir / "intersection.net.xml"
    nod.write_text(_node_xml(net), encoding="utf-8")
    edg.write_text(_edge_xml(net), encoding="utf-8")
    con.write_text(_connection_xml(net), encoding="utf-8")
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
        raise RuntimeError(f"netconvert falhou:\n{result.stderr}")
    return net_file
