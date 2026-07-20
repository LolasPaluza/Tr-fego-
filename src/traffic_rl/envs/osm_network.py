"""Fase 4 — importação de malha real (OpenStreetMap) para o SUMO.

`import_osm` converte um recorte .osm em rede SUMO com semáforos; a
extração do recorte real (ex.: bairro de SP) acontece fora do sandbox —
ver docs/PENDENTE_LOCAL.md §Fase 4.

`discover_controllable_tls` identifica os cruzamentos que o nosso modelo de
4 estágios consegue controlar: 4 aproximações mapeáveis aos eixos (E/W/N/S,
pela geometria) e todos os estágios liberando algum movimento. Cruzamentos
irregulares (3 vias, 5+ vias, ângulos estranhos) ficam com o programa
semafórico default do netconvert — controle deles é trabalho futuro
(ADR-019).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import sumolib

from traffic_rl.config import APPROACHES
from traffic_rl.envs.phases import PhaseTable, build_phase_table
from traffic_rl.sumo_home import netconvert_binary


def import_osm(osm_file: str | Path, out_dir: Path, name: str = "osm") -> Path:
    """Converte .osm -> .net.xml urbano (vias de carro, semáforos inferidos)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    net_file = out_dir / f"{name}.net.xml"
    cmd = [
        netconvert_binary(),
        "--osm-files", str(osm_file),
        "--output-file", str(net_file),
        # limpeza urbana padrão
        "--geometry.remove", "true",
        "--junctions.join", "true",
        "--tls.guess-signals", "true",
        "--tls.discard-simple", "true",
        "--no-turnarounds", "true",
        "--remove-edges.isolated", "true",
        # só malha de veículos (sem calçadas/ciclovias como arestas próprias)
        "--keep-edges.by-vclass", "passenger",
        "--tls.yellow.time", "3",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"netconvert (OSM) falhou:\n{result.stderr[-2000:]}")
    return net_file


def discover_controllable_tls(net_file: str | Path) -> dict[str, PhaseTable]:
    """{tls_id: PhaseTable} dos cruzamentos compatíveis com os 4 estágios."""
    net = sumolib.net.readNet(str(net_file))
    controllable: dict[str, PhaseTable] = {}
    for tls in net.getTrafficLights():
        tls_id = tls.getID()
        try:
            table = build_phase_table(str(net_file), tls_id)
        except Exception:
            continue  # geometria não mapeável (ângulos/valores degenerados)
        approaches = {li.approach for li in table.links}
        if approaches != set(APPROACHES):
            continue  # não é um cruzamento de 4 aproximações
        if any("G" not in state for state in table.green_states):
            continue  # algum estágio não liberaria movimento algum
        controllable[tls_id] = table
    return controllable
