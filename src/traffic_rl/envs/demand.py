"""Geração paramétrica de demanda (.rou.xml) — nunca escrita à mão.

Chegadas estocásticas: cada fluxo usa o atributo `probability` do SUMO
(processo de Bernoulli por segundo ⇒ chegadas binomiais, aproximação de
Poisson para taxas baixas). A variação entre episódios vem da seed da
simulação (`sumo --seed`), então o MESMO .rou.xml serve para todos os
episódios de um cenário — reprodutível e barato.

Veículos pesados: 10% de caminhões (aceleração menor) nos cenários de pico,
via `truck_share` no cenário.

O id do fluxo codifica origem e tipo (ex.: `E_s_carro` → veículos
`E_s_carro.0`, ...), o que permite à análise separar espera da avenida vs
via local a partir do tripinfo.
"""

from __future__ import annotations

from pathlib import Path

from traffic_rl.config import ScenarioConfig
from traffic_rl.envs.network import AVENUE_APPROACHES, LOCAL_APPROACHES, TURN_MAP

VTYPES_XML = """    <vType id="carro" accel="2.6" decel="4.5" length="5.0" minGap="2.5"
           maxSpeed="33.33" speedFactor="normc(1.0,0.1,0.8,1.2)"/>
    <vType id="caminhao" accel="1.0" decel="3.5" length="12.0" minGap="3.0"
           maxSpeed="25.0" speedFactor="normc(0.95,0.05,0.8,1.1)"/>
"""


def _movement_shares(scenario: ScenarioConfig) -> dict[str, float]:
    t = scenario.turn_shares
    return {"s": t.straight, "r": t.right, "l": t.left}


def build_routes(scenario: ScenarioConfig, out_dir: Path) -> Path:
    """Escreve {cenario}.rou.xml com fluxos por (aproximação × movimento × tipo)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    route_file = out_dir / f"{scenario.name}.rou.xml"
    shares = _movement_shares(scenario)
    lines: list[str] = ["<routes>", VTYPES_XML.rstrip("\n")]

    for approach in AVENUE_APPROACHES + LOCAL_APPROACHES:
        base_vph = (
            scenario.avenue_flow_vph if approach in AVENUE_APPROACHES else scenario.local_flow_vph
        )
        for movement, share in shares.items():
            dest = TURN_MAP[(approach, movement)]
            edges = f"in_{approach} out_{dest}"
            for vtype, frac in (
                ("carro", 1.0 - scenario.truck_share),
                ("caminhao", scenario.truck_share),
            ):
                vph = base_vph * share * frac
                if vph <= 0:
                    continue
                prob = vph / 3600.0
                lines.append(
                    f'    <flow id="{approach}_{movement}_{vtype}" type="{vtype}" '
                    f'begin="0" end="172800" probability="{prob:.6f}" '
                    f'departLane="best" departSpeed="max">'
                    f'<route edges="{edges}"/></flow>'
                )
    lines.append("</routes>")
    route_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return route_file


def road_class_of_vehicle(veh_id: str) -> str:
    """Classe viária ('avenida' | 'local') a partir do id do veículo."""
    approach = veh_id.split("_", 1)[0]
    return "avenida" if approach in AVENUE_APPROACHES else "local"
