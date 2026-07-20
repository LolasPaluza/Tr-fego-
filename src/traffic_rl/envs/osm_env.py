"""Fase 4 — ambiente multi-semáforo sobre malha real importada do OSM.

Herda TODO o maquinário do ambiente de grid (máquina de estados de
segurança por semáforo, observação estruturada, recompensas locais,
retentativas de reset); o que muda é a origem dos dados:
- semáforos e faixas vêm da rede importada (`discover_controllable_tls`);
- cruzamentos não mapeáveis ao modelo de 4 estágios ficam com o programa
  default do netconvert (o agente não os toca) — ADR-019;
- a demanda por episódio vem do randomTrips (provisória até a OD).
"""

from __future__ import annotations

from pathlib import Path

import sumolib

from traffic_rl.envs.grid_env import (
    GridMinimalVectorizer,
    GridTrafficEnv,
    _reward_cfg,
    _TLSState,
)
from traffic_rl.envs.osm_demand import build_osm_routes
from traffic_rl.envs.osm_network import discover_controllable_tls, import_osm
from traffic_rl.osm_config import OsmProjectConfig
from traffic_rl.paths import generated_dir
from traffic_rl.sumo_home import ensure_sumo_home


class OsmTrafficEnv(GridTrafficEnv):
    """K semáforos controláveis de um recorte OSM real."""

    def __init__(
        self,
        cfg: OsmProjectConfig,
        use_libsumo: bool | None = None,
        tripinfo_dir: str | Path | None = None,
        label: str = "osm",
    ) -> None:
        # NÃO chama super().__init__ (que constrói o grid sintético);
        # monta os mesmos atributos a partir da rede importada.
        ensure_sumo_home()
        self.cfg = cfg
        self.topo = None  # sem topologia sintética na Fase 4
        self.use_libsumo = cfg.env.use_libsumo if use_libsumo is None else use_libsumo
        self.tripinfo_dir = Path(tripinfo_dir) if tripinfo_dir else None
        self._label = label
        self.net_file = import_osm(cfg.osm_file, generated_dir())
        net = sumolib.net.readNet(str(self.net_file))

        tables = discover_controllable_tls(self.net_file)
        if not tables:
            raise RuntimeError(
                f"nenhum cruzamento controlável (4 aproximações) em {cfg.osm_file}"
            )
        self.tls: list[_TLSState] = []
        for tls_id, table in sorted(tables.items()):
            lanes_by_approach: dict[str, list[str]] = {"E": [], "W": [], "N": [], "S": []}
            lane_caps: dict[str, float] = {}
            speed_by_approach: dict[str, float] = {}
            out_lane_count: dict[str, int] = {}
            for link in table.links:
                if link.from_lane not in lanes_by_approach[link.approach]:
                    lanes_by_approach[link.approach].append(link.from_lane)
                lane = net.getLane(link.from_lane)
                lane_caps[link.from_lane] = max(lane.getLength() / 7.5, 1.0)
                speed_by_approach[link.approach] = max(
                    speed_by_approach.get(link.approach, 0.0), lane.getSpeed()
                )
                out_lane_count[link.to_edge] = len(net.getEdge(link.to_edge).getLanes())
            state = _TLSState(
                tls_id,
                table,
                lanes_by_approach,
                lane_caps,
                speed_by_approach,
                out_lane_count,
                vectorizer=None,  # type: ignore[arg-type]
                reward_calc=None,  # type: ignore[arg-type]
            )
            state.vectorizer = GridMinimalVectorizer(
                state.queue_caps(), cfg.env.signal.max_green_s
            )
            state.reward_calc = _reward_cfg_calc(cfg)
            self.tls.append(state)

        self.n_tls = len(self.tls)
        self._conn = None
        self._episode_max_queue = 0.0
        self._arrived_cum = 0
        self._current_tripinfo: Path | None = None

    def _route_file_for(self, traffic_seed: int) -> Path:
        return build_osm_routes(
            self.net_file,
            generated_dir(),
            traffic_seed,
            self.cfg.env.episode_length_s,
            self.cfg.demand.total_vph,
        )


def _reward_cfg_calc(cfg: OsmProjectConfig):
    from traffic_rl.envs.rewards import RewardCalculator

    return RewardCalculator(_reward_cfg(cfg))
