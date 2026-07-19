"""Controladores da Fase 2: uma instância POR CRUZAMENTO do grid.

Reuso direto da Fase 1 via o contrato `Controller.act(Observation)`:
- fixo_igual e max_pressure: idênticos, instanciados K vezes;
- fixo_proporcional: verde ∝ demanda das DUAS ruas que se cruzam ali
  (continua "injusto a favor": conhece o flow_vph declarado no grid.yaml);
- atuado_gap: mesma lógica de extensão/gap-out, com o mapa de faixas por
  estágio construído a partir das faixas reais do cruzamento (uma rua de
  1 faixa usa a mesma faixa para frente e esquerda, como a via local da
  Fase 1);
- dqn: UM modelo compartilhado + vetorizador local de cada cruzamento.
"""

from __future__ import annotations

from pathlib import Path

from stable_baselines3 import DQN

from traffic_rl.config import N_STAGES
from traffic_rl.controllers.actuated import GapActuatedController
from traffic_rl.controllers.base import Controller
from traffic_rl.controllers.fixed import FixedTimeController
from traffic_rl.controllers.max_pressure import MaxPressureController
from traffic_rl.envs.grid_env import GridTrafficEnv
from traffic_rl.envs.observation import Observation
from traffic_rl.grid_config import GridProjectConfig

GRID_BASELINES = ("fixo_igual", "fixo_proporcional", "atuado_gap", "max_pressure")


class _GridGapActuated(GapActuatedController):
    """Atuado por gap com faixas por estágio derivadas do cruzamento real."""

    def __init__(self, signal, ew_lanes: int, ns_lanes: int) -> None:
        self.min_green_s = signal.min_green_s
        self.max_green_s = 60.0
        self.threshold = 1e-6
        # layout de lane_densities: E(ew), W(ew), N(ns), S(ns)
        e0, w0 = 0, ew_lanes
        n0, s0 = 2 * ew_lanes, 2 * ew_lanes + ns_lanes

        def axis(base0: int, base1: int, n: int) -> tuple[list[int], list[int]]:
            all0 = [base0 + i for i in range(n)]
            all1 = [base1 + i for i in range(n)]
            if n == 1:  # faixa única serve frente E esquerda
                return all0 + all1, all0 + all1
            through = all0[:-1] + all1[:-1]
            left = [all0[-1], all1[-1]]
            return through, left

        ew_through, ew_left = axis(e0, w0, ew_lanes)
        ns_through, ns_left = axis(n0, s0, ns_lanes)
        self._stage_lanes = [ew_through, ew_left, ns_through, ns_left]


class GridDQNController(Controller):
    """Política DQN compartilhada, vista de UM cruzamento."""

    name = "dqn"

    def __init__(self, model: DQN, vectorizer) -> None:
        self.model = model
        self.vectorizer = vectorizer

    def act(self, obs: Observation) -> int:
        action, _ = self.model.predict(self.vectorizer(obs), deterministic=True)
        return int(action)


def make_grid_controllers(
    name: str,
    cfg: GridProjectConfig,
    env: GridTrafficEnv,
    model_path: str | Path | None = None,
) -> list[Controller]:
    """Uma instância de controlador por semáforo, na ordem de env.tls."""
    signal = cfg.env.signal
    topo = env.topo
    controllers: list[Controller] = []
    shared_model: DQN | None = None
    if name == "dqn":
        if model_path is None:
            raise ValueError("controlador dqn exige model_path")
        shared_model = DQN.load(str(model_path), device="cpu")
    for idx, st in enumerate(env.tls):
        i, j = idx // topo.C, idx % topo.C
        row, col = cfg.grid.rows[i], cfg.grid.cols[j]
        if name == "fixo_igual":
            controllers.append(FixedTimeController(signal))
        elif name == "fixo_proporcional":
            t = cfg.grid.turn_shares
            demand = [
                row.flow_vph * (t.straight + t.right),
                row.flow_vph * t.left,
                col.flow_vph * (t.straight + t.right),
                col.flow_vph * t.left,
            ]
            total = sum(demand)
            lost = N_STAGES * (signal.yellow_s + signal.all_red_s)
            effective = 120.0 - lost
            greens = [max(signal.min_green_s, effective * d / total) for d in demand]
            controllers.append(FixedTimeController(signal, greens))
        elif name == "atuado_gap":
            controllers.append(_GridGapActuated(signal, row.lanes, col.lanes))
        elif name == "max_pressure":
            controllers.append(MaxPressureController(signal))
        elif name == "dqn":
            controllers.append(GridDQNController(shared_model, st.vectorizer))
        else:
            raise ValueError(f"controlador desconhecido para grid: {name!r}")
    return controllers
