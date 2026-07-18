"""Baselines de tempo fixo.

- fixo_igual: verdes iguais para todos os estágios — o "semáforo burro" de SP.
- fixo_proporcional: verdes proporcionais à demanda média esperada do cenário.
  Este baseline é deliberadamente "injusto" A FAVOR dele: recebe de graça o
  conhecimento perfeito da demanda média (que na prática exigiria medição em
  campo), enquanto o agente RL precisa inferi-la das observações. Se o RL o
  bate, o mérito é real — ver README.
"""

from __future__ import annotations

from traffic_rl.config import N_STAGES, ScenarioConfig, SignalConfig
from traffic_rl.controllers.base import Controller
from traffic_rl.envs.observation import Observation

DEFAULT_CYCLE_S = 120.0


class FixedTimeController(Controller):
    """Cicla os estágios com tempos de verde pré-definidos."""

    name = "fixo_igual"

    def __init__(self, signal: SignalConfig, green_times_s: list[float] | None = None) -> None:
        lost_time = N_STAGES * (signal.yellow_s + signal.all_red_s)
        effective = DEFAULT_CYCLE_S - lost_time
        if green_times_s is None:
            green_times_s = [effective / N_STAGES] * N_STAGES
        if len(green_times_s) != N_STAGES:
            raise ValueError(f"esperados {N_STAGES} tempos de verde")
        self.green_times_s = [max(g, signal.min_green_s) for g in green_times_s]

    def act(self, obs: Observation) -> int:
        if obs.time_in_stage_s >= self.green_times_s[obs.stage]:
            return (obs.stage + 1) % N_STAGES
        return obs.stage


class ProportionalFixedController(FixedTimeController):
    name = "fixo_proporcional"

    def __init__(self, signal: SignalConfig, scenario: ScenarioConfig) -> None:
        t = scenario.turn_shares
        # demanda atendida por estágio (veíc/h por sentido; simétrico nos 2 sentidos)
        stage_demand = [
            scenario.avenue_flow_vph * (t.straight + t.right),
            scenario.avenue_flow_vph * t.left,
            scenario.local_flow_vph * (t.straight + t.right),
            scenario.local_flow_vph * t.left,
        ]
        total = sum(stage_demand)
        lost_time = N_STAGES * (signal.yellow_s + signal.all_red_s)
        effective = DEFAULT_CYCLE_S - lost_time
        greens = [max(signal.min_green_s, effective * d / total) for d in stage_demand]
        super().__init__(signal, greens)
