"""Controle atuado por gap — o que semáforos "inteligentes" comerciais fazem.

Lógica clássica de gap-out: o verde é estendido enquanto seguem chegando
veículos nas faixas servidas pelo estágio atual, até um verde máximo; quando
abre um "gap" (nenhum veículo detectado), passa ao próximo estágio com
demanda. A detecção usa a ocupação por faixa da `Observation`, que é
exatamente a medida de um detector de área (E2) cobrindo a faixa — ver
docs/DECISOES.md sobre essa equivalência.
"""

from __future__ import annotations

from traffic_rl.config import N_STAGES, EnvConfig
from traffic_rl.controllers.base import Controller
from traffic_rl.envs.observation import Observation

# Faixas servidas por estágio, como índices no vetor lane_densities
# (layout: E0..E{n-1}, W0.., N0.., S0.. — ordem canônica APPROACHES).


def _stage_lane_indices(cfg: EnvConfig) -> list[list[int]]:
    av, loc = cfg.network.avenue_lanes, cfg.network.local_lanes
    e0, w0 = 0, av
    n0, s0 = 2 * av, 2 * av + loc
    through_e = [e0 + i for i in range(av - 1)]  # faixas de frente/direita
    through_w = [w0 + i for i in range(av - 1)]
    left_e, left_w = [e0 + av - 1], [w0 + av - 1]  # faixa dedicada à esquerda
    local_n = [n0 + i for i in range(loc)]
    local_s = [s0 + i for i in range(loc)]
    return [
        through_e + through_w,  # 0: avenida frente/direita
        left_e + left_w,  # 1: avenida esquerda
        local_n + local_s,  # 2: local frente/direita
        local_n + local_s,  # 3: local esquerda (mesma faixa única)
    ]


class GapActuatedController(Controller):
    name = "atuado_gap"

    def __init__(
        self,
        cfg: EnvConfig,
        detection_threshold: float = 1e-6,
        max_green_s: float | None = None,
    ) -> None:
        self.min_green_s = cfg.signal.min_green_s
        self.max_green_s = max_green_s if max_green_s is not None else 60.0
        self.threshold = detection_threshold
        self._stage_lanes = _stage_lane_indices(cfg)

    def _has_demand(self, obs: Observation, stage: int) -> bool:
        return any(obs.lane_densities[i] > self.threshold for i in self._stage_lanes[stage])

    def act(self, obs: Observation) -> int:
        stage = obs.stage
        if obs.time_in_stage_s < self.min_green_s:
            return stage
        if obs.time_in_stage_s < self.max_green_s and self._has_demand(obs, stage):
            return stage  # estende o verde: ainda chegam veículos
        # gap-out (ou verde máximo): próximo estágio com demanda
        for k in range(1, N_STAGES):
            candidate = (stage + k) % N_STAGES
            if self._has_demand(obs, candidate):
                return candidate
        return stage  # cruzamento vazio: mantém
