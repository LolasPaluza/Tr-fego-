"""Observação estruturada e suas duas projeções vetoriais (ablação 1.3).

O ambiente sempre calcula a `Observation` COMPLETA (estrutura nomeada).
Os controladores clássicos leem os campos nomeados diretamente; o agente RL
usa um vetorizador (`obs_minimal` ou `obs_rica`) que projeta a estrutura em
um Box [0,1]. Isso mantém o contrato `Controller.act(obs)` único e evita
que baselines dependam do layout do vetor do DQN.

Normalização (máximos documentados — ver README §observações):
- fila:            capacidade da aproximação = n_faixas × (comprimento / 7.5 m)
                   (7.5 m = veículo médio 5 m + gap 2.5 m)
- densidade/faixa: idem, por faixa
- velocidade:      velocidade máxima da via da aproximação
- esperas:         teto de 300 s (2.5× o limiar de starvation), com clip em 1.0
- tempo no verde:  max_green_s
Sem normalização, entradas em escalas diferentes (fila 0-40 vs one-hot 0-1)
desequilibram os gradientes do MLP do DQN — ver README.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from traffic_rl.config import APPROACHES, N_STAGES, EnvConfig, ObsMode

WAIT_NORM_S = 300.0


@dataclass(frozen=True)
class Observation:
    """Foto instantânea do cruzamento em um passo de decisão.

    Arrays por aproximação seguem a ordem canônica APPROACHES = (E, W, N, S).
    """

    stage: int
    time_in_stage_s: float
    queues: np.ndarray  # [4] veículos parados por aproximação (contagem crua)
    lane_densities: np.ndarray  # [n_faixas_total] ocupação normalizada [0,1] por faixa
    mean_speeds_norm: np.ndarray  # [4] velocidade média / limite da via, [0,1]
    first_vehicle_waits_s: np.ndarray  # [4] espera acumulada do 1º veículo da fila (s)
    max_waits_s: np.ndarray  # [4] maior espera acumulada na aproximação (s)
    stage_pressures: np.ndarray  # [N_STAGES] pressão (montante - jusante) por estágio


class ObservationVectorizer:
    """Projeta `Observation` em vetor [0,1] conforme o modo da ablação."""

    def __init__(self, cfg: EnvConfig, mode: ObsMode | None = None) -> None:
        self.mode: ObsMode = mode or cfg.obs_mode
        net = cfg.network
        cap_per_lane = net.approach_length_m / 7.5
        lanes_by_approach = {
            "E": net.avenue_lanes,
            "W": net.avenue_lanes,
            "N": net.local_lanes,
            "S": net.local_lanes,
        }
        self._queue_caps = np.array(
            [lanes_by_approach[a] * cap_per_lane for a in APPROACHES], dtype=np.float32
        )
        self._max_green = cfg.signal.max_green_s
        self._n_lanes_total = sum(lanes_by_approach[a] for a in APPROACHES)

    @property
    def size(self) -> int:
        base = N_STAGES + 1 + len(APPROACHES)  # one-hot + tempo + filas
        if self.mode == "obs_minimal":
            return base
        return base + self._n_lanes_total + 2 * len(APPROACHES)

    def __call__(self, obs: Observation) -> np.ndarray:
        one_hot = np.zeros(N_STAGES, dtype=np.float32)
        one_hot[obs.stage] = 1.0
        time_norm = min(obs.time_in_stage_s / self._max_green, 1.0)
        queues_norm = np.clip(obs.queues / self._queue_caps, 0.0, 1.0)
        parts = [one_hot, np.array([time_norm], dtype=np.float32), queues_norm]
        if self.mode == "obs_rica":
            first_waits_norm = np.clip(obs.first_vehicle_waits_s / WAIT_NORM_S, 0.0, 1.0)
            parts += [
                np.clip(obs.lane_densities, 0.0, 1.0),
                np.clip(obs.mean_speeds_norm, 0.0, 1.0),
                first_waits_norm,
            ]
        return np.concatenate(parts).astype(np.float32)
