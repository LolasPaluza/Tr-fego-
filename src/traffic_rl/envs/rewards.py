"""As três funções de recompensa da ablação 1.4 + componente anti-starvation.

- r_espera:  -Δ(espera total acumulada) / 100 — recompensa padrão do sumo-rl
             (diff-waiting-time). Positiva quando a espera total DIMINUI.
- r_fila:    -(soma das filas) / 100 no passo de decisão.
- r_pressao: -(Σ_movimentos |pressão|) / 100, pressão = veíc. a montante
             − veíc. a jusante (por movimento) — a grandeza que o controle
             max-pressure (Varaiya, 2013) minimiza. Ver README.
- anti-starvation (configurável): penalidade fixa por aproximação com veículo
  esperando além do limiar (120 s). Sem ela, otimizar só a média permite
  sacrificar a via local indefinidamente — o clássico problema de "otimizar
  a média e ignorar a cauda". Discussão no README.

Escala /100: mantém as três recompensas na mesma ordem de grandeza (~[-5, 5])
para que os mesmos hiperparâmetros do DQN sirvam à ablação inteira.
"""

from __future__ import annotations

import numpy as np

from traffic_rl.config import EnvConfig, RewardMode
from traffic_rl.envs.observation import Observation

SCALE = 100.0


class RewardCalculator:
    def __init__(self, cfg: EnvConfig, mode: RewardMode | None = None) -> None:
        self.mode: RewardMode = mode or cfg.reward_mode
        self.starvation = cfg.starvation
        self._prev_total_wait: float | None = None

    def reset(self) -> None:
        self._prev_total_wait = None

    def __call__(self, obs: Observation, total_accumulated_wait_s: float) -> float:
        if self.mode == "r_espera":
            if self._prev_total_wait is None:
                base = 0.0
            else:
                base = (self._prev_total_wait - total_accumulated_wait_s) / SCALE
            self._prev_total_wait = total_accumulated_wait_s
        elif self.mode == "r_fila":
            base = -float(np.sum(obs.queues)) / SCALE
        elif self.mode == "r_pressao":
            base = -float(np.sum(np.abs(obs.stage_pressures))) / SCALE
        else:  # pragma: no cover - pydantic impede
            raise ValueError(f"recompensa desconhecida: {self.mode}")

        if self.starvation.enabled:
            n_starved = int(np.sum(obs.max_waits_s > self.starvation.threshold_s))
            base -= self.starvation.penalty * n_starved
        return base
