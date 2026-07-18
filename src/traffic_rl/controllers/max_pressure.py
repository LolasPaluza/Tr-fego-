"""Controlador max-pressure clássico (sem aprendizado).

A cada oportunidade de decisão (respeitado o verde mínimo), ativa o estágio
de MAIOR pressão, onde pressão do estágio = Σ sobre seus movimentos de
(veículos a montante − veículos a jusante). Varaiya (2013) provou que essa
política maximiza a vazão estabilizável em redes de semáforos — é a teoria
por trás dos melhores controladores clássicos e o baseline mais difícil de
bater. Referência completa no README.
"""

from __future__ import annotations

import numpy as np

from traffic_rl.config import SignalConfig
from traffic_rl.controllers.base import Controller
from traffic_rl.envs.observation import Observation


class MaxPressureController(Controller):
    name = "max_pressure"

    def __init__(self, signal: SignalConfig) -> None:
        self.min_green_s = signal.min_green_s

    def act(self, obs: Observation) -> int:
        if obs.time_in_stage_s < self.min_green_s:
            return obs.stage
        best = int(np.argmax(obs.stage_pressures))
        # desempate conservador: só troca se a pressão for estritamente maior
        if obs.stage_pressures[best] <= obs.stage_pressures[obs.stage]:
            return obs.stage
        return best
