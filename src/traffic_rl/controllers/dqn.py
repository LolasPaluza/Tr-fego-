"""Adaptador do agente DQN treinado para a interface Controller.

Na avaliação, o modelo carregado do checkpoint recebe a `Observation`
estruturada, vetoriza com o MESMO modo de observação usado no treino e age
de forma greedy (determinística). Assim o agente entra no protocolo de
avaliação exatamente como qualquer baseline.
"""

from __future__ import annotations

from pathlib import Path

from stable_baselines3 import DQN

from traffic_rl.config import EnvConfig, ObsMode
from traffic_rl.controllers.base import Controller
from traffic_rl.envs.observation import Observation, ObservationVectorizer


class DQNController(Controller):
    name = "dqn"

    def __init__(self, cfg: EnvConfig, model_path: str | Path, obs_mode: ObsMode | None = None):
        self.model = DQN.load(str(model_path), device="cpu")
        self.vectorizer = ObservationVectorizer(cfg, obs_mode)
        expected = self.model.observation_space.shape
        if expected != (self.vectorizer.size,):
            raise ValueError(
                f"modelo espera observação {expected}, mas o modo "
                f"{self.vectorizer.mode!r} produz ({self.vectorizer.size},) — "
                "use o mesmo obs_mode do treino"
            )

    def act(self, obs: Observation) -> int:
        action, _ = self.model.predict(self.vectorizer(obs), deterministic=True)
        return int(action)
