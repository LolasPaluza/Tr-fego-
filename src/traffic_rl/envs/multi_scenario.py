"""Ambiente multi-cenário (treino generalista) — corrige a especialização.

O `SumoIntersectionEnv` fixa um cenário na construção. Aqui, a MESMA rede é
reusada e, a CADA episódio, um cenário diferente é escolhido (round-robin
sobre a lista) — o agente encara pico normal, fora de pico, balanceado e pico
invertido ao longo do treino. Assim ele não consegue decorar "priorize a
avenida": é forçado a aprender a regra geral "priorize quem tem fila".

É a técnica de domain randomization aplicada à demanda. Ver ADR-020.
"""

from __future__ import annotations

from pathlib import Path

from traffic_rl.config import EnvConfig, ObsMode, ProjectConfig, RewardMode
from traffic_rl.envs.demand import build_routes
from traffic_rl.envs.network import build_network
from traffic_rl.envs.observation import Observation
from traffic_rl.envs.traffic_env import SumoIntersectionEnv
from traffic_rl.paths import generated_dir


class MultiScenarioIntersectionEnv(SumoIntersectionEnv):
    """Cruzamento único cuja demanda troca de cenário a cada episódio."""

    def __init__(
        self,
        cfg: EnvConfig,
        net_file: str | Path,
        route_choices: list[tuple[str, Path]],
        **kwargs,
    ) -> None:
        if not route_choices:
            raise ValueError("route_choices não pode ser vazio")
        self._route_choices = list(route_choices)
        super().__init__(cfg, net_file, self._route_choices[0][1], **kwargs)
        self._current_scenario = self._route_choices[0][0]

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        # escolhe o cenário do episódio ANTES de o SUMO iniciar (round-robin)
        name, path = self._route_choices[self._episode_index % len(self._route_choices)]
        self.route_file = str(path)
        self._current_scenario = name
        return super().reset(seed=seed, options=options)

    def _info(self, obs: Observation, total_wait: float = 0.0) -> dict:
        info = super()._info(obs, total_wait)
        info["scenario"] = self._current_scenario
        return info


def make_multi_scenario_env(
    cfg: ProjectConfig,
    scenario_names: list[str],
    traffic_seed_base: int = 0,
    obs_mode: ObsMode | None = None,
    reward_mode: RewardMode | None = None,
    tripinfo_dir: str | Path | None = None,
) -> MultiScenarioIntersectionEnv:
    out = generated_dir()
    net_file = build_network(cfg.env.network, out)
    route_choices = [(name, build_routes(cfg.scenario(name), out)) for name in scenario_names]
    return MultiScenarioIntersectionEnv(
        cfg.env,
        net_file=net_file,
        route_choices=route_choices,
        traffic_seed_base=traffic_seed_base,
        obs_mode=obs_mode,
        reward_mode=reward_mode,
        tripinfo_dir=tripinfo_dir,
    )
