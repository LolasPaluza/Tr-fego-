"""Fábrica de controladores por nome — usada pelo CLI e pela avaliação."""

from __future__ import annotations

from pathlib import Path

from traffic_rl.config import ObsMode, ProjectConfig
from traffic_rl.controllers.actuated import GapActuatedController
from traffic_rl.controllers.base import Controller
from traffic_rl.controllers.fixed import FixedTimeController, ProportionalFixedController
from traffic_rl.controllers.max_pressure import MaxPressureController

BASELINE_NAMES = ("fixo_igual", "fixo_proporcional", "atuado_gap", "max_pressure")


def make_controller(
    name: str,
    cfg: ProjectConfig,
    scenario_name: str,
    model_path: str | Path | None = None,
    obs_mode: ObsMode | None = None,
) -> Controller:
    signal = cfg.env.signal
    if name == "fixo_igual":
        return FixedTimeController(signal)
    if name == "fixo_proporcional":
        return ProportionalFixedController(signal, cfg.scenario(scenario_name))
    if name == "atuado_gap":
        return GapActuatedController(cfg.env)
    if name == "max_pressure":
        return MaxPressureController(signal)
    if name == "dqn":
        if model_path is None:
            raise ValueError("controlador dqn exige model_path (checkpoint treinado)")
        from traffic_rl.controllers.dqn import DQNController

        return DQNController(cfg.env, model_path, obs_mode)
    raise ValueError(f"controlador desconhecido: {name!r}")
