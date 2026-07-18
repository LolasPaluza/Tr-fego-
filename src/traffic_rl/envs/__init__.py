"""Construção de redes, cenários de demanda e o ambiente Gymnasium."""

from __future__ import annotations

from pathlib import Path

from traffic_rl.config import EnvConfig, ObsMode, ProjectConfig, RewardMode
from traffic_rl.envs.demand import build_routes
from traffic_rl.envs.network import build_network
from traffic_rl.envs.traffic_env import SumoIntersectionEnv
from traffic_rl.paths import generated_dir

__all__ = ["SumoIntersectionEnv", "ensure_scenario_files", "make_env"]


def ensure_scenario_files(cfg: ProjectConfig, scenario_name: str) -> tuple[Path, Path]:
    """Gera (idempotente) rede e rotas do cenário; retorna (net_file, route_file).

    Regenerar é barato (<1 s), então sempre regeneramos — garante que os
    arquivos refletem a config atual e elimina estados obsoletos.
    """
    out = generated_dir()
    net_file = build_network(cfg.env.network, out)
    route_file = build_routes(cfg.scenario(scenario_name), out)
    return net_file, route_file


def make_env(
    cfg: ProjectConfig,
    scenario_name: str,
    traffic_seed_base: int = 0,
    obs_mode: ObsMode | None = None,
    reward_mode: RewardMode | None = None,
    tripinfo_dir: str | Path | None = None,
    env_cfg: EnvConfig | None = None,
) -> SumoIntersectionEnv:
    net_file, route_file = ensure_scenario_files(cfg, scenario_name)
    return SumoIntersectionEnv(
        env_cfg or cfg.env,
        net_file=net_file,
        route_file=route_file,
        traffic_seed_base=traffic_seed_base,
        obs_mode=obs_mode,
        reward_mode=reward_mode,
        tripinfo_dir=tripinfo_dir,
    )
