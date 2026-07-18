"""Fixtures compartilhadas: config curta e arquivos de cenário gerados."""

from __future__ import annotations

import pytest

from traffic_rl.config import ProjectConfig, load_config
from traffic_rl.paths import configs_dir


@pytest.fixture(scope="session")
def cfg() -> ProjectConfig:
    """Config com episódios curtos (200 s) para testes rápidos."""
    return load_config(
        configs_dir() / "default.yaml",
        overrides={"env": {"episode_length_s": 200.0}},
    )


@pytest.fixture(scope="session")
def scenario_files(cfg):
    """(net_file, route_file) do pico_assimetrico, gerados uma vez por sessão."""
    from traffic_rl.envs import ensure_scenario_files

    return ensure_scenario_files(cfg, "pico_assimetrico")
