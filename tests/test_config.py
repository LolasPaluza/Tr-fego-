import pytest
from pydantic import ValidationError

from traffic_rl.config import (
    EnvConfig,
    ScenarioConfig,
    SignalConfig,
    TurnShares,
    load_config,
)
from traffic_rl.paths import configs_dir


def test_default_config_loads(cfg):
    assert set(cfg.scenarios) == {
        "pico_assimetrico", "fora_pico", "balanceado", "pico_invertido"
    }
    assert cfg.env.signal.min_green_s == 10.0
    assert cfg.train.dqn.buffer_size >= 100_000
    assert cfg.train.seeds == [42, 123, 7, 2024, 777]


def test_smoke_overlay_reduces_parameters():
    import yaml

    with open(configs_dir() / "smoke.yaml", encoding="utf-8") as f:
        overrides = yaml.safe_load(f)
    smoke = load_config(configs_dir() / "default.yaml", overrides)
    assert smoke.train.total_timesteps <= 10_000
    assert smoke.env.episode_length_s < 3600
    # o overlay não pode destruir o resto da config
    assert set(smoke.scenarios) == {
        "pico_assimetrico", "fora_pico", "balanceado", "pico_invertido"
    }


def test_min_green_floor():
    with pytest.raises(ValidationError):
        SignalConfig(min_green_s=1.0)


def test_max_green_must_exceed_min():
    with pytest.raises(ValidationError):
        SignalConfig(min_green_s=30.0, max_green_s=20.0)


def test_turn_shares_must_sum_to_one():
    with pytest.raises(ValidationError):
        TurnShares(straight=0.9, right=0.3, left=0.1)


def test_delta_time_cannot_exceed_min_green():
    with pytest.raises(ValidationError):
        EnvConfig(delta_time_s=15.0)


def test_unknown_scenario_raises(cfg):
    with pytest.raises(KeyError):
        cfg.scenario("inexistente")


def test_scenario_flows_positive():
    with pytest.raises(ValidationError):
        ScenarioConfig(name="x", avenue_flow_vph=0, local_flow_vph=100)
