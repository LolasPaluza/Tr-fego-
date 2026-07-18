"""Sanidade das recompensas em observações sintéticas (sinal correto)."""

import numpy as np

from traffic_rl.config import EnvConfig
from traffic_rl.envs.observation import Observation
from traffic_rl.envs.rewards import RewardCalculator


def _obs(queues=(0, 0, 0, 0), max_waits=(0, 0, 0, 0), pressures=(0, 0, 0, 0)):
    return Observation(
        stage=0,
        time_in_stage_s=0.0,
        queues=np.array(queues, dtype=np.float32),
        lane_densities=np.zeros(8, dtype=np.float32),
        mean_speeds_norm=np.ones(4, dtype=np.float32),
        first_vehicle_waits_s=np.zeros(4, dtype=np.float32),
        max_waits_s=np.array(max_waits, dtype=np.float32),
        stage_pressures=np.array(pressures, dtype=np.float32),
    )


def _cfg(**kwargs) -> EnvConfig:
    return EnvConfig(**kwargs)


def test_r_espera_sign():
    calc = RewardCalculator(_cfg(reward_mode="r_espera", starvation={"enabled": False}))
    calc.reset()
    calc(_obs(), total_accumulated_wait_s=500.0)  # referência
    r_melhora = calc(_obs(), total_accumulated_wait_s=300.0)
    assert r_melhora > 0, "espera caiu => recompensa positiva"
    r_piora = calc(_obs(), total_accumulated_wait_s=800.0)
    assert r_piora < 0, "espera subiu => recompensa negativa"


def test_r_fila_sign_and_monotonicity():
    calc = RewardCalculator(_cfg(reward_mode="r_fila", starvation={"enabled": False}))
    assert calc(_obs(queues=(0, 0, 0, 0)), 0.0) == 0.0
    pouco = calc(_obs(queues=(1, 1, 0, 0)), 0.0)
    muito = calc(_obs(queues=(10, 10, 5, 5)), 0.0)
    assert pouco < 0 and muito < pouco


def test_r_pressao_zero_when_balanced():
    calc = RewardCalculator(_cfg(reward_mode="r_pressao", starvation={"enabled": False}))
    assert calc(_obs(pressures=(0, 0, 0, 0)), 0.0) == 0.0
    assert calc(_obs(pressures=(5, -3, 0, 0)), 0.0) < 0


def test_starvation_penalty_applied():
    cfg = _cfg(
        reward_mode="r_fila",
        starvation={"enabled": True, "threshold_s": 120.0, "penalty": 2.0},
    )
    calc = RewardCalculator(cfg)
    sem = calc(_obs(max_waits=(100, 0, 0, 0)), 0.0)
    com = calc(_obs(max_waits=(150, 0, 0, 0)), 0.0)
    assert sem == 0.0
    assert com == -2.0, "1 aproximação acima do limiar => -penalidade"
    duas = calc(_obs(max_waits=(150, 0, 200, 0)), 0.0)
    assert duas == -4.0


def test_starvation_disabled():
    cfg = _cfg(reward_mode="r_fila", starvation={"enabled": False})
    calc = RewardCalculator(cfg)
    assert calc(_obs(max_waits=(999, 999, 999, 999)), 0.0) == 0.0
