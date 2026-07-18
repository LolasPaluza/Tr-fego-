"""Contrato da interface Controller para TODOS os controladores (spec Parte 6)."""

import numpy as np
import pytest

from traffic_rl.config import N_STAGES
from traffic_rl.controllers import BASELINE_NAMES, make_controller
from traffic_rl.controllers.base import Controller
from traffic_rl.envs import make_env
from traffic_rl.envs.observation import Observation


def _synthetic_obs(stage=0, time_in_stage=0.0, queues=(3, 2, 8, 1), pressures=(1, 0, 9, 0)):
    return Observation(
        stage=stage,
        time_in_stage_s=time_in_stage,
        queues=np.array(queues, dtype=np.float32),
        lane_densities=np.array([0.1] * 8, dtype=np.float32),
        mean_speeds_norm=np.ones(4, dtype=np.float32),
        first_vehicle_waits_s=np.zeros(4, dtype=np.float32),
        max_waits_s=np.zeros(4, dtype=np.float32),
        stage_pressures=np.array(pressures, dtype=np.float32),
    )


@pytest.fixture(scope="module")
def dqn_model_path(cfg, tmp_path_factory):
    """Modelo DQN não treinado (pesos aleatórios) só para o contrato."""
    from traffic_rl.training.double_dqn import DoubleDQN

    env = make_env(cfg, "pico_assimetrico")
    model = DoubleDQN("MlpPolicy", env, buffer_size=100_000, learning_starts=10, device="cpu")
    path = tmp_path_factory.mktemp("model") / "dqn_contract.zip"
    model.save(path)
    env.close()
    return path


def all_controllers(cfg, dqn_model_path):
    ctrls = [make_controller(n, cfg, "pico_assimetrico") for n in BASELINE_NAMES]
    ctrls.append(
        make_controller("dqn", cfg, "pico_assimetrico", model_path=dqn_model_path)
    )
    return ctrls


def test_contract_synthetic(cfg, dqn_model_path):
    """act() devolve int válido para observações sintéticas variadas."""
    for ctrl in all_controllers(cfg, dqn_model_path):
        assert isinstance(ctrl, Controller)
        ctrl.reset()
        for stage in range(N_STAGES):
            for t in (0.0, 15.0, 200.0):
                action = ctrl.act(_synthetic_obs(stage=stage, time_in_stage=t))
                assert isinstance(action, int)
                assert 0 <= action < N_STAGES


def test_contract_full_episode(cfg, dqn_model_path):
    """Todos os controladores completam um episódio real pelo mesmo loop."""
    env = make_env(cfg, "pico_assimetrico")
    try:
        for ctrl in all_controllers(cfg, dqn_model_path):
            ctrl.reset()
            _, info = env.reset(options={"traffic_seed": 5})
            obs, done = info["observation"], False
            while not done:
                _, _, term, trunc, info = env.step(ctrl.act(obs))
                obs = info["observation"]
                done = term or trunc
    finally:
        env.close()


def test_fixo_igual_cycles(cfg):
    ctrl = make_controller("fixo_igual", cfg, "pico_assimetrico")
    green = ctrl.green_times_s[0]
    assert ctrl.act(_synthetic_obs(stage=0, time_in_stage=green - 1)) == 0
    assert ctrl.act(_synthetic_obs(stage=0, time_in_stage=green)) == 1
    assert ctrl.act(_synthetic_obs(stage=3, time_in_stage=green)) == 0


def test_fixo_proporcional_prioritizes_avenue(cfg):
    ctrl = make_controller("fixo_proporcional", cfg, "pico_assimetrico")
    # avenida com 4x a demanda da local => verde do estágio 0 > estágio 2
    assert ctrl.green_times_s[0] > ctrl.green_times_s[2]


def test_max_pressure_picks_highest(cfg):
    ctrl = make_controller("max_pressure", cfg, "pico_assimetrico")
    obs = _synthetic_obs(stage=0, time_in_stage=30.0, pressures=(1, 0, 9, 0))
    assert ctrl.act(obs) == 2
    # antes do verde mínimo, mantém
    obs_early = _synthetic_obs(stage=0, time_in_stage=5.0, pressures=(1, 0, 9, 0))
    assert ctrl.act(obs_early) == 0


def test_atuado_gap_extends_and_switches(cfg):
    ctrl = make_controller("atuado_gap", cfg, "pico_assimetrico")
    dens = np.zeros(8, dtype=np.float32)
    dens[0] = 0.2  # veículos na faixa E0 (estágio 0)
    obs_demand = Observation(
        stage=0, time_in_stage_s=20.0, queues=np.zeros(4, np.float32),
        lane_densities=dens, mean_speeds_norm=np.ones(4, np.float32),
        first_vehicle_waits_s=np.zeros(4, np.float32),
        max_waits_s=np.zeros(4, np.float32),
        stage_pressures=np.zeros(4, np.float32),
    )
    assert ctrl.act(obs_demand) == 0, "com demanda no estágio atual, estende"
    dens2 = np.zeros(8, dtype=np.float32)
    dens2[6] = 0.3  # só a via local (N0) tem veículos
    obs_gap = Observation(
        stage=0, time_in_stage_s=20.0, queues=np.zeros(4, np.float32),
        lane_densities=dens2, mean_speeds_norm=np.ones(4, np.float32),
        first_vehicle_waits_s=np.zeros(4, np.float32),
        max_waits_s=np.zeros(4, np.float32),
        stage_pressures=np.zeros(4, np.float32),
    )
    assert ctrl.act(obs_gap) == 2, "gap-out pula para o estágio com demanda"


def test_dqn_rejects_wrong_obs_mode(cfg, dqn_model_path):
    with pytest.raises(ValueError):
        make_controller(
            "dqn", cfg, "pico_assimetrico", model_path=dqn_model_path, obs_mode="obs_rica"
        )
