import numpy as np
import pytest

from traffic_rl.envs import make_env


@pytest.fixture()
def env(cfg):
    e = make_env(cfg, "pico_assimetrico")
    yield e
    e.close()


def test_observation_spaces(cfg):
    e_min = make_env(cfg, "pico_assimetrico", obs_mode="obs_minimal")
    vec, _ = e_min.reset(options={"traffic_seed": 1})
    assert vec.shape == (9,)
    assert e_min.observation_space.contains(vec)
    e_min.close()
    e_rica = make_env(cfg, "pico_assimetrico", obs_mode="obs_rica")
    vec, _ = e_rica.reset(options={"traffic_seed": 1})
    assert vec.shape == (25,)
    assert e_rica.observation_space.contains(vec)
    e_rica.close()


def test_min_green_enforced(env):
    env.reset(options={"traffic_seed": 2})
    _, _, _, _, info = env.step(2)  # troca imediata deve ser negada
    assert info["observation"].stage == 0
    _, _, _, _, info = env.step(0)  # 10 s de verde acumulados
    _, _, _, _, info = env.step(2)  # agora a troca é permitida
    assert info["observation"].stage == 2


def test_max_green_forces_switch(cfg):
    env = make_env(cfg, "pico_assimetrico")
    env.reset(options={"traffic_seed": 2})
    max_green = cfg.env.signal.max_green_s
    steps_needed = int(max_green / cfg.env.delta_time_s) + 1
    stage = 0
    for _ in range(steps_needed):
        _, _, _, _, info = env.step(0)  # insiste em manter o estágio 0
        stage = info["observation"].stage
    assert stage != 0, "verde máximo deve forçar a troca"
    env.close()


def test_episode_truncates_at_length(env, cfg):
    env.reset(options={"traffic_seed": 3})
    done, steps = False, 0
    while not done:
        _, _, term, trunc, info = env.step(0)
        done = term or trunc
        steps += 1
        assert steps < 1000, "episódio não terminou"
    assert info["sim_time"] >= cfg.env.episode_length_s


def test_determinism_same_seed(cfg):
    """Mesma seed de tráfego + mesmas ações ⇒ mesma trajetória."""

    def rollout():
        env = make_env(cfg, "pico_assimetrico")
        env.reset(options={"traffic_seed": 7})
        acc = []
        actions = [0, 0, 0, 2, 2, 2, 1, 1, 0, 0, 3, 3, 0, 0, 0]
        for a in actions:
            vec, r, _, _, info = env.step(a)
            acc.append((vec.copy(), r, info["observation"].queues.copy()))
        env.close()
        return acc

    r1, r2 = rollout(), rollout()
    for (v1, rew1, q1), (v2, rew2, q2) in zip(r1, r2, strict=True):
        np.testing.assert_array_equal(v1, v2)
        assert rew1 == rew2
        np.testing.assert_array_equal(q1, q2)


def test_different_seeds_differ(cfg):
    def total_arrivals(seed):
        env = make_env(cfg, "pico_assimetrico")
        env.reset(options={"traffic_seed": seed})
        done = False
        while not done:
            _, _, term, trunc, info = env.step(0)
            done = term or trunc
        env.close()
        return info["arrived_cum"], tuple(info["observation"].queues)

    assert total_arrivals(11) != total_arrivals(12)


def test_observation_normalized(env):
    env.reset(options={"traffic_seed": 4})
    for _ in range(30):
        vec, _, _, trunc, _ = env.step(env.action_space.sample())
        assert np.all(vec >= 0.0) and np.all(vec <= 1.0)
        if trunc:
            break
