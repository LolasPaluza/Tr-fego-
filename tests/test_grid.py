"""Testes da Fase 2: topologia, demanda, fases, env multi-TLS e controladores."""

import numpy as np
import pytest

from traffic_rl.envs.grid_demand import build_grid_routes, grid_road_class_of_vehicle
from traffic_rl.envs.grid_env import GridTrafficEnv, GridVecEnv
from traffic_rl.envs.grid_network import GridTopology, build_grid_network
from traffic_rl.envs.phases import build_phase_table
from traffic_rl.grid_config import load_grid_config, slugify
from traffic_rl.paths import configs_dir


@pytest.fixture(scope="module")
def gcfg():
    return load_grid_config(
        configs_dir() / "grid.yaml", overrides={"env": {"episode_length_s": 150.0}}
    )


@pytest.fixture(scope="module")
def topo(gcfg):
    return GridTopology(gcfg.grid)


def test_slugify():
    assert slugify("Av. Paulista") == "avpaulista"
    assert slugify("Rua Aspicuelta") == "ruaaspicuelta"


def test_entry_points(topo):
    pts = topo.entry_points()
    assert len(pts) == 2 * (topo.R + topo.C)
    assert len({p.entry_edge for p in pts}) == len(pts)


def test_random_walks_always_exit(topo):
    """De qualquer entrada, qualquer sequência de movimentos sai do grid."""
    rng = np.random.default_rng(3)
    for entry in topo.entry_points():
        for _ in range(20):
            edge, hops = entry.entry_edge, 0
            while (moves := topo.next_moves(edge)) is not None:
                edge = moves[rng.choice(["s", "r", "l"])]
                hops += 1
                assert hops < 500, f"caminhada não terminou a partir de {entry.entry_edge}"


def test_grid_network_builds(gcfg, tmp_path):
    net = build_grid_network(gcfg.grid, tmp_path)
    assert net.exists()
    # todos os cruzamentos internos têm tabela de fases válida e sem conflito
    for tls in gcfg.grid.tls_ids():
        table = build_phase_table(str(net), tls)
        ew = {i for i, c in enumerate(table.green_states[0]) if c == "G"}
        ns = {i for i, c in enumerate(table.green_states[2]) if c == "G"}
        assert ew and ns and not (ew & ns), f"conflito de estágios em {tls}"


def test_routes_deterministic_and_scaled(gcfg, tmp_path):
    r1 = build_grid_routes(gcfg.grid, tmp_path, traffic_seed=7, episode_length_s=600)
    content1 = r1.read_text()
    r2 = build_grid_routes(gcfg.grid, tmp_path, traffic_seed=7, episode_length_s=600)
    assert content1 == r2.read_text(), "mesma seed => mesmo tráfego"
    r3 = build_grid_routes(gcfg.grid, tmp_path, traffic_seed=8, episode_length_s=600)
    assert content1 != r3.read_text()
    # volume esperado: soma dos fluxos de entrada × horizonte
    n_veh = content1.count("<vehicle ")
    expected = sum(2 * s.flow_vph for s in gcfg.grid.rows + gcfg.grid.cols) * 600 / 3600
    assert 0.6 * expected < n_veh < 1.4 * expected


def test_grid_road_class():
    assert grid_road_class_of_vehicle("av_avpaulista_carro.3") == "avenida"
    assert grid_road_class_of_vehicle("loc_ruawisard_caminhao.0") == "local"


def test_grid_env_min_green_and_shapes(gcfg):
    env = GridTrafficEnv(gcfg)
    try:
        obs, _ = env.reset(traffic_seed=5)
        assert len(obs) == env.n_tls == 9
        assert env.vectorize(obs).shape == (9, 9)
        # troca precoce negada em todos os cruzamentos
        obs, _, _, _ = env.step([2] * env.n_tls)
        assert all(o.stage == 0 for o in obs)
        obs, _, _, _ = env.step([0] * env.n_tls)  # 10 s de verde
        obs, _, _, _ = env.step([2] * env.n_tls)  # agora permitido
        assert all(o.stage == 2 for o in obs)
    finally:
        env.close()


def test_grid_vecenv_contract(gcfg):
    venv = GridVecEnv(gcfg, traffic_seed_base=100)
    try:
        obs = venv.reset()
        assert obs.shape == (9, 9)
        obs, rewards, dones, infos = venv.step(np.zeros(9, dtype=int))
        assert rewards.shape == (9,) and not dones.any()
        # roda até o fim do episódio: auto-reset com TimeLimit.truncated
        done = False
        for _ in range(100):
            obs, rewards, dones, infos = venv.step(np.zeros(9, dtype=int))
            if dones.any():
                done = True
                assert all(i.get("TimeLimit.truncated") for i in infos)
                assert "terminal_observation" in infos[0]
                break
        assert done, "episódio de 150s deveria terminar em <100 passos"
    finally:
        venv.close()


def test_grid_controllers_contract(gcfg):
    from traffic_rl.controllers.grid import GRID_BASELINES, make_grid_controllers

    env = GridTrafficEnv(gcfg)
    try:
        observations, _ = env.reset(traffic_seed=6)
        for name in GRID_BASELINES:
            ctrls = make_grid_controllers(name, gcfg, env)
            assert len(ctrls) == env.n_tls
            for ctrl, obs in zip(ctrls, observations, strict=True):
                action = ctrl.act(obs)
                assert isinstance(action, int) and 0 <= action < 4
    finally:
        env.close()


def test_grid_proportional_prioritizes_declared_avenue(gcfg):
    from traffic_rl.controllers.grid import make_grid_controllers

    env = GridTrafficEnv(gcfg)
    try:
        ctrls = make_grid_controllers("fixo_proporcional", gcfg, env)
        # n_1_0 = Av. Paulista (row 1, 1100 vph) × Rua Girassol (col 0, 220 vph)
        idx = 1 * 3 + 0
        greens = ctrls[idx].green_times_s
        assert greens[0] > greens[2], "avenida declarada deve receber mais verde"
    finally:
        env.close()


def test_grid_coordination_observation(gcfg):
    """Fase 3: com coordination=True, cada cruzamento vê os 4 vizinhos (obs 9→13)."""
    import numpy as np

    from traffic_rl.grid_config import GridProjectConfig
    coord_cfg = GridProjectConfig(
        grid=gcfg.grid,
        env={**gcfg.env.model_dump(), "coordination": True},
        train=gcfg.train, eval=gcfg.eval,
    )
    env = GridTrafficEnv(coord_cfg)
    try:
        assert env.obs_dim == 13
        obs, _ = env.reset(traffic_seed=1)
        vec = env.vectorize(obs)
        assert vec.shape == (env.n_tls, 13)
        assert np.all(vec >= 0) and np.all(vec <= 1)
        # cruzamento central (idx 4 num 3x3) tem 4 vizinhos válidos
        assert all(n >= 0 for n in env._neighbors[4])
        # cruzamento de canto (idx 0) tem 2 vizinhos de borda (-1)
        assert env._neighbors[0].count(-1) == 2
    finally:
        env.close()
