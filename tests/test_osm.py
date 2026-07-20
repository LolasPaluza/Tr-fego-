"""Fase 4: pipeline OSM — importação, cruzamentos controláveis, demanda, env."""

import pytest

from tests.osm_fixture import write_osm_fixture
from traffic_rl.envs.osm_demand import build_osm_routes
from traffic_rl.envs.osm_network import discover_controllable_tls, import_osm


@pytest.fixture(scope="module")
def osm_net(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("osm")
    osm = write_osm_fixture(tmp / "bairro.osm")
    net = import_osm(osm, tmp)
    return osm, net


def test_import_and_discover(osm_net):
    _, net = osm_net
    assert net.exists()
    tables = discover_controllable_tls(net)
    assert len(tables) >= 1, "fixture 2×2 deveria ter cruzamentos controláveis"
    for table in tables.values():
        assert len(table.green_states) == 4
        assert all("G" in s for s in table.green_states)


def _sem_comentarios(text: str) -> str:
    """Remove comentários XML (carregam timestamp de geração)."""
    import re

    return re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)


def test_routes_generate_and_are_deterministic(osm_net, tmp_path):
    _, net = osm_net
    r1 = build_osm_routes(net, tmp_path, traffic_seed=3, episode_length_s=300,
                          total_vph=600)
    assert r1.exists()
    content = r1.read_text()
    assert "<vehicle" in content or "<trip" in content
    r2 = build_osm_routes(net, tmp_path, traffic_seed=3, episode_length_s=300,
                          total_vph=600)
    assert _sem_comentarios(content) == _sem_comentarios(r2.read_text()), (
        "mesma seed => mesmas rotas"
    )


def test_osm_env_episode(osm_net, tmp_path, monkeypatch):
    osm, _ = osm_net
    monkeypatch.setenv("TRAFFIC_RL_ROOT", str(tmp_path))
    from traffic_rl.osm_config import OsmProjectConfig

    cfg = OsmProjectConfig(
        osm_file=str(osm),
        env={"episode_length_s": 120.0},
        demand={"total_vph": 800},
    )
    from traffic_rl.envs.osm_env import OsmTrafficEnv

    env = OsmTrafficEnv(cfg)
    try:
        obs, info = env.reset(traffic_seed=5)
        assert len(obs) == env.n_tls >= 1
        # verde mínimo vale também na malha real
        obs, _, _, _ = env.step([2] * env.n_tls)
        assert all(o.stage == 0 for o in obs)
        done = False
        while not done:
            obs, rewards, done, info = env.step([0] * env.n_tls)
        assert info["sim_time"] >= 120.0
    finally:
        env.close()
