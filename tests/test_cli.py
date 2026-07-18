import pandas as pd
import pytest
import yaml

from traffic_rl.cli import main
from traffic_rl.paths import configs_dir


@pytest.fixture()
def mini_config(tmp_path):
    """Config real com 1 cenário e episódios curtos para o CLI."""
    with open(configs_dir() / "default.yaml", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    raw["env"]["episode_length_s"] = 200.0
    raw["eval"] = {"n_episodes": 2, "traffic_seed_base": 10000,
                   "scenarios": ["pico_assimetrico"]}
    p = tmp_path / "mini.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return p


def test_cli_requires_command(capsys):
    with pytest.raises(SystemExit):
        main([])


def test_cli_evaluate_baselines(mini_config, tmp_path):
    out = tmp_path / "eval"
    rc = main([
        "evaluate", "--config", str(mini_config),
        "--methods", "fixo_igual,max_pressure", "--out", str(out),
    ])
    assert rc == 0
    df = pd.read_csv(out / "metrics.csv")
    assert set(df["method"]) == {"fixo_igual", "max_pressure"}
    assert len(df) == 2 * 2  # 2 métodos × 2 episódios
    assert (df["espera_media_s"] >= 0).all()


def test_cli_evaluate_dqn_without_model_fails(mini_config, tmp_path, monkeypatch):
    monkeypatch.setenv("TRAFFIC_RL_ROOT", str(tmp_path / "vazio"))
    rc = main(["evaluate", "--config", str(mini_config), "--methods", "dqn"])
    assert rc == 1
