"""Smoke test de treino (spec Parte 6): 1k passos de ponta a ponta."""

import numpy as np

from traffic_rl.config import load_config
from traffic_rl.paths import configs_dir
from traffic_rl.training.train import find_best_models, train_seed


def test_train_smoke_1k(tmp_path, monkeypatch):
    config_path = configs_dir() / "default.yaml"  # resolve ANTES de mudar a raiz
    monkeypatch.setenv("TRAFFIC_RL_ROOT", str(tmp_path))
    cfg = load_config(
        config_path,
        overrides={
            "env": {"episode_length_s": 300.0},
            "train": {
                "total_timesteps": 1000, "eval_freq": 500, "n_eval_episodes": 1,
                "checkpoint_freq": 500, "dqn": {"learning_starts": 100},
            },
        },
    )
    run_dir = train_seed(cfg, seed=42)
    assert (run_dir / "best_model.zip").exists()
    assert (run_dir / "git_hash.txt").exists()
    assert (run_dir / "config.yaml").exists()
    assert list((run_dir / "checkpoints").glob("ckpt_*_steps.zip"))
    data = np.load(run_dir / "evaluations.npz")
    assert len(data["timesteps"]) >= 1, "avaliação periódica precisa ter rodado"
    assert (run_dir / "eval_log.csv").exists()
    # descoberta de best models enxerga o run
    models = find_best_models(run_dir.parent.name)
    assert 42 in models
