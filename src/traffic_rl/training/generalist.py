"""Treino generalista (multi-cenário) — o experimento que corrige a
especialização detectada na Fase 1.

Mesma máquina do `train_seed`, com duas diferenças:
- o env de treino E o de avaliação periódica trocam de cenário a cada
  episódio (round-robin sobre os 4 cenários);
- o run vive sob a tag `dqn_generalista_...`, separado do especialista, para
  a comparação especialista × generalista ficar limpa.

Reusa os helpers e constantes do módulo `train` (git hash, retomada de
checkpoint, dump da curva de avaliação).
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import yaml
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from traffic_rl.config import ObsMode, ProjectConfig, RewardMode
from traffic_rl.paths import runs_dir
from traffic_rl.training.double_dqn import DoubleDQN
from traffic_rl.training.train import (
    EVAL_DURING_TRAIN_BASE,
    TRAIN_SEED_STRIDE,
    _dump_eval_csv,
    _git_hash,
    _latest_checkpoint,
)


def generalist_tag(
    cfg: ProjectConfig, obs_mode: ObsMode | None, reward_mode: RewardMode | None
) -> str:
    obs = obs_mode or cfg.env.obs_mode
    rew = reward_mode or cfg.env.reward_mode
    return f"dqn_generalista_{obs}_{rew}"


def _multi_env_factory(cfg, scenario_names, traffic_seed_base, obs_mode, reward_mode):
    # import tardio: roda dentro do subprocesso do SubprocVecEnv
    from traffic_rl.envs.multi_scenario import make_multi_scenario_env

    env = make_multi_scenario_env(
        cfg, scenario_names,
        traffic_seed_base=traffic_seed_base, obs_mode=obs_mode, reward_mode=reward_mode,
    )
    return Monitor(env)


def find_generalist_models(tag: str) -> dict[int, Path]:
    models: dict[int, Path] = {}
    for seed_dir in sorted((runs_dir() / tag).glob("seed_*")):
        best = seed_dir / "best_model.zip"
        if best.exists():
            try:
                models[int(seed_dir.name.removeprefix("seed_"))] = best
            except ValueError:
                continue
    return models


def train_generalist_seed(
    cfg: ProjectConfig,
    seed: int,
    scenario_names: list[str] | None = None,
    obs_mode: ObsMode | None = None,
    reward_mode: RewardMode | None = None,
) -> Path:
    scenarios = scenario_names or list(cfg.eval.scenarios)
    tag = generalist_tag(cfg, obs_mode, reward_mode)
    run_dir = runs_dir() / tag / f"seed_{seed}"
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)
    (run_dir / "git_hash.txt").write_text(_git_hash() + "\n", encoding="utf-8")
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.model_dump(mode="json"), f, allow_unicode=True, sort_keys=False)
    (run_dir / "scenarios.txt").write_text("\n".join(scenarios) + "\n", encoding="utf-8")

    train_env = SubprocVecEnv(
        [partial(
            _multi_env_factory, cfg, scenarios,
            seed * TRAIN_SEED_STRIDE, obs_mode, reward_mode,
        )],
        start_method="fork",
    )
    eval_env = SubprocVecEnv(
        [partial(
            _multi_env_factory, cfg, scenarios,
            EVAL_DURING_TRAIN_BASE + seed, obs_mode, reward_mode,
        )],
        start_method="fork",
    )
    try:
        dqn_cfg = cfg.train.dqn
        algo = DoubleDQN if dqn_cfg.double_dqn else DQN
        resume = _latest_checkpoint(ckpt_dir)
        if resume is not None:
            ckpt_path, done_steps = resume
            print(f"[generalista seed {seed}] retomando de {ckpt_path.name} ({done_steps} passos)")
            model = algo.load(ckpt_path, env=train_env, device="cpu")
            buffer_path = ckpt_path.with_name(ckpt_path.stem + "_replay_buffer.pkl")
            if buffer_path.exists():
                model.load_replay_buffer(buffer_path)
            remaining = max(cfg.train.total_timesteps - done_steps, 0)
            reset_timesteps = False
        else:
            model = algo(
                "MlpPolicy", train_env,
                learning_rate=dqn_cfg.learning_rate, buffer_size=dqn_cfg.buffer_size,
                learning_starts=dqn_cfg.learning_starts, batch_size=dqn_cfg.batch_size,
                gamma=dqn_cfg.gamma, train_freq=dqn_cfg.train_freq,
                target_update_interval=dqn_cfg.target_update_interval,
                exploration_fraction=dqn_cfg.exploration_fraction,
                exploration_initial_eps=dqn_cfg.exploration_initial_eps,
                exploration_final_eps=dqn_cfg.exploration_final_eps,
                policy_kwargs={"net_arch": list(dqn_cfg.net_arch)},
                seed=seed, device="cpu",
                tensorboard_log=str(run_dir / "tensorboard"), verbose=0,
            )
            remaining = cfg.train.total_timesteps
            reset_timesteps = True
        if remaining == 0:
            print(f"[generalista seed {seed}] treino já completo")
            return run_dir

        eval_cb = EvalCallback(
            eval_env, best_model_save_path=str(run_dir), log_path=str(run_dir),
            eval_freq=cfg.train.eval_freq,
            n_eval_episodes=max(len(scenarios), cfg.train.n_eval_episodes),
            deterministic=True, verbose=0,
        )
        ckpt_cb = CheckpointCallback(
            save_freq=cfg.train.checkpoint_freq, save_path=str(ckpt_dir),
            name_prefix="ckpt", save_replay_buffer=True, verbose=0,
        )
        model.learn(
            total_timesteps=remaining, callback=[eval_cb, ckpt_cb],
            reset_num_timesteps=reset_timesteps, progress_bar=False,
        )
        if not (run_dir / "best_model.zip").exists():
            model.save(run_dir / "best_model")
        _dump_eval_csv(run_dir)
    finally:
        train_env.close()
        eval_env.close()
    return run_dir


def train_generalist_all_seeds(
    cfg: ProjectConfig, seeds: list[int] | None = None,
    obs_mode: ObsMode | None = None, reward_mode: RewardMode | None = None,
) -> list[Path]:
    dirs = []
    for seed in seeds or cfg.train.seeds:
        print(f"=== generalista: treinando seed {seed} (multi-cenário) ===")
        dirs.append(train_generalist_seed(cfg, seed, obs_mode=obs_mode, reward_mode=reward_mode))
    return dirs
