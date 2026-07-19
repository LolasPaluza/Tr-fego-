"""Treino da Fase 2: DQN com política compartilhada sobre o GridVecEnv.

Mesmo protocolo da Fase 1 (multi-seed, avaliação periódica greedy, melhor
checkpoint, retomada), com duas diferenças estruturais:
- o "VecEnv" são os K cruzamentos da MESMA simulação (parameter sharing);
  cada passo de decisão coleta K transições — 300k timesteps do sb3 são
  300k transições, não 300k decisões;
- o env de treino usa libsumo (em processo) e o de avaliação periódica usa
  TraCI (processo sumo externo): os dois mecanismos coexistem sem ferir a
  regra "1 simulação libsumo por processo" (ADR-013).
"""

from __future__ import annotations

from pathlib import Path

import yaml
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.vec_env import VecMonitor

from traffic_rl.envs.grid_env import GridVecEnv
from traffic_rl.grid_config import GridProjectConfig
from traffic_rl.paths import runs_dir
from traffic_rl.training.double_dqn import DoubleDQN
from traffic_rl.training.train import (
    EVAL_DURING_TRAIN_BASE,
    TRAIN_SEED_STRIDE,
    _dump_eval_csv,
    _git_hash,
    _latest_checkpoint,
)


def grid_run_tag(cfg: GridProjectConfig) -> str:
    return f"grid_{len(cfg.grid.rows)}x{len(cfg.grid.cols)}_{cfg.env.reward_mode}"


def find_grid_best_models(tag: str) -> dict[int, Path]:
    models: dict[int, Path] = {}
    for seed_dir in sorted((runs_dir() / tag).glob("seed_*")):
        best = seed_dir / "best_model.zip"
        if best.exists():
            try:
                models[int(seed_dir.name.removeprefix("seed_"))] = best
            except ValueError:
                continue
    return models


def train_grid_seed(cfg: GridProjectConfig, seed: int) -> Path:
    tag = grid_run_tag(cfg)
    run_dir = runs_dir() / tag / f"seed_{seed}"
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)
    (run_dir / "git_hash.txt").write_text(_git_hash() + "\n", encoding="utf-8")
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.model_dump(mode="json"), f, allow_unicode=True, sort_keys=False)

    train_env = VecMonitor(
        GridVecEnv(
            cfg, traffic_seed_base=seed * TRAIN_SEED_STRIDE, use_libsumo=True, label="treino"
        )
    )
    eval_env = VecMonitor(
        GridVecEnv(
            cfg,
            traffic_seed_base=EVAL_DURING_TRAIN_BASE + seed,
            use_libsumo=False,  # TraCI: coexiste com o libsumo do env de treino
            label=f"aval_{seed}",
        )
    )
    try:
        dqn_cfg = cfg.train.dqn
        algo = DoubleDQN if dqn_cfg.double_dqn else DQN
        resume = _latest_checkpoint(ckpt_dir)
        if resume is not None:
            ckpt_path, done_steps = resume
            print(f"[grid seed {seed}] retomando de {ckpt_path.name} ({done_steps} passos)")
            model = algo.load(ckpt_path, env=train_env, device="cpu")
            buffer_path = ckpt_path.with_name(ckpt_path.stem + "_replay_buffer.pkl")
            if buffer_path.exists():
                model.load_replay_buffer(buffer_path)
            remaining = max(cfg.train.total_timesteps - done_steps, 0)
            reset_timesteps = False
        else:
            model = algo(
                "MlpPolicy",
                train_env,
                learning_rate=dqn_cfg.learning_rate,
                buffer_size=dqn_cfg.buffer_size,
                learning_starts=dqn_cfg.learning_starts,
                batch_size=dqn_cfg.batch_size,
                gamma=dqn_cfg.gamma,
                train_freq=dqn_cfg.train_freq,
                target_update_interval=dqn_cfg.target_update_interval,
                exploration_fraction=dqn_cfg.exploration_fraction,
                exploration_initial_eps=dqn_cfg.exploration_initial_eps,
                exploration_final_eps=dqn_cfg.exploration_final_eps,
                policy_kwargs={"net_arch": list(dqn_cfg.net_arch)},
                seed=seed,
                device="cpu",
                tensorboard_log=str(run_dir / "tensorboard"),
                verbose=0,
            )
            remaining = cfg.train.total_timesteps
            reset_timesteps = True
        if remaining == 0:
            print(f"[grid seed {seed}] treino já completo")
            return run_dir

        stop_cb = None
        if cfg.train.early_stopping:
            stop_cb = StopTrainingOnNoModelImprovement(
                max_no_improvement_evals=cfg.train.early_stopping_patience, min_evals=5
            )
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=str(run_dir),
            log_path=str(run_dir),
            eval_freq=max(cfg.train.eval_freq // train_env.num_envs, 1),
            # 1 rodada de simulação de avaliação = K episódios (um por cruzamento)
            n_eval_episodes=train_env.num_envs * cfg.train.n_eval_episodes,
            deterministic=True,
            callback_after_eval=stop_cb,
            verbose=0,
        )
        ckpt_cb = CheckpointCallback(
            save_freq=max(cfg.train.checkpoint_freq // train_env.num_envs, 1),
            save_path=str(ckpt_dir),
            name_prefix="ckpt",
            save_replay_buffer=True,
            verbose=0,
        )
        model.learn(
            total_timesteps=remaining,
            callback=[eval_cb, ckpt_cb],
            reset_num_timesteps=reset_timesteps,
            progress_bar=False,
        )
        if not (run_dir / "best_model.zip").exists():
            model.save(run_dir / "best_model")
        _dump_eval_csv(run_dir)
    finally:
        train_env.close()
        eval_env.close()
    return run_dir


def train_grid_all_seeds(cfg: GridProjectConfig, seeds: list[int] | None = None) -> list[Path]:
    dirs = []
    for seed in seeds or cfg.train.seeds:
        print(f"=== grid: treinando seed {seed} ({cfg.train.total_timesteps} transições) ===")
        dirs.append(train_grid_seed(cfg, seed))
    return dirs
