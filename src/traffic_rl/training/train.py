"""Protocolo de treino do DQN: multi-seed, avaliação periódica, retomável.

Cada run (= uma seed) vive em results/runs/{tag}/seed_{seed}/ com:
- config.yaml       config completa resolvida (reprodutibilidade)
- git_hash.txt      commit exato do código
- best_model.zip    melhor checkpoint segundo a avaliação periódica (greedy)
- evaluations.npz   curva de avaliação (EvalCallback) + eval_log.csv derivado
- checkpoints/      últimos checkpoints com replay buffer (retomada após queda)
- tensorboard/      métricas de treino

Isolamento de processos: cada ambiente SUMO roda num subprocesso
(SubprocVecEnv), porque o libsumo só suporta UMA simulação por processo —
assim o env de treino e o de avaliação periódica coexistem sem conflito.

Seeds de tráfego: treino usa base seed*1_000_000; avaliação periódica usa
base 900_000_000+seed — ambas disjuntas das seeds do protocolo final
(10_000+i). A seed do DQN (pesos, replay, epsilon) é a própria `seed`.
"""

from __future__ import annotations

import csv
import subprocess
from functools import partial
from pathlib import Path

import numpy as np
import yaml
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from traffic_rl.config import ObsMode, ProjectConfig, RewardMode
from traffic_rl.paths import project_root, runs_dir
from traffic_rl.training.double_dqn import DoubleDQN

TRAIN_SEED_STRIDE = 1_000_000
EVAL_DURING_TRAIN_BASE = 900_000_000


def _git_hash() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=project_root(), timeout=10,
        )
        return out.stdout.strip() or "desconhecido"
    except Exception:
        return "desconhecido"


def _env_factory(
    cfg: ProjectConfig,
    scenario: str,
    traffic_seed_base: int,
    obs_mode: ObsMode | None,
    reward_mode: RewardMode | None,
):
    # Import tardio: executa DENTRO do subprocesso do SubprocVecEnv.
    from traffic_rl.envs import make_env

    env = make_env(
        cfg, scenario,
        traffic_seed_base=traffic_seed_base,
        obs_mode=obs_mode,
        reward_mode=reward_mode,
    )
    return Monitor(env)


def run_tag(cfg: ProjectConfig, obs_mode: ObsMode | None, reward_mode: RewardMode | None) -> str:
    obs = obs_mode or cfg.env.obs_mode
    rew = reward_mode or cfg.env.reward_mode
    return f"dqn_{cfg.train.scenario}_{obs}_{rew}"


def _latest_checkpoint(ckpt_dir: Path) -> tuple[Path, int] | None:
    best: tuple[Path, int] | None = None
    for p in ckpt_dir.glob("ckpt_*_steps.zip"):
        try:
            steps = int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        if best is None or steps > best[1]:
            best = (p, steps)
    return best


def _dump_eval_csv(run_dir: Path) -> None:
    npz = run_dir / "evaluations.npz"
    if not npz.exists():
        return
    data = np.load(npz)
    with open(run_dir / "eval_log.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timesteps", "recompensa_media", "recompensa_std", "duracao_media"])
        for i, t in enumerate(data["timesteps"]):
            results = data["results"][i]
            lengths = data["ep_lengths"][i]
            writer.writerow(
                [int(t), float(np.mean(results)), float(np.std(results)), float(np.mean(lengths))]
            )


def train_seed(
    cfg: ProjectConfig,
    seed: int,
    obs_mode: ObsMode | None = None,
    reward_mode: RewardMode | None = None,
    tag: str | None = None,
) -> Path:
    """Treina (ou retoma) uma seed. Retorna o diretório do run."""
    tag = tag or run_tag(cfg, obs_mode, reward_mode)
    run_dir = runs_dir() / tag / f"seed_{seed}"
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)

    (run_dir / "git_hash.txt").write_text(_git_hash() + "\n", encoding="utf-8")
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.model_dump(mode="json"), f, allow_unicode=True, sort_keys=False)

    scenario = cfg.train.scenario
    # start_method="fork": barato no Linux e imune a problemas de reimport do
    # __main__; cada filho tem seu próprio libsumo (1 simulação por processo).
    train_env = SubprocVecEnv(
        [partial(_env_factory, cfg, scenario, seed * TRAIN_SEED_STRIDE, obs_mode, reward_mode)],
        start_method="fork",
    )
    eval_env = SubprocVecEnv(
        [
            partial(
                _env_factory, cfg, scenario, EVAL_DURING_TRAIN_BASE + seed, obs_mode, reward_mode
            )
        ],
        start_method="fork",
    )
    try:
        dqn_cfg = cfg.train.dqn
        resume = _latest_checkpoint(ckpt_dir)
        algo = DoubleDQN if dqn_cfg.double_dqn else DQN
        if resume is not None:
            ckpt_path, done_steps = resume
            print(f"[seed {seed}] retomando de {ckpt_path.name} ({done_steps} passos)")
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
            print(f"[seed {seed}] treino já completo ({cfg.train.total_timesteps} passos)")
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
            eval_freq=cfg.train.eval_freq,
            n_eval_episodes=cfg.train.n_eval_episodes,
            deterministic=True,
            callback_after_eval=stop_cb,
            verbose=0,
        )
        ckpt_cb = CheckpointCallback(
            save_freq=cfg.train.checkpoint_freq,
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
        # fallback: se a avaliação periódica nunca rodou (treino < eval_freq),
        # salva o modelo final como best_model para o pipeline seguir
        if not (run_dir / "best_model.zip").exists():
            model.save(run_dir / "best_model")
        _dump_eval_csv(run_dir)
    finally:
        train_env.close()
        eval_env.close()
    return run_dir


def find_best_models(tag: str) -> dict[int, Path]:
    """{seed: best_model.zip} dos runs de treino existentes de um tag."""
    models: dict[int, Path] = {}
    group = runs_dir() / tag
    for seed_dir in sorted(group.glob("seed_*")):
        best = seed_dir / "best_model.zip"
        if best.exists():
            try:
                models[int(seed_dir.name.removeprefix("seed_"))] = best
            except ValueError:
                continue
    return models


def train_all_seeds(
    cfg: ProjectConfig,
    obs_mode: ObsMode | None = None,
    reward_mode: RewardMode | None = None,
    seeds: list[int] | None = None,
) -> list[Path]:
    dirs = []
    for seed in seeds or cfg.train.seeds:
        print(f"=== treinando seed {seed} ({cfg.train.total_timesteps} passos) ===")
        dirs.append(train_seed(cfg, seed, obs_mode, reward_mode))
    return dirs
