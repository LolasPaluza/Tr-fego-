"""Protocolo de avaliação: método × cenário × episódios com seeds controladas.

Todos os métodos (baselines e DQN) passam pelo MESMO loop, via a interface
`Controller.act(Observation)`. Seeds de tráfego: `traffic_seed_base + i`
(padrão 10_000 — disjuntas das seeds de treino e da avaliação periódica).

O tripinfo só é escrito completo quando o SUMO fecha, então cada episódio é
fechado antes do parsing (fechar e reabrir o SUMO custa ~0,1 s com libsumo).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from traffic_rl.config import ObsMode, ProjectConfig
from traffic_rl.controllers import make_controller
from traffic_rl.controllers.base import Controller
from traffic_rl.envs import make_env
from traffic_rl.evaluation.metrics import EpisodeMetrics, parse_tripinfo


def evaluate_controller(
    cfg: ProjectConfig,
    controller: Controller,
    method_name: str,
    scenario_name: str,
    n_episodes: int | None = None,
    traffic_seed_base: int | None = None,
) -> pd.DataFrame:
    """Roda `n_episodes` e retorna um DataFrame de métricas por episódio."""
    n_episodes = n_episodes if n_episodes is not None else cfg.eval.n_episodes
    seed_base = (
        traffic_seed_base if traffic_seed_base is not None else cfg.eval.traffic_seed_base
    )
    rows: list[EpisodeMetrics] = []
    with tempfile.TemporaryDirectory(prefix="tripinfo_") as tmp:
        env = make_env(cfg, scenario_name, tripinfo_dir=tmp)
        try:
            for ep in range(n_episodes):
                traffic_seed = seed_base + ep
                controller.reset()
                _, info = env.reset(options={"traffic_seed": traffic_seed})
                obs = info["observation"]
                done = False
                while not done:
                    _, _, term, trunc, info = env.step(controller.act(obs))
                    obs = info["observation"]
                    done = term or trunc
                fila_maxima = float(info["episode_max_queue"])
                tripinfo_path = info["tripinfo_path"]
                env.close()  # fecha o SUMO -> tripinfo completo em disco
                rows.append(
                    parse_tripinfo(
                        tripinfo_path,
                        method=method_name,
                        scenario=scenario_name,
                        episode=ep,
                        traffic_seed=traffic_seed,
                        fila_maxima=fila_maxima,
                    )
                )
        finally:
            env.close()
    return pd.DataFrame([r.as_dict() for r in rows])


def run_protocol(
    cfg: ProjectConfig,
    methods: list[str],
    dqn_models: dict[int, Path] | None = None,
    dqn_obs_mode: ObsMode | None = None,
    scenarios: list[str] | None = None,
    n_episodes: int | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Protocolo completo. Para o método 'dqn', avalia o best model DE CADA
    seed de treino (`dqn_models: {seed: caminho}`) e agrega os episódios com
    a coluna `dqn_seed` — a variabilidade entre seeds fica preservada nos
    dados e é tratada na análise estatística."""
    scenarios = scenarios or cfg.eval.scenarios
    frames: list[pd.DataFrame] = []
    for scenario in scenarios:
        for method in methods:
            if method == "dqn":
                if not dqn_models:
                    raise ValueError("método 'dqn' exige dqn_models {seed: best_model.zip}")
                for seed, model_path in sorted(dqn_models.items()):
                    ctrl = make_controller(
                        "dqn", cfg, scenario, model_path=model_path, obs_mode=dqn_obs_mode
                    )
                    if verbose:
                        print(f"[avaliação] dqn(seed {seed}) × {scenario}")
                    df = evaluate_controller(cfg, ctrl, "dqn", scenario, n_episodes)
                    df["dqn_seed"] = seed
                    frames.append(df)
            else:
                ctrl = make_controller(method, cfg, scenario)
                if verbose:
                    print(f"[avaliação] {method} × {scenario}")
                df = evaluate_controller(cfg, ctrl, method, scenario, n_episodes)
                df["dqn_seed"] = pd.NA
                frames.append(df)
    return pd.concat(frames, ignore_index=True)
