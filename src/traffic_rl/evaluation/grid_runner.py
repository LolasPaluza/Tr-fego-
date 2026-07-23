"""Protocolo de avaliação da Fase 2: método × episódios no SEU grid.

O "cenário" é o próprio grid.yaml (as ruas e fluxos que você declarou);
as seeds de tráfego seguem a mesma disciplina da Fase 1 (base 10000+i,
disjuntas do treino). As métricas por episódio são as mesmas — a espera por
classe viária agora separa avenidas × locais conforme a classe declarada
de cada rua de origem do veículo.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from traffic_rl.envs.grid_demand import grid_road_class_of_vehicle
from traffic_rl.envs.grid_env import GridTrafficEnv
from traffic_rl.evaluation.metrics import parse_tripinfo
from traffic_rl.grid_config import GridProjectConfig

GRID_SCENARIO_NAME = "grid"


def evaluate_grid_controller(
    cfg: GridProjectConfig,
    method_name: str,
    model_path: Path | None = None,
    n_episodes: int | None = None,
    traffic_seed_base: int | None = None,
) -> pd.DataFrame:
    from traffic_rl.controllers.grid import make_grid_controllers

    n_episodes = n_episodes if n_episodes is not None else cfg.eval.n_episodes
    seed_base = (
        traffic_seed_base if traffic_seed_base is not None else cfg.eval.traffic_seed_base
    )
    coordinated_dqn = method_name == "dqn" and getattr(cfg.env, "coordination", False)
    rows = []
    with tempfile.TemporaryDirectory(prefix="tripinfo_grid_") as tmp:
        env = GridTrafficEnv(cfg, tripinfo_dir=tmp)
        try:
            model = None
            controllers = None
            if coordinated_dqn:
                # a política coordenada precisa do vetor com vizinhos (13d),
                # que só o env monta — não passa pelo contrato act(obs) local
                from stable_baselines3 import DQN

                model = DQN.load(str(model_path), device="cpu")
            else:
                controllers = make_grid_controllers(method_name, cfg, env, model_path)
            for ep in range(n_episodes):
                traffic_seed = seed_base + ep
                if controllers is not None:
                    for ctrl in controllers:
                        ctrl.reset()
                observations, info = env.reset(traffic_seed)
                done = False
                while not done:
                    if coordinated_dqn:
                        vecs = env.vectorize(observations)
                        acts, _ = model.predict(vecs, deterministic=True)
                        actions = [int(a) for a in acts]
                    else:
                        actions = [
                            ctrl.act(obs)
                            for ctrl, obs in zip(controllers, observations, strict=True)
                        ]
                    observations, _rewards, done, info = env.step(actions)
                fila_maxima = float(info["episode_max_queue"])
                tripinfo_path = info["tripinfo_path"]
                env.close()  # flush do tripinfo
                rows.append(
                    parse_tripinfo(
                        tripinfo_path,
                        method=method_name,
                        scenario=GRID_SCENARIO_NAME,
                        episode=ep,
                        traffic_seed=traffic_seed,
                        fila_maxima=fila_maxima,
                        classifier=grid_road_class_of_vehicle,
                    )
                )
        finally:
            env.close()
    return pd.DataFrame([r.as_dict() for r in rows])


def run_grid_protocol(
    cfg: GridProjectConfig,
    methods: list[str],
    dqn_models: dict[int, Path] | None = None,
    n_episodes: int | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for method in methods:
        if method == "dqn":
            if not dqn_models:
                raise ValueError("método 'dqn' exige dqn_models {seed: best_model.zip}")
            for seed, model_path in sorted(dqn_models.items()):
                if verbose:
                    print(f"[grid avaliação] dqn(seed {seed})")
                df = evaluate_grid_controller(cfg, "dqn", model_path, n_episodes)
                df["dqn_seed"] = seed
                frames.append(df)
        else:
            if verbose:
                print(f"[grid avaliação] {method}")
            df = evaluate_grid_controller(cfg, method, None, n_episodes)
            df["dqn_seed"] = pd.NA
            frames.append(df)
    return pd.concat(frames, ignore_index=True)
