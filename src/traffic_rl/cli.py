"""CLI do projeto: traffic-rl {train, evaluate, compare, ablation}.

Todos os comandos aceitam `--config` (default configs/default.yaml) e
`--smoke`, que aplica configs/smoke.yaml por cima — parâmetros reduzidos para
validar o pipeline de ponta a ponta em minutos.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from traffic_rl.config import ProjectConfig, load_config
from traffic_rl.paths import configs_dir, results_dir, runs_dir


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", type=Path, default=None, help="YAML de configuração")
    p.add_argument(
        "--smoke", action="store_true",
        help="aplica configs/smoke.yaml (pipeline reduzido de validação)",
    )


def _load(args: argparse.Namespace) -> ProjectConfig:
    config_path = args.config or (configs_dir() / "default.yaml")
    overrides = None
    if args.smoke:
        with open(configs_dir() / "smoke.yaml", encoding="utf-8") as f:
            overrides = yaml.safe_load(f)
    return load_config(config_path, overrides)


def cmd_train(args: argparse.Namespace) -> int:
    from traffic_rl.training.train import run_tag, train_all_seeds

    cfg = _load(args)
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    dirs = train_all_seeds(cfg, obs_mode=args.obs_mode, reward_mode=args.reward_mode, seeds=seeds)
    print(f"treino concluído — tag: {run_tag(cfg, args.obs_mode, args.reward_mode)}")
    for d in dirs:
        print(" -", d)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from traffic_rl.controllers.registry import BASELINE_NAMES
    from traffic_rl.evaluation.runner import run_protocol
    from traffic_rl.training.train import find_best_models, run_tag

    cfg = _load(args)
    methods = args.methods.split(",") if args.methods else list(BASELINE_NAMES)
    dqn_models = None
    if "dqn" in methods:
        tag = args.dqn_tag or run_tag(cfg, args.obs_mode, args.reward_mode)
        dqn_models = find_best_models(tag)
        if not dqn_models:
            print(f"erro: nenhum best_model.zip em {runs_dir() / tag}/seed_*/ — treine antes.",
                  file=sys.stderr)
            return 1
    out_dir = args.out or (results_dir() / "evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)
    df = run_protocol(cfg, methods, dqn_models=dqn_models, dqn_obs_mode=args.obs_mode)
    df.to_csv(out_dir / "metrics.csv", index=False)
    print(f"métricas: {out_dir / 'metrics.csv'} ({len(df)} episódios)")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Protocolo completo (baselines + DQN) + REPORT.md final."""
    from traffic_rl.analysis.report import generate_report, load_eval_curves
    from traffic_rl.controllers.registry import BASELINE_NAMES
    from traffic_rl.evaluation.runner import run_protocol
    from traffic_rl.training.train import _git_hash, find_best_models, run_tag

    cfg = _load(args)
    tag = args.dqn_tag or run_tag(cfg, args.obs_mode, args.reward_mode)
    dqn_models = find_best_models(tag)
    methods = list(BASELINE_NAMES)
    if dqn_models:
        methods.append("dqn")
    else:
        print(f"aviso: sem modelos DQN em {runs_dir() / tag} — comparando só baselines.")
    df = run_protocol(cfg, methods, dqn_models=dqn_models or None, dqn_obs_mode=args.obs_mode)
    out_dir = args.out or results_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "metrics.csv", index=False)
    curves = load_eval_curves(runs_dir() / tag)
    report = generate_report(
        df, out_dir, eval_curves=curves or None, git_hash=_git_hash(),
        title_suffix=" (smoke)" if args.smoke else "",
    )
    print(f"relatório: {report}")
    return 0


def _load_grid(args: argparse.Namespace):
    from traffic_rl.grid_config import load_grid_config

    grid_path = args.grid or (configs_dir() / "grid.yaml")
    overrides = None
    if args.smoke:
        with open(configs_dir() / "smoke.yaml", encoding="utf-8") as f:
            overrides = yaml.safe_load(f)
    return load_grid_config(grid_path, overrides)


def cmd_grid_train(args: argparse.Namespace) -> int:
    from traffic_rl.training.grid_train import grid_run_tag, train_grid_all_seeds

    cfg = _load_grid(args)
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    dirs = train_grid_all_seeds(cfg, seeds=seeds)
    print(f"treino do grid concluído — tag: {grid_run_tag(cfg)}")
    for d in dirs:
        print(" -", d)
    return 0


def cmd_grid_compare(args: argparse.Namespace) -> int:
    from traffic_rl.analysis.report import generate_report, load_eval_curves
    from traffic_rl.controllers.grid import GRID_BASELINES
    from traffic_rl.evaluation.grid_runner import run_grid_protocol
    from traffic_rl.training.grid_train import find_grid_best_models, grid_run_tag
    from traffic_rl.training.train import _git_hash

    cfg = _load_grid(args)
    tag = grid_run_tag(cfg)
    dqn_models = find_grid_best_models(tag)
    methods = list(GRID_BASELINES)
    if dqn_models:
        methods.append("dqn")
    else:
        print(f"aviso: sem modelos DQN em {runs_dir() / tag} — comparando só baselines.")
    df = run_grid_protocol(cfg, methods, dqn_models=dqn_models or None)
    out_dir = args.out or (results_dir() / "grid")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "metrics.csv", index=False)
    curves = load_eval_curves(runs_dir() / tag)
    report = generate_report(
        df, out_dir, eval_curves=curves or None, git_hash=_git_hash(),
        title_suffix=" — grid (smoke)" if args.smoke else " — grid",
    )
    print(f"relatório do grid: {report}")
    return 0


def cmd_ablation(args: argparse.Namespace) -> int:
    from traffic_rl.evaluation.ablation import run_ablation

    cfg = _load(args)
    kinds = ["obs", "reward"] if args.type == "all" else [args.type]
    for kind in kinds:
        report = run_ablation(cfg, kind, skip_training=args.skip_training)
        print(f"ablação {kind}: {report}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="traffic-rl", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="treina o DQN (multi-seed, retomável)")
    _add_common(p_train)
    p_train.add_argument("--seeds", help="lista separada por vírgula (default: config)")
    p_train.add_argument("--obs-mode", choices=["obs_minimal", "obs_rica"])
    p_train.add_argument("--reward-mode", choices=["r_espera", "r_fila", "r_pressao"])
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate", help="avalia métodos e salva metrics.csv")
    _add_common(p_eval)
    p_eval.add_argument("--methods", help="ex.: fixo_igual,max_pressure,dqn (default: baselines)")
    p_eval.add_argument("--dqn-tag", help="tag do treino (default: derivado da config)")
    p_eval.add_argument("--obs-mode", choices=["obs_minimal", "obs_rica"])
    p_eval.add_argument("--reward-mode", choices=["r_espera", "r_fila", "r_pressao"])
    p_eval.add_argument("--out", type=Path)
    p_eval.set_defaults(func=cmd_evaluate)

    p_cmp = sub.add_parser("compare", help="protocolo completo + REPORT.md")
    _add_common(p_cmp)
    p_cmp.add_argument("--dqn-tag")
    p_cmp.add_argument("--obs-mode", choices=["obs_minimal", "obs_rica"])
    p_cmp.add_argument("--reward-mode", choices=["r_espera", "r_fila", "r_pressao"])
    p_cmp.add_argument("--out", type=Path)
    p_cmp.set_defaults(func=cmd_compare)

    p_gt = sub.add_parser("grid-train", help="Fase 2: treina a política compartilhada no grid")
    p_gt.add_argument("--grid", type=Path, help="YAML do grid (default: configs/grid.yaml)")
    p_gt.add_argument("--smoke", action="store_true")
    p_gt.add_argument("--seeds", help="lista separada por vírgula (default: config)")
    p_gt.set_defaults(func=cmd_grid_train)

    p_gc = sub.add_parser("grid-compare", help="Fase 2: baselines + DQN no grid + REPORT.md")
    p_gc.add_argument("--grid", type=Path, help="YAML do grid (default: configs/grid.yaml)")
    p_gc.add_argument("--smoke", action="store_true")
    p_gc.add_argument("--out", type=Path)
    p_gc.set_defaults(func=cmd_grid_compare)

    p_abl = sub.add_parser("ablation", help="ablações de observação e recompensa")
    _add_common(p_abl)
    p_abl.add_argument("--type", choices=["obs", "reward", "all"], default="all")
    p_abl.add_argument("--skip-training", action="store_true",
                       help="só avalia modelos já treinados")
    p_abl.set_defaults(func=cmd_ablation)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
