"""As duas ablações obrigatórias: espaço de observação e função de recompensa.

Cada ablação treina o DQN por variante (mesmas seeds/timesteps da config),
avalia os best models nos 4 cenários e gera tabela própria com testes
pareados (Mann-Whitney + Bonferroni) em results/ablations/{tipo}/.

Hipóteses registradas (spec 1.3):
- obs_rica deve VENCER no pico_invertido (mais informação para generalizar
  além de "priorize a avenida") e EMPATAR no fora_pico (pouca demanda —
  informação extra não paga). A verificação é feita por regra sobre os testes.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Literal

import pandas as pd

from traffic_rl.analysis.stats import bootstrap_ci, pairwise_tests
from traffic_rl.config import ObsMode, ProjectConfig, RewardMode
from traffic_rl.evaluation.runner import run_protocol
from traffic_rl.paths import results_dir
from traffic_rl.training.train import _git_hash, find_best_models, train_seed

AblationKind = Literal["obs", "reward"]

OBS_VARIANTS: list[ObsMode] = ["obs_minimal", "obs_rica"]
REWARD_VARIANTS: list[RewardMode] = ["r_espera", "r_fila", "r_pressao"]


def _tag(kind: AblationKind, variant: str, scenario: str) -> str:
    return f"ablation_{kind}_{scenario}_{variant}"


def run_ablation(cfg: ProjectConfig, kind: AblationKind, skip_training: bool = False) -> Path:
    """Treina (se preciso) e avalia as variantes; retorna o caminho do relatório."""
    scenario_train = cfg.train.scenario
    variants = OBS_VARIANTS if kind == "obs" else REWARD_VARIANTS

    frames: list[pd.DataFrame] = []
    for variant in variants:
        tag = _tag(kind, variant, scenario_train)
        obs_mode: ObsMode | None = variant if kind == "obs" else None  # type: ignore[assignment]
        reward_mode: RewardMode | None = variant if kind == "reward" else None  # type: ignore[assignment]
        models = find_best_models(tag)
        missing = [s for s in cfg.train.seeds if s not in models]
        if missing:
            if skip_training:
                raise FileNotFoundError(f"sem modelos treinados para {tag} (seeds {missing})")
            print(f"[ablação {kind}] treinando variante {variant} (seeds {missing})")
            for seed in missing:
                train_seed(cfg, seed, obs_mode=obs_mode, reward_mode=reward_mode, tag=tag)
            models = find_best_models(tag)
        print(f"[ablação {kind}] avaliando {variant} ({len(models)} seeds)")
        df = run_protocol(
            cfg, ["dqn"], dqn_models=models, dqn_obs_mode=obs_mode, verbose=False
        )
        df["method"] = f"dqn_{variant}"
        frames.append(df)

    metrics = pd.concat(frames, ignore_index=True)
    out_dir = results_dir() / "ablations" / kind
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    report = _write_ablation_report(metrics, out_dir, kind)
    return report


def _write_ablation_report(metrics: pd.DataFrame, out_dir: Path, kind: AblationKind) -> Path:
    tests = pairwise_tests(metrics)
    lines = [
        f"# Ablação — {'espaço de observação' if kind == 'obs' else 'função de recompensa'}",
        "",
        f"- Gerado em: {dt.datetime.now().isoformat(timespec='seconds')}",
        f"- Git hash: `{_git_hash()}`",
        "- Valores: espera média em s, média [IC 95% bootstrap]; testes Mann-Whitney U "
        "com Bonferroni por cenário.",
        "",
        "## Espera média por cenário",
        "",
    ]
    scenarios = sorted(metrics["scenario"].unique())
    methods = sorted(metrics["method"].unique())
    lines.append("| Variante | " + " | ".join(scenarios) + " |")
    lines.append("|" + "---|" * (len(scenarios) + 1))
    for m in methods:
        cells = []
        for sc in scenarios:
            vals = metrics.loc[
                (metrics["method"] == m) & (metrics["scenario"] == sc), "espera_media_s"
            ]
            mean, lo, hi = bootstrap_ci(vals.to_numpy())
            cells.append(f"{mean:.1f} [{lo:.1f}, {hi:.1f}]")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines += ["", "## Testes pareados (espera média)", ""]
    if tests.empty:
        lines.append("_Sem pares suficientes para teste._")
    else:
        lines.append("| Cenário | A | B | p (Bonferroni) | Significativo |")
        lines.append("|---|---|---|---|---|")
        for _, r in tests.iterrows():
            lines.append(
                f"| {r['scenario']} | {r['method_a']} | {r['method_b']} "
                f"| {r['p_bonferroni']:.4f} | {'sim' if r['significativo'] else 'não'} |"
            )
    if kind == "obs":
        lines += ["", _obs_hypothesis_check(metrics, tests)]
    lines.append("")
    report = out_dir / "REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def _obs_hypothesis_check(metrics: pd.DataFrame, tests: pd.DataFrame) -> str:
    """Verificação por regra da hipótese 1.3 (template, não LLM)."""
    def verdict(scenario: str) -> str:
        sub = tests[
            (tests["scenario"] == scenario)
            & (tests["method_a"] == "dqn_obs_minimal")
            & (tests["method_b"] == "dqn_obs_rica")
        ]
        if sub.empty:
            return f"- {scenario}: sem dados para testar."
        r = sub.iloc[0]
        rica_melhor = r["mean_b"] < r["mean_a"]
        if scenario == "pico_invertido":
            esperado = "obs_rica vence"
            ok = bool(r["significativo"] and rica_melhor)
        else:
            esperado = "empate"
            ok = not bool(r["significativo"])
        obtido = (
            "empate estatístico" if not r["significativo"]
            else ("obs_rica venceu" if rica_melhor else "obs_minimal venceu")
        )
        status = "CONFIRMADA" if ok else "NÃO confirmada"
        return (
            f"- {scenario}: esperado {esperado}; obtido {obtido} "
            f"(p corrigido = {r['p_bonferroni']:.4f}) → hipótese {status}."
        )

    return (
        "## Verificação da hipótese (1.3)\n\n"
        "Hipótese: obs_rica vence no pico_invertido; empata no fora_pico.\n\n"
        + verdict("pico_invertido") + "\n" + verdict("fora_pico")
    )
