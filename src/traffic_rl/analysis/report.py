"""Geração automática de results/REPORT.md.

Tabelas em markdown + parágrafos de interpretação POR TEMPLATE (regras
determinísticas sobre os números — nenhum texto vem de LLM), como exige a
spec. O relatório é reproduzível: mesmo CSV de métricas ⇒ mesmo relatório.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from traffic_rl.analysis.plots import (
    METHOD_LABELS,
    plot_bars_with_ci,
    plot_fairness_tradeoff,
    plot_learning_curves,
    plot_p95_boxplot,
)
from traffic_rl.analysis.stats import (
    METRIC_LABELS,
    PRIMARY_METRIC,
    bootstrap_ci,
    dqn_vs_baselines,
)

TABLE_METRICS = [
    "espera_media_s",
    "espera_p95_s",
    "espera_max_s",
    "fila_maxima",
    "throughput",
    "tempo_viagem_medio_s",
]


def _fmt_ci(values) -> str:
    mean, lo, hi = bootstrap_ci(values.to_numpy())
    return f"{mean:.1f} [{lo:.1f}, {hi:.1f}]"


def _scenario_table(df: pd.DataFrame, scenario: str) -> str:
    sub = df[df["scenario"] == scenario]
    methods = [m for m in METHOD_LABELS if m in set(sub["method"])]
    header = "| Método | " + " | ".join(METRIC_LABELS[m] for m in TABLE_METRICS) + " |"
    sep = "|" + "---|" * (len(TABLE_METRICS) + 1)
    lines = [header, sep]
    for method in methods:
        g = sub[sub["method"] == method]
        cells = [_fmt_ci(g[m]) for m in TABLE_METRICS]
        lines.append(f"| {METHOD_LABELS[method]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _pvalue_table(tests: pd.DataFrame) -> str:
    if tests.empty:
        return "_Sem comparações (DQN ausente do conjunto avaliado)._"
    lines = [
        "| Cenário | Baseline | p (bruto) | p (Bonferroni) "
        "| Significativo (α=0,05) | DQN melhor? |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in tests.iterrows():
        lines.append(
            f"| {r['scenario']} | {METHOD_LABELS.get(r['baseline'], r['baseline'])} "
            f"| {r['p_value']:.4f} | {r['p_bonferroni']:.4f} "
            f"| {'sim' if r['significativo'] else 'não'} "
            f"| {'sim' if r['dqn_melhor'] else 'não'} |"
        )
    return "\n".join(lines)


def _interpret_scenario(tests: pd.DataFrame, scenario: str) -> str:
    """Parágrafo de interpretação por regras (template, não LLM)."""
    sub = tests[tests["scenario"] == scenario]
    if sub.empty:
        return ""
    frases: list[str] = []
    for _, r in sub.iterrows():
        nome = METHOD_LABELS.get(r["baseline"], r["baseline"])
        base_mean = max(r["baseline_mean"], 1e-9)
        delta = 100.0 * (r["baseline_mean"] - r["dqn_mean"]) / base_mean
        if r["significativo"] and r["dqn_melhor"]:
            frases.append(
                f"o DQN reduziu a espera média em {delta:.0f}% frente ao {nome} "
                f"(p corrigido = {r['p_bonferroni']:.4f})"
            )
        elif r["significativo"] and not r["dqn_melhor"]:
            frases.append(
                f"o {nome} superou o DQN ({-delta:.0f}% de vantagem, "
                f"p corrigido = {r['p_bonferroni']:.4f}) — resultado reportado honestamente"
            )
        else:
            frases.append(f"não houve diferença significativa frente ao {nome} "
                          f"(p corrigido = {r['p_bonferroni']:.4f})")
    corpo = "; ".join(frases)
    return (
        f"**Interpretação ({scenario}):** com correção de Bonferroni, {corpo}. "
        "Lembrete metodológico: o mérito real do agente é medido contra os baselines "
        "fortes (atuado por gap e max-pressure), não apenas contra o tempo fixo."
    )


def generate_report(
    metrics_df: pd.DataFrame,
    out_dir: Path,
    eval_curves: dict[int, pd.DataFrame] | None = None,
    git_hash: str = "desconhecido",
    title_suffix: str = "",
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    figures: list[Path] = []
    figures.append(plot_bars_with_ci(metrics_df, fig_dir, PRIMARY_METRIC))
    figures.append(plot_p95_boxplot(metrics_df, fig_dir))
    for scenario in sorted(metrics_df["scenario"].unique()):
        figures.append(plot_fairness_tradeoff(metrics_df, fig_dir, scenario))
    if eval_curves:
        figures.append(plot_learning_curves(eval_curves, fig_dir))

    tests = dqn_vs_baselines(metrics_df, PRIMARY_METRIC)

    lines: list[str] = []
    lines.append(f"# TrafficRL — Relatório de avaliação{title_suffix}")
    lines.append("")
    lines.append(f"- Gerado em: {dt.datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Git hash: `{git_hash}`")
    n_eps = metrics_df.groupby(["scenario", "method"]).size().max()
    lines.append(f"- Episódios por método × cenário (máx.): {n_eps}")
    lines.append(
        "- Valores: média [IC 95% via bootstrap]. Testes: Mann-Whitney U bicaudal, "
        "correção de Bonferroni por cenário."
    )
    lines.append("")

    for scenario in sorted(metrics_df["scenario"].unique()):
        lines.append(f"## Cenário: {scenario}")
        lines.append("")
        lines.append(_scenario_table(metrics_df, scenario))
        lines.append("")
        interp = _interpret_scenario(tests, scenario)
        if interp:
            lines.append(interp)
            lines.append("")

    lines.append("## Testes estatísticos — DQN vs baselines (espera média)")
    lines.append("")
    lines.append(_pvalue_table(tests))
    lines.append("")
    lines.append("## Figuras")
    lines.append("")
    for fig in figures:
        rel = fig.relative_to(out_dir)
        lines.append(f"![{fig.stem}]({rel})")
    lines.append("")

    report_path = out_dir / "REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def load_eval_curves(run_group_dir: Path) -> dict[int, pd.DataFrame]:
    """Lê eval_log.csv de cada seed_*/ de um grupo de runs de treino."""
    curves: dict[int, pd.DataFrame] = {}
    for seed_dir in sorted(run_group_dir.glob("seed_*")):
        csv = seed_dir / "eval_log.csv"
        if csv.exists():
            try:
                seed = int(seed_dir.name.removeprefix("seed_"))
            except ValueError:
                continue
            curves[seed] = pd.read_csv(csv)
    return curves
