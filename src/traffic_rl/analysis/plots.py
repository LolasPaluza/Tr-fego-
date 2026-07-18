"""Gráficos do relatório (rótulos em português, prontos para apresentação).

Todos recebem DataFrames e um diretório de saída; nenhum estado global além
do estilo. Excluídos da meta de cobertura (código de plot).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: sandbox cloud e servidores sem display

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from traffic_rl.analysis.stats import METRIC_LABELS, bootstrap_ci, summary_table

METHOD_LABELS = {
    "fixo_igual": "Fixo igual",
    "fixo_proporcional": "Fixo proporcional",
    "atuado_gap": "Atuado (gap)",
    "max_pressure": "Max-pressure",
    "dqn": "DQN (nosso)",
}
METHOD_ORDER = ["fixo_igual", "fixo_proporcional", "atuado_gap", "max_pressure", "dqn"]


def _style() -> None:
    sns.set_theme(style="whitegrid", font_scale=1.05)


def _method_order(df: pd.DataFrame) -> list[str]:
    return [m for m in METHOD_ORDER if m in set(df["method"])]


def plot_bars_with_ci(df: pd.DataFrame, out_dir: Path, metric: str = "espera_media_s") -> Path:
    """Barras (média ± IC95 bootstrap) por cenário × método."""
    _style()
    table = summary_table(df, metric)
    scenarios = sorted(table["scenario"].unique())
    methods = _method_order(df)
    fig, axes = plt.subplots(1, len(scenarios), figsize=(4.2 * len(scenarios), 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    palette = sns.color_palette("deep", len(methods))
    for ax, scenario in zip(axes, scenarios, strict=False):
        sub = table[table["scenario"] == scenario].set_index("method").reindex(methods)
        x = np.arange(len(methods))
        err = np.array([sub["mean"] - sub["ci_low"], sub["ci_high"] - sub["mean"]])
        ax.bar(x, sub["mean"], yerr=err, capsize=4, color=palette)
        ax.set_xticks(x, [METHOD_LABELS.get(m, m) for m in methods], rotation=30, ha="right")
        ax.set_title(scenario)
        ax.set_xlabel("")
    axes[0].set_ylabel(METRIC_LABELS.get(metric, metric))
    fig.suptitle(f"{METRIC_LABELS.get(metric, metric)} por cenário (média ± IC 95%)")
    fig.tight_layout()
    out = out_dir / f"barras_{metric}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_learning_curves(eval_curves: dict[int, pd.DataFrame], out_dir: Path) -> Path:
    """Curvas de avaliação periódica das seeds (média ± sombra min-max).

    `eval_curves`: {seed: DataFrame(timesteps, recompensa_media)}.
    """
    _style()
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    common: pd.DataFrame | None = None
    for seed, df in sorted(eval_curves.items()):
        s = df.set_index("timesteps")["recompensa_media"].rename(f"seed {seed}")
        common = s.to_frame() if common is None else common.join(s, how="outer")
        ax.plot(df["timesteps"], df["recompensa_media"], alpha=0.35, linewidth=1.0)
    if common is not None and len(common.columns) > 1:
        mean = common.mean(axis=1)
        ax.plot(mean.index, mean.values, color="black", linewidth=2.2, label="média das seeds")
        ax.fill_between(
            common.index, common.min(axis=1), common.max(axis=1), alpha=0.15, color="black"
        )
        ax.legend()
    ax.set_xlabel("Passos de treino")
    ax.set_ylabel("Recompensa média (avaliação greedy)")
    ax.set_title("Curva de aprendizado — avaliação periódica por seed")
    fig.tight_layout()
    out = out_dir / "curvas_aprendizado.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_p95_boxplot(df: pd.DataFrame, out_dir: Path) -> Path:
    """Boxplot do p95 da espera (justiça) por método, facetado por cenário."""
    _style()
    methods = _method_order(df)
    g = sns.catplot(
        data=df, x="method", y="espera_p95_s", col="scenario", kind="box",
        order=methods, col_wrap=2, height=3.6, aspect=1.25,
    )
    g.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods], rotation=30, ha="right")
    g.set_axis_labels("", "Espera p95 (s)")
    g.set_titles("{col_name}")
    g.figure.suptitle("Justiça: p95 do tempo de espera por método", y=1.02)
    out = out_dir / "boxplot_p95_espera.png"
    g.figure.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(g.figure)
    return out


def plot_fairness_tradeoff(df: pd.DataFrame, out_dir: Path, scenario: str) -> Path:
    """Espera média na avenida vs na via local por método — o trade-off de
    justiça: priorizar a avenida não pode significar starvation da local."""
    _style()
    sub = df[df["scenario"] == scenario]
    methods = _method_order(sub)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    x = np.arange(len(methods))
    width = 0.38
    for offset, (col, label, color) in enumerate(
        [
            ("espera_media_avenida_s", "Avenida (L-O)", "#4C72B0"),
            ("espera_media_local_s", "Via local (N-S)", "#DD8452"),
        ]
    ):
        means, err_lo, err_hi = [], [], []
        for m in methods:
            mean, lo, hi = bootstrap_ci(sub.loc[sub["method"] == m, col].to_numpy())
            means.append(mean); err_lo.append(mean - lo); err_hi.append(hi - mean)
        ax.bar(
            x + (offset - 0.5) * width, means, width, yerr=[err_lo, err_hi],
            capsize=4, label=label, color=color,
        )
    ax.set_xticks(x, [METHOD_LABELS.get(m, m) for m in methods], rotation=30, ha="right")
    ax.set_ylabel("Espera média (s)")
    ax.set_title(f"Trade-off de justiça — espera por classe viária ({scenario})")
    ax.legend()
    fig.tight_layout()
    out = out_dir / f"justica_avenida_vs_local_{scenario}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
