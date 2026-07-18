"""Estatística do protocolo: bootstrap para IC 95% e Mann-Whitney U com
correção de Bonferroni para DQN vs cada baseline, por cenário.

Escolhas (justificativa em docs/DECISOES.md):
- IC 95% por bootstrap (10k reamostragens) — não assume normalidade;
- Mann-Whitney U (bicaudal) — teste não-paramétrico para amostras pequenas
  (20 episódios) e distribuições assimétricas de espera;
- Bonferroni sobre o número de comparações DQN×baseline dentro de cada
  métrica (4 comparações por cenário) — conservador e simples de auditar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

PRIMARY_METRIC = "espera_media_s"
METRIC_LABELS: dict[str, str] = {
    "espera_media_s": "Espera média (s)",
    "espera_p95_s": "Espera p95 (s)",
    "espera_max_s": "Espera máxima (s)",
    "fila_maxima": "Fila máxima (veíc.)",
    "throughput": "Throughput (veíc.)",
    "tempo_viagem_medio_s": "Tempo de viagem médio (s)",
    "espera_media_avenida_s": "Espera média — avenida (s)",
    "espera_media_local_s": "Espera média — via local (s)",
}


def bootstrap_ci(
    values: np.ndarray, n_boot: int = 10_000, ci: float = 95.0, seed: int = 0
) -> tuple[float, float, float]:
    """(média, IC inferior, IC superior) via bootstrap percentílico."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return (float("nan"),) * 3
    if len(values) == 1:
        return float(values[0]), float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    alpha = (100.0 - ci) / 2.0
    return float(values.mean()), float(np.percentile(means, alpha)), float(
        np.percentile(means, 100.0 - alpha)
    )


def summary_table(df: pd.DataFrame, metric: str = PRIMARY_METRIC) -> pd.DataFrame:
    """média ± IC95 por (cenário, método) para uma métrica."""
    rows = []
    for (scenario, method), group in df.groupby(["scenario", "method"]):
        mean, lo, hi = bootstrap_ci(group[metric].to_numpy())
        rows.append(
            {"scenario": scenario, "method": method, "mean": mean, "ci_low": lo, "ci_high": hi,
             "n": len(group)}
        )
    return pd.DataFrame(rows)


_TEST_COLUMNS = [
    "scenario", "baseline", "p_value", "p_bonferroni", "significativo",
    "dqn_mean", "baseline_mean", "dqn_melhor",
]


def dqn_vs_baselines(
    df: pd.DataFrame, metric: str = PRIMARY_METRIC, alpha: float = 0.05
) -> pd.DataFrame:
    """Mann-Whitney U (DQN vs baseline) por cenário, p corrigido (Bonferroni).

    Retorna colunas: scenario, baseline, p_value, p_bonferroni, significativo,
    dqn_mean, baseline_mean, dqn_melhor (menor é melhor, exceto throughput).
    """
    rows = []
    higher_is_better = metric == "throughput"
    for scenario, group in df.groupby("scenario"):
        dqn_vals = group.loc[group["method"] == "dqn", metric].to_numpy()
        baselines = sorted(m for m in group["method"].unique() if m != "dqn")
        n_comp = max(len(baselines), 1)
        for baseline in baselines:
            base_vals = group.loc[group["method"] == baseline, metric].to_numpy()
            if len(dqn_vals) == 0 or len(base_vals) == 0:
                continue
            stat = scipy_stats.mannwhitneyu(dqn_vals, base_vals, alternative="two-sided")
            p_corr = min(stat.pvalue * n_comp, 1.0)
            dqn_mean, base_mean = float(dqn_vals.mean()), float(base_vals.mean())
            better = dqn_mean > base_mean if higher_is_better else dqn_mean < base_mean
            rows.append(
                {
                    "scenario": scenario,
                    "baseline": baseline,
                    "p_value": float(stat.pvalue),
                    "p_bonferroni": float(p_corr),
                    "significativo": bool(p_corr < alpha),
                    "dqn_mean": dqn_mean,
                    "baseline_mean": base_mean,
                    "dqn_melhor": bool(better),
                }
            )
    return pd.DataFrame(rows, columns=_TEST_COLUMNS)


_PAIR_COLUMNS = [
    "scenario", "method_a", "method_b", "p_value", "p_bonferroni",
    "significativo", "mean_a", "mean_b",
]


def pairwise_tests(
    df: pd.DataFrame, metric: str = PRIMARY_METRIC, alpha: float = 0.05
) -> pd.DataFrame:
    """Mann-Whitney U entre TODOS os pares de métodos, por cenário, com
    Bonferroni sobre o nº de pares — usado nas tabelas das ablações."""
    from itertools import combinations

    rows = []
    for scenario, group in df.groupby("scenario"):
        methods = sorted(group["method"].unique())
        pairs = list(combinations(methods, 2))
        for a, b in pairs:
            va = group.loc[group["method"] == a, metric].to_numpy()
            vb = group.loc[group["method"] == b, metric].to_numpy()
            if len(va) == 0 or len(vb) == 0:
                continue
            stat = scipy_stats.mannwhitneyu(va, vb, alternative="two-sided")
            p_corr = min(stat.pvalue * max(len(pairs), 1), 1.0)
            rows.append(
                {
                    "scenario": scenario, "method_a": a, "method_b": b,
                    "p_value": float(stat.pvalue), "p_bonferroni": float(p_corr),
                    "significativo": bool(p_corr < alpha),
                    "mean_a": float(va.mean()), "mean_b": float(vb.mean()),
                }
            )
    return pd.DataFrame(rows, columns=_PAIR_COLUMNS)
