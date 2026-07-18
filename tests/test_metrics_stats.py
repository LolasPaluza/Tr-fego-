import numpy as np
import pandas as pd

from traffic_rl.analysis.stats import bootstrap_ci, dqn_vs_baselines, pairwise_tests
from traffic_rl.evaluation.metrics import parse_tripinfo

TRIPINFO_XML = """<tripinfos>
    <tripinfo id="E_s_carro.0" depart="0.0" arrival="100.0" duration="95.0" waitingTime="10.0"/>
    <tripinfo id="E_s_carro.1" depart="5.0" arrival="130.0" duration="120.0" waitingTime="30.0"/>
    <tripinfo id="N_l_carro.0" depart="8.0" arrival="-1" duration="200.0" waitingTime="180.0"/>
    <tripinfo id="S_s_caminhao.0" depart="9.0" arrival="220.0" duration="205.0" waitingTime="60.0"/>
</tripinfos>
"""


def test_parse_tripinfo(tmp_path):
    p = tmp_path / "tripinfo.xml"
    p.write_text(TRIPINFO_XML, encoding="utf-8")
    m = parse_tripinfo(p, "teste", "cenario", episode=0, traffic_seed=1, fila_maxima=12.0)
    assert m.n_veiculos == 4
    assert m.throughput == 3, "veículo não concluído (arrival=-1) não conta no throughput"
    assert m.espera_max_s == 180.0, "starvation check inclui veículos não concluídos"
    assert abs(m.espera_media_s - (10 + 30 + 180 + 60) / 4) < 1e-9
    assert abs(m.espera_media_avenida_s - 20.0) < 1e-9
    assert abs(m.espera_media_local_s - 120.0) < 1e-9
    # tempo de viagem médio: só concluídos
    assert abs(m.tempo_viagem_medio_s - (95 + 120 + 205) / 3) < 1e-9
    assert m.fila_maxima == 12.0


def test_bootstrap_ci_contains_mean():
    vals = np.array([10.0, 12.0, 9.0, 11.0, 10.5, 13.0, 8.5, 10.2])
    mean, lo, hi = bootstrap_ci(vals, n_boot=2000, seed=1)
    assert lo <= mean <= hi
    assert abs(mean - vals.mean()) < 1e-9


def test_bootstrap_ci_degenerate():
    mean, lo, hi = bootstrap_ci(np.array([5.0]))
    assert mean == lo == hi == 5.0
    assert np.isnan(bootstrap_ci(np.array([]))[0])


def _fake_df():
    rng = np.random.default_rng(0)
    rows = []
    for sc in ("A", "B"):
        for method, base in (("dqn", 10.0), ("fixo_igual", 50.0), ("max_pressure", 11.0)):
            for _ep in range(20):
                rows.append(
                    {"scenario": sc, "method": method,
                     "espera_media_s": base + rng.normal(0, 1)}
                )
    return pd.DataFrame(rows)


def test_dqn_vs_baselines_detects_difference():
    tests = dqn_vs_baselines(_fake_df())
    row = tests[(tests["scenario"] == "A") & (tests["baseline"] == "fixo_igual")].iloc[0]
    assert row["significativo"] and row["dqn_melhor"]
    assert row["p_bonferroni"] >= row["p_value"], "Bonferroni nunca diminui o p"


def test_dqn_vs_baselines_empty_safe():
    df = pd.DataFrame({"scenario": ["A"], "method": ["fixo_igual"], "espera_media_s": [1.0]})
    tests = dqn_vs_baselines(df)
    assert tests.empty and "scenario" in tests.columns


def test_pairwise_tests_all_pairs():
    tests = pairwise_tests(_fake_df())
    per_scenario = tests.groupby("scenario").size()
    assert (per_scenario == 3).all()  # C(3,2) pares
