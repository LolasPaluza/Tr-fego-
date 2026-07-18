import numpy as np
import pandas as pd

from traffic_rl.analysis.report import generate_report


def _fake_metrics():
    rng = np.random.default_rng(1)
    rows = []
    for sc in ("pico_assimetrico", "fora_pico"):
        for method, base in (
            ("fixo_igual", 60.0), ("fixo_proporcional", 40.0),
            ("atuado_gap", 35.0), ("max_pressure", 30.0), ("dqn", 25.0),
        ):
            for ep in range(8):
                w = base + rng.normal(0, 3)
                rows.append(
                    {
                        "scenario": sc, "method": method, "episode": ep,
                        "traffic_seed": 10000 + ep,
                        "espera_media_s": w, "espera_p95_s": w * 3, "espera_max_s": w * 6,
                        "fila_maxima": 40.0, "throughput": 900, "tempo_viagem_medio_s": 120.0,
                        "espera_media_avenida_s": w * 0.8, "espera_media_local_s": w * 1.6,
                        "n_veiculos": 1000,
                    }
                )
    return pd.DataFrame(rows)


def test_generate_report(tmp_path):
    curves = {
        42: pd.DataFrame({"timesteps": [1000, 2000], "recompensa_media": [-50.0, -20.0]}),
        123: pd.DataFrame({"timesteps": [1000, 2000], "recompensa_media": [-55.0, -22.0]}),
    }
    report = generate_report(_fake_metrics(), tmp_path, eval_curves=curves, git_hash="abc123")
    text = report.read_text(encoding="utf-8")
    assert "pico_assimetrico" in text
    assert "Bonferroni" in text
    assert "abc123" in text
    assert "DQN (nosso)" in text
    assert "Interpretação" in text
    figs = list((tmp_path / "figures").glob("*.png"))
    assert len(figs) >= 4  # barras, boxplot, 2× justiça, curvas
