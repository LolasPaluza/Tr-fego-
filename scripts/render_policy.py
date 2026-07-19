#!/usr/bin/env python3
"""Raio-X da política: arquitetura da rede + mapas de decisão aprendidos.

Gera uma figura com (1) o diagrama do MLP e (2) mapas "fila da avenida ×
fila da via local → ação escolhida" para o cruzamento com avenida verde e
com via local verde. Útil para comparar o agente em estágios diferentes do
treino (demo vs completo) — as fronteiras de decisão ficam mais limpas com
mais treino.

Uso:
  python scripts/render_policy.py                       # best model do tag default
  python scripts/render_policy.py --model results/runs/<tag>/seed_42/best_model.zip
  python scripts/render_policy.py --out results/figures/politica_final.png

Restrição: o mapa varre observações obs_minimal (9 dims); modelos treinados
com obs_rica são rejeitados com mensagem clara.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, Patch

ACTION_NAMES = ["av. frente", "av. esq.", "local frente", "local esq."]
IN_LABELS = [
    "estágio 0 (av. frente)", "estágio 1 (av. esq.)", "estágio 2 (loc. frente)",
    "estágio 3 (loc. esq.)", "tempo no verde", "fila Leste", "fila Oeste",
    "fila Norte", "fila Sul",
]
OUT_LABELS = [
    "verde: av. frente/dir", "verde: av. esquerda",
    "verde: local frente/dir", "verde: local esquerda",
]


def _draw_architecture(ax, n_params: int, net_arch: list[int]) -> None:
    ax.axis("off")
    ax.set_title(f"Arquitetura — {n_params:,} parâmetros".replace(",", "."), fontsize=11)
    xs = np.linspace(0.06, 0.90, 2 + len(net_arch))
    ys_in = np.linspace(0.92, 0.08, len(IN_LABELS))
    ys_h = np.linspace(0.85, 0.15, 8)
    ys_out = np.linspace(0.78, 0.22, len(OUT_LABELS))
    layers = [ys_in] + [ys_h] * len(net_arch) + [ys_out]
    for y, lab in zip(ys_in, IN_LABELS, strict=True):
        ax.add_patch(Circle((xs[0], y), 0.014, color="#4C72B0"))
        ax.text(xs[0] - 0.03, y, lab, ha="right", va="center", fontsize=7.5)
    for li, width in enumerate(net_arch):
        for y in ys_h:
            ax.add_patch(Circle((xs[1 + li], y), 0.014, color="#999999"))
        ax.text(xs[1 + li], 0.02, f"{width} neurônios\n(ReLU)", ha="center", fontsize=8)
    for y, lab in zip(ys_out, OUT_LABELS, strict=True):
        ax.add_patch(Circle((xs[-1], y), 0.014, color="#DD8452"))
        ax.text(xs[-1] + 0.03, y, lab, ha="left", va="center", fontsize=7.5)
    for x0, x1, y0s, y1s in zip(xs[:-1], xs[1:], layers[:-1], layers[1:], strict=True):
        for y0 in y0s:
            for y1 in y1s:
                ax.plot([x0 + 0.015, x1 - 0.015], [y0, y1], color="#cccccc",
                        linewidth=0.3, zorder=0)
    ax.set_xlim(-0.28, 1.22)
    ax.set_ylim(0, 1.02)


def _policy_map(ax, model, stage: int, title: str, time_in_green_s: float = 15.0,
                max_green_s: float = 120.0, n: int = 60, q_max: float = 0.6) -> None:
    cmap = ListedColormap(["#4C72B0", "#8ab0e0", "#DD8452", "#f0b48a"])
    grid = np.zeros((n, n))
    q_axis = np.linspace(0, q_max, n)
    for i, ql in enumerate(q_axis):
        batch = []
        for qa in q_axis:
            one_hot = np.zeros(4)
            one_hot[stage] = 1.0
            batch.append(
                np.concatenate([one_hot, [time_in_green_s / max_green_s], [qa, qa, ql, ql]])
            )
        acts, _ = model.predict(np.array(batch, dtype=np.float32), deterministic=True)
        grid[i, :] = acts
    pct = int(q_max * 100)
    ax.imshow(grid, origin="lower", extent=[0, pct, 0, pct], cmap=cmap, vmin=0, vmax=3,
              aspect="auto", interpolation="nearest")
    ax.set_xlabel("fila da avenida (% da capacidade)")
    ax.set_ylabel("fila da via local (% da capacidade)")
    ax.set_title(f"{title}\n({time_in_green_s:.0f} s de verde decorridos)", fontsize=10)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--title", default=None, help="sufixo do título da figura")
    args = parser.parse_args()

    from stable_baselines3 import DQN

    from traffic_rl.paths import results_dir, runs_dir

    model_path = args.model
    if model_path is None:
        default_tag = "dqn_pico_assimetrico_obs_minimal_r_espera"
        candidates = sorted(runs_dir().glob(f"{default_tag}/seed_*/best_model.zip"))
        if not candidates:
            print(f"erro: nenhum best_model.zip em runs/{default_tag}; use --model",
                  file=sys.stderr)
            return 1
        model_path = candidates[0]
    model = DQN.load(str(model_path), device="cpu")
    if model.observation_space.shape != (9,):
        print(f"erro: o mapa de política requer obs_minimal (9 dims); o modelo "
              f"espera {model.observation_space.shape}", file=sys.stderr)
        return 1

    q_net = model.policy.q_net
    n_params = sum(p.numel() for p in q_net.parameters())
    net_arch = [m.out_features for m in q_net.q_net if hasattr(m, "out_features")][:-1]

    fig = plt.figure(figsize=(15, 6.5))
    _draw_architecture(fig.add_subplot(1, 3, 1), n_params, net_arch)
    _policy_map(fig.add_subplot(1, 3, 2), model, stage=0,
                title="Decisão aprendida quando a avenida está verde")
    _policy_map(fig.add_subplot(1, 3, 3), model, stage=2,
                title="Decisão aprendida quando a via local está verde")
    cmap = ListedColormap(["#4C72B0", "#8ab0e0", "#DD8452", "#f0b48a"])
    fig.legend(
        handles=[Patch(color=cmap(i), label=f"dá verde: {ACTION_NAMES[i]}") for i in range(4)],
        loc="lower center", ncol=4, fontsize=9, frameon=False,
    )
    suffix = args.title or model_path.parent.name
    fig.suptitle(f"A rede neural do TrafficRL — {suffix}", fontsize=13)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out = args.out or (results_dir() / "figures" / "rede_neural.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print("figura:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
