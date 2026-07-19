"""Visualização do grid sem GUI: mapa da malha e animação de congestionamento.

Feita para quem acompanha de longe (iPad, relatório, apresentação):
- `render_grid_map`: mapa estático da malha declarada no grid.yaml — largura
  do traço ∝ nº de faixas, avenidas destacadas, nomes das ruas;
- `render_congestion_gif`: roda UM episódio com o controlador escolhido e
  pinta cada quadra pela fila (verde = livre, vermelho = parado), gerando um
  GIF do episódio inteiro.

Excluído da meta de cobertura (código de plot).
"""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from PIL import Image

from traffic_rl.envs.grid_env import GridTrafficEnv
from traffic_rl.envs.grid_network import GridTopology
from traffic_rl.grid_config import GridProjectConfig

Segment = tuple[tuple[float, float], tuple[float, float]]


def _edge_segments(topo: GridTopology) -> dict[str, Segment]:
    """Coordenadas (início, fim) de cada aresta, com leve offset perpendicular
    para separar os dois sentidos da mesma rua no desenho."""
    spec = topo.spec
    b = spec.boundary_length_m
    segs: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {}
    off = spec.block_length_m * 0.03
    for i in range(topo.R):
        _, y = topo.node_xy(i, 0)
        xs = [-b] + [topo.node_xy(i, j)[0] for j in range(topo.C)] + [
            topo.node_xy(i, topo.C - 1)[0] + b
        ]
        for k in range(len(xs) - 1):
            segs[f"h_{i}_{k}_E"] = ((xs[k], y - off), (xs[k + 1], y - off))
            segs[f"h_{i}_{k}_W"] = ((xs[k + 1], y + off), (xs[k], y + off))
    for j in range(topo.C):
        x, _ = topo.node_xy(0, j)
        ys = [b] + [topo.node_xy(i, j)[1] for i in range(topo.R)] + [
            topo.node_xy(topo.R - 1, j)[1] - b
        ]
        for k in range(len(ys) - 1):
            segs[f"v_{j}_{k}_S"] = ((x + off, ys[k]), (x + off, ys[k + 1]))
            segs[f"v_{j}_{k}_N"] = ((x - off, ys[k + 1]), (x - off, ys[k]))
    return segs


def _draw_base(ax, topo: GridTopology) -> None:
    spec = topo.spec
    b = spec.boundary_length_m
    for i, row in enumerate(spec.rows):
        _, y = topo.node_xy(i, 0)
        x_end = topo.node_xy(i, topo.C - 1)[0] + b
        ax.annotate(
            row.name, (-b, y + 0.06 * spec.block_length_m),
            fontsize=9, fontweight="bold" if row.street_class == "avenida" else "normal",
        )
        ax.plot([-b, x_end], [y, y], color="#dddddd", linewidth=1, zorder=0)
    for j, col in enumerate(spec.cols):
        x, _ = topo.node_xy(0, j)
        y_end = topo.node_xy(topo.R - 1, j)[1] - b
        ax.annotate(
            col.name, (x + 0.05 * spec.block_length_m, b),
            fontsize=9, rotation=90, va="top",
            fontweight="bold" if col.street_class == "avenida" else "normal",
        )
        ax.plot([x, x], [b, y_end], color="#dddddd", linewidth=1, zorder=0)
    for i in range(topo.R):
        for j in range(topo.C):
            x, y = topo.node_xy(i, j)
            ax.plot(x, y, "s", color="#333333", markersize=5, zorder=5)
    ax.set_aspect("equal")
    ax.axis("off")


def render_grid_map(cfg: GridProjectConfig, out_path: Path) -> Path:
    """Mapa estático da malha declarada (sem simulação)."""
    topo = GridTopology(cfg.grid)
    segs = _edge_segments(topo)
    fig, ax = plt.subplots(figsize=(8, 8))
    _draw_base(ax, topo)
    lines, widths, colors = [], [], []
    for edge_id, (p0, p1) in segs.items():
        street = topo.street_of_edge(edge_id)
        lines.append([p0, p1])
        widths.append(1.5 + 1.3 * street.lanes)
        colors.append("#4C72B0" if street.street_class == "avenida" else "#999999")
    ax.add_collection(LineCollection(lines, linewidths=widths, colors=colors, zorder=1))
    ax.autoscale()
    ax.set_title(
        f"Grid {topo.R}×{topo.C} — azul: avenidas (3 faixas), cinza: locais (1 faixa)\n"
        "quadrados: cruzamentos semaforizados"
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def render_congestion_gif(
    cfg: GridProjectConfig,
    method: str,
    out_path: Path,
    model_path: Path | None = None,
    traffic_seed: int = 10_000,
    sample_every_s: float = 20.0,
    fps: int = 8,
) -> Path:
    """Roda um episódio com `method` e gera um GIF do congestionamento."""
    from traffic_rl.controllers.grid import make_grid_controllers

    topo = GridTopology(cfg.grid)
    segs = _edge_segments(topo)
    edge_ids = list(segs.keys())
    def _seg_len(e: str) -> float:
        (x0, y0), (x1, y1) = segs[e]
        return float(np.hypot(x1 - x0, y1 - y0))

    caps = {e: topo.street_of_edge(e).lanes * (_seg_len(e) / 7.5) for e in edge_ids}

    env = GridTrafficEnv(cfg)
    frames_data: list[tuple[float, dict[str, int]]] = []
    try:
        controllers = make_grid_controllers(method, cfg, env, model_path)
        observations, info = env.reset(traffic_seed)
        next_sample = 0.0
        done = False
        while not done:
            actions = [c.act(o) for c, o in zip(controllers, observations, strict=True)]
            observations, _r, done, info = env.step(actions)
            if info["sim_time"] >= next_sample:
                counts = {
                    e: env._conn.edge.getLastStepHaltingNumber(e) for e in edge_ids
                }
                frames_data.append((info["sim_time"], counts))
                next_sample += sample_every_s
    finally:
        env.close()

    cmap = plt.get_cmap("RdYlGn_r")
    images: list[Image.Image] = []
    for sim_time, counts in frames_data:
        fig, ax = plt.subplots(figsize=(7, 7))
        _draw_base(ax, topo)
        lines, widths, colors = [], [], []
        for e in edge_ids:
            street = topo.street_of_edge(e)
            level = min(counts[e] / max(caps[e], 1.0), 1.0)
            lines.append([segs[e][0], segs[e][1]])
            widths.append(1.5 + 1.3 * street.lanes)
            colors.append(cmap(level))
        ax.add_collection(LineCollection(lines, linewidths=widths, colors=colors, zorder=1))
        ax.autoscale()
        total = sum(counts.values())
        ax.set_title(
            f"Congestionamento — método: {method} | t = {sim_time:5.0f} s | "
            f"veículos parados: {total}"
        )
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=90)
        plt.close(fig)
        buf.seek(0)
        images.append(Image.open(buf).convert("P", palette=Image.Palette.ADAPTIVE))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        out_path,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / fps),
        loop=0,
    )
    return out_path
