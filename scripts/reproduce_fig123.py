from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ebcm_repro.models import empirical_degree_pgf, solve_cm, solve_mass_action
from ebcm_repro.simulation import (
    SimulationResult,
    build_configuration_model_edges,
    run_sir_tauleap,
    sample_degree_sequence,
)


@dataclass(frozen=True)
class PlotTheme:
    name: str
    text: str
    canvas: str
    panel: str
    flow_box_face: str
    flow_box_edge: str
    flow_arrow: str
    grid: str
    palette: dict[str, str]
    ma_color: str


def _get_theme(style: str) -> PlotTheme:
    s = style.lower()
    if s == "mono":
        return PlotTheme(
            name="mono",
            text="#111111",
            canvas="#ffffff",
            panel="#ffffff",
            flow_box_face="#f6f6f6",
            flow_box_edge="#1a1a1a",
            flow_arrow="#1a1a1a",
            grid="#c4c4c4",
            palette={
                "Homogeneous": "#1a1a1a",
                "Poisson": "#4a4a4a",
                "Bimodal": "#6f6f6f",
                "Truncated Powerlaw": "#919191",
            },
            ma_color="#b0b0b0",
        )
    if s == "classic":
        return PlotTheme(
            name="classic",
            text="#0f172a",
            canvas="#ffffff",
            panel="#ffffff",
            flow_box_face="#f8fafc",
            flow_box_edge="#1f2937",
            flow_arrow="#111827",
            grid="#94a3b8",
            palette={
                "Homogeneous": "#1f77b4",
                "Poisson": "#2ca02c",
                "Bimodal": "#ff7f0e",
                "Truncated Powerlaw": "#d62728",
            },
            ma_color="#9467bd",
        )
    return PlotTheme(
        name="bold",
        text="#102a43",
        canvas="#f4f8fb",
        panel="#fefefe",
        flow_box_face="#e8f2ff",
        flow_box_edge="#0b5394",
        flow_arrow="#0b5394",
        grid="#b8d0e6",
        palette={
            "Homogeneous": "#0b84a5",
            "Poisson": "#f6c85f",
            "Bimodal": "#6f4e7c",
            "Truncated Powerlaw": "#d45087",
        },
        ma_color="#7f7f7f",
    )


def _draw_box(
    ax: plt.Axes, x: float, y: float, text: str, theme: PlotTheme, w: float = 0.22, h: float = 0.16
) -> None:
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.2,
        edgecolor=theme.flow_box_edge,
        facecolor=theme.flow_box_face,
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=11, color=theme.text)


def _draw_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    label: str,
    lx: float,
    ly: float,
    theme: PlotTheme,
    style: str = "-",
) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", lw=1.4, color=theme.flow_arrow, linestyle=style),
    )
    ax.text(lx, ly, label, fontsize=10, ha="center", va="center", color=theme.text)


def make_fig1(output_dir: Path, theme: PlotTheme) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 2.8))
    fig.patch.set_facecolor(theme.canvas)
    ax.set_facecolor(theme.panel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _draw_box(ax, 0.18, 0.55, r"$S$", theme)
    _draw_box(ax, 0.50, 0.55, r"$I$", theme)
    _draw_box(ax, 0.82, 0.55, r"$R$", theme)
    _draw_arrow(ax, (0.29, 0.55), (0.39, 0.55), r"$\beta SI$", 0.34, 0.64, theme)
    _draw_arrow(ax, (0.61, 0.55), (0.71, 0.55), r"$\gamma I$", 0.66, 0.64, theme)

    ax.text(0.18, 0.25, r"$\dot{S}=-\beta SI$", ha="center", fontsize=10, color=theme.text)
    ax.text(0.50, 0.25, r"$\dot{I}=\beta SI-\gamma I$", ha="center", fontsize=10, color=theme.text)
    ax.text(0.82, 0.25, r"$\dot{R}=\gamma I$", ha="center", fontsize=10, color=theme.text)
    ax.set_title(f"Figure 1 (Style: {theme.name})", fontsize=12, pad=8, color=theme.text)

    fig.tight_layout()
    fig.savefig(output_dir / "fig1_mass_action_flow.png", dpi=220)
    fig.savefig(output_dir / "fig1_mass_action_flow.pdf")
    plt.close(fig)


def make_fig2(output_dir: Path, theme: PlotTheme) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2))
    fig.patch.set_facecolor(theme.canvas)
    for ax in axes:
        ax.set_facecolor(theme.panel)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    left = axes[0]
    _draw_box(left, 0.18, 0.70, r"$\phi_S$", theme)
    _draw_box(left, 0.50, 0.70, r"$\phi_I$", theme)
    _draw_box(left, 0.82, 0.70, r"$\phi_R$", theme)
    _draw_box(left, 0.50, 0.26, r"$1-\theta$", theme, w=0.24)
    _draw_arrow(left, (0.61, 0.70), (0.71, 0.70), r"$\gamma\phi_I$", 0.66, 0.79, theme)
    _draw_arrow(left, (0.50, 0.62), (0.50, 0.36), r"$\beta\phi_I$", 0.61, 0.50, theme)
    _draw_arrow(
        left,
        (0.29, 0.70),
        (0.39, 0.70),
        r"$\beta\phi_I\frac{\psi''(\theta)}{\psi'(1)}$",
        0.34,
        0.79,
        theme,
        style="--",
    )
    left.text(0.18, 0.53, r"$\phi_S=\frac{\psi'(\theta)}{\psi'(1)}$", ha="center", fontsize=10, color=theme.text)
    left.set_title("Neighbor-State Flow", fontsize=11, color=theme.text)

    right = axes[1]
    _draw_box(right, 0.18, 0.70, r"$S$", theme)
    _draw_box(right, 0.50, 0.70, r"$I$", theme)
    _draw_box(right, 0.82, 0.70, r"$R$", theme)
    _draw_arrow(right, (0.29, 0.70), (0.39, 0.70), r"$-\dot{S}$", 0.34, 0.79, theme, style="--")
    _draw_arrow(right, (0.61, 0.70), (0.71, 0.70), r"$\gamma I$", 0.66, 0.79, theme)
    right.text(0.18, 0.53, r"$S=\psi(\theta)$", ha="center", fontsize=10, color=theme.text)
    right.text(0.50, 0.53, r"$I=1-S-R$", ha="center", fontsize=10, color=theme.text)
    right.set_title("Population-State Flow", fontsize=11, color=theme.text)

    fig.suptitle(f"Figure 2 (Style: {theme.name})", fontsize=13, y=0.98, color=theme.text)
    fig.tight_layout()
    fig.savefig(output_dir / "fig2_cm_flow.png", dpi=220)
    fig.savefig(output_dir / "fig2_cm_flow.pdf")
    plt.close(fig)


def _first_crossing_time(t: np.ndarray, y: np.ndarray, threshold: float) -> float:
    idx = np.flatnonzero(y >= threshold)
    if idx.size == 0:
        return float(t[0])
    j = int(idx[0])
    if j == 0:
        return float(t[0])
    t0, t1 = t[j - 1], t[j]
    y0, y1 = y[j - 1], y[j]
    if y1 == y0:
        return float(t1)
    frac = (threshold - y0) / (y1 - y0)
    return float(t0 + frac * (t1 - t0))


def _theta_from_i0(pgf, i0: float) -> float:
    target_s = 1.0 - i0
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        val = pgf.psi(mid)
        if val > target_s:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def _simulate_until_major_outbreak(
    name: str,
    n_nodes: int,
    beta: float,
    gamma: float,
    t_max: float,
    dt: float,
    rng: np.random.Generator,
    attempts: int = 8,
) -> tuple[np.ndarray, SimulationResult, dict[str, int]]:
    for _ in range(attempts):
        degrees = sample_degree_sequence(name, n_nodes=n_nodes, rng=rng)
        edges_u, edges_v = build_configuration_model_edges(degrees, rng=rng)
        initial_infected = max(50, int(0.0002 * n_nodes))
        sim = run_sir_tauleap(
            edges_u=edges_u,
            edges_v=edges_v,
            n_nodes=n_nodes,
            beta=beta,
            gamma=gamma,
            t_max=t_max,
            dt=dt,
            initial_infected=initial_infected,
            rng=rng,
        )
        if sim.cumulative[-1] > 0.04:
            return degrees, sim, {"initial_infected": initial_infected}
    raise RuntimeError(f"failed to generate a major outbreak after {attempts} attempts for {name}")


def make_fig3(
    output_dir: Path,
    n_nodes: int,
    beta: float,
    gamma: float,
    t_max: float,
    dt: float,
    ode_steps: int,
    seed: int,
    theme: PlotTheme,
) -> None:
    rng = np.random.default_rng(seed)
    scenarios = [
        ("homogeneous", "Homogeneous"),
        ("poisson", "Poisson"),
        ("bimodal", "Bimodal"),
        ("truncated_powerlaw", "Truncated Powerlaw"),
    ]
    colors = theme.palette

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 7.4), sharex=True)
    fig.patch.set_facecolor(theme.canvas)
    ax_cum, ax_inf = axes
    ax_cum.set_facecolor(theme.panel)
    ax_inf.set_facecolor(theme.panel)

    for name, label in scenarios:
        degrees, sim, meta = _simulate_until_major_outbreak(
            name=name,
            n_nodes=n_nodes,
            beta=beta,
            gamma=gamma,
            t_max=t_max,
            dt=dt,
            rng=rng,
        )
        pgf = empirical_degree_pgf(degrees)
        i0 = meta["initial_infected"] / n_nodes
        theta0 = _theta_from_i0(pgf, i0)

        cm = solve_cm(
            pgf=pgf,
            beta=beta,
            gamma=gamma,
            t_max=t_max,
            n_steps=ode_steps,
            theta0=theta0,
            r0=0.0,
        )
        ma = solve_mass_action(
            beta_hat=beta * pgf.mean_degree,
            gamma=gamma,
            t_max=t_max,
            i0=i0,
            r0=0.0,
            n_steps=ode_steps,
        )

        shift = _first_crossing_time(sim.t, sim.cumulative, threshold=0.01)
        sim_t = sim.t - shift
        cm_t = cm.t - _first_crossing_time(cm.t, cm.cumulative, threshold=0.01)
        ma_t = ma.t - _first_crossing_time(ma.t, ma.cumulative, threshold=0.01)

        c = colors[label]
        ax_cum.plot(sim_t, sim.cumulative, color=c, lw=2.3, label=label, marker="o", markevery=40, ms=3.4, alpha=0.95)
        ax_cum.plot(cm_t, cm.cumulative, color=c, lw=2.1, ls="-.")
        ax_cum.plot(ma_t, ma.cumulative, color=theme.ma_color, lw=1.2, ls=":", alpha=0.9)

        ax_inf.plot(sim_t, sim.I, color=c, lw=2.3, label=label, marker="o", markevery=40, ms=3.4, alpha=0.95)
        ax_inf.plot(cm_t, cm.I, color=c, lw=2.1, ls="-.")
        ax_inf.plot(ma_t, ma.I, color=theme.ma_color, lw=1.2, ls=":", alpha=0.9)

    ax_cum.set_ylabel("Cumulative infections", color=theme.text)
    ax_inf.set_ylabel("Infections", color=theme.text)
    ax_inf.set_xlabel("t", color=theme.text)

    ax_cum.set_xlim(-5.0, 15.0)
    ax_cum.set_ylim(0.0, 1.0)
    ax_inf.set_ylim(0.0, 0.37)

    for ax in axes:
        ax.grid(alpha=0.3, linewidth=0.8, color=theme.grid)
        ax.legend(loc="upper left", frameon=True, fontsize=9)
        ax.tick_params(colors=theme.text)
        for spine in ax.spines.values():
            spine.set_color(theme.grid)

    ax_cum.set_title(
        f"Figure 3 Reproduction ({theme.name}): markers=simulation, dash-dot=CM, dotted=MA",
        color=theme.text,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "fig3_cm_example.png", dpi=240)
    fig.savefig(output_dir / "fig3_cm_example.pdf")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce Fig1-3 in the EBCM paper.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "figures")
    parser.add_argument("--n-nodes", type=int, default=120000)
    parser.add_argument("--beta", type=float, default=0.6)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--t-max", type=float, default=24.0)
    parser.add_argument("--dt", type=float, default=0.04)
    parser.add_argument("--ode-steps", type=int, default=1401)
    parser.add_argument("--seed", type=int, default=20260220)
    parser.add_argument("--style", choices=["bold", "classic", "mono"], default="bold")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    theme = _get_theme(args.style)

    print(f"[1/3] generating fig1 in {args.output_dir}")
    make_fig1(args.output_dir, theme=theme)
    print(f"[2/3] generating fig2 in {args.output_dir}")
    make_fig2(args.output_dir, theme=theme)
    print(
        "[3/3] generating fig3 with "
        f"n_nodes={args.n_nodes}, beta={args.beta}, gamma={args.gamma}, dt={args.dt}, seed={args.seed}, style={args.style}"
    )
    make_fig3(
        output_dir=args.output_dir,
        n_nodes=args.n_nodes,
        beta=args.beta,
        gamma=args.gamma,
        t_max=args.t_max,
        dt=args.dt,
        ode_steps=args.ode_steps,
        seed=args.seed,
        theme=theme,
    )
    print("done")


if __name__ == "__main__":
    main()
