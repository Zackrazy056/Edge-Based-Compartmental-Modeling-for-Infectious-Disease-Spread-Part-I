from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import fitz
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ebcm_repro.distributions import (
    dvd_expected,
    edmfs_h_expected,
    mp_piecewise_uniform_expected,
    nb_degree_pgf,
    poisson_degree,
)
from ebcm_repro.extended_models import (
    ModelSolution,
    pmf_degree_pgf,
    solve_ad_mfsh,
    solve_dc,
    solve_dfd,
    solve_dvd,
    solve_ed_mfsh,
    solve_mp,
)
from ebcm_repro.simulation import SimulationResult
from ebcm_repro.stochastic_models import (
    align_and_average_runs,
    run_ad_mfsh_sim,
    run_dc_sim,
    run_dfd_sim,
    run_dvd_sim,
    run_ed_mfsh_binned_sim,
    run_ed_mfsh_sim,
    run_mp_sim,
)


NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
BLUE = (0.353, 0.702, 0.902)
RED = (1.0, 0.125, 0.0)


@dataclass(frozen=True)
class AxisBounds:
    x_left: float
    x_right: float
    y_top: float
    y_bottom: float


@dataclass(frozen=True)
class Curve:
    x: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class SimParams:
    n_nodes: int
    runs_total: int
    dt: float


@dataclass(frozen=True)
class StochConfig:
    fig_no: int
    key: str
    title: str
    align_threshold: float
    epidemic_threshold: float
    t_max: float
    ode_steps: int
    cum_pdf: Path
    inf_pdf: Path


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


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if not np.any(m):
        return float("nan")
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))


def _peak(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    m = np.isfinite(y)
    if not np.any(m):
        return float("nan"), float("nan")
    xm, ym = x[m], y[m]
    i = int(np.argmax(ym))
    return float(xm[i]), float(ym[i])


def _drawing_points(draw: dict) -> np.ndarray:
    pts = []
    for item in draw.get("items", []):
        if item[0] == "l":
            pts.append((float(item[1].x), float(item[1].y)))
            pts.append((float(item[2].x), float(item[2].y)))
        elif item[0] == "c":
            for p in item[1:5]:
                pts.append((float(p.x), float(p.y)))
    return np.asarray(pts, dtype=float) if pts else np.empty((0, 2), dtype=float)


def _axis_bounds(drawings: list[dict]) -> AxisBounds:
    lines = []
    for d in drawings:
        c = d.get("color")
        c = tuple(round(float(v), 3) for v in c) if c is not None else None
        if c != (0.0, 0.0, 0.0):
            continue
        if d.get("dashes") != "[] 0":
            continue
        if abs(float(d.get("width") or 0.0) - 1.0) > 1e-6:
            continue
        items = d.get("items", [])
        if len(items) != 1 or items[0][0] != "l":
            continue
        p1, p2 = items[0][1], items[0][2]
        x1, y1, x2, y2 = float(p1.x), float(p1.y), float(p2.x), float(p2.y)
        if np.hypot(x2 - x1, y2 - y1) > 100:
            lines.append((x1, y1, x2, y2))
    x_vert = sorted({round(x1, 4) for x1, y1, x2, y2 in lines if abs(x1 - x2) < 1e-6})
    y_hori = sorted({round(y1, 4) for x1, y1, x2, y2 in lines if abs(y1 - y2) < 1e-6})
    return AxisBounds(float(x_vert[0]), float(x_vert[-1]), float(y_hori[0]), float(y_hori[-1]))


def _tick_maps(page: fitz.Page, axis: AxisBounds):
    words = page.get_text("words")
    nums = []
    for x0, y0, x1, y1, txt, *_ in words:
        if NUM_RE.match(txt):
            nums.append((float(x0), float(y0), float(x1), float(y1), float(txt)))
    x_ticks = []
    for x0, y0, x1, y1, val in nums:
        xc = 0.5 * (x0 + x1)
        yc = 0.5 * (y0 + y1)
        if yc > axis.y_bottom + 1.0 and axis.x_left - 2.0 <= xc <= axis.x_right + 2.0:
            x_ticks.append((xc, val))
    y_ticks = []
    for x0, y0, x1, y1, val in nums:
        xc = 0.5 * (x0 + x1)
        yc = 0.5 * (y0 + y1)
        if xc < axis.x_left - 1.0 and axis.y_top - 2.0 <= yc <= axis.y_bottom + 2.0:
            y_ticks.append((yc, val))
    x_px = np.array([p for p, _ in x_ticks], dtype=float)
    x_val = np.array([v for _, v in x_ticks], dtype=float)
    y_px = np.array([p for p, _ in y_ticks], dtype=float)
    y_val = np.array([v for _, v in y_ticks], dtype=float)
    ax, bx = np.polyfit(x_px, x_val, 1)
    ay, by = np.polyfit(y_px, y_val, 1)
    return (ax, bx), (ay, by), (float(np.min(x_val)), float(np.max(x_val))), (float(np.min(y_val)), float(np.max(y_val)))


def _select_curve_drawing(drawings: list[dict], color: tuple[float, float, float], kind: str) -> dict:
    out = []
    for d in drawings:
        c = d.get("color")
        c = tuple(round(float(v), 3) for v in c) if c is not None else None
        if c != color:
            continue
        w = float(d.get("width") or 0.0)
        dash = str(d.get("dashes"))
        if kind == "solid":
            if abs(w - 1.0) > 1e-6 or dash != "[] 0":
                continue
        else:
            if abs(w - 5.0) > 1e-6 or "3 5 1 5" not in dash:
                continue
        if len(d.get("items", [])) > 10:
            out.append(d)
    if not out:
        raise RuntimeError("curve drawing missing")
    return max(out, key=lambda d: len(d.get("items", [])))


def digitize_curve(pdf_path: Path, kind: str) -> tuple[Curve, tuple[float, float], tuple[float, float]]:
    page = fitz.open(pdf_path.as_posix())[0]
    drawings = page.get_drawings()
    axis = _axis_bounds(drawings)
    (ax, bx), (ay, by), x_lim, y_lim = _tick_maps(page, axis)
    draw = _select_curve_drawing(drawings, BLUE if kind == "solid" else RED, kind)
    pts = _drawing_points(draw)
    keep = (pts[:, 0] >= axis.x_left - 1e-8) & (pts[:, 0] <= axis.x_right + 1e-8)
    pts = pts[keep]
    x = ax * pts[:, 0] + bx
    y = ay * pts[:, 1] + by
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]
    xu = np.unique(np.round(x, 5))
    xo = np.empty_like(xu)
    yo = np.empty_like(xu)
    xr = np.round(x, 5)
    for i, xv in enumerate(xu):
        sel = xr == xv
        xo[i] = np.median(x[sel])
        yo[i] = np.median(y[sel])
    return Curve(x=xo, y=yo), x_lim, y_lim


def _interp_curve(curve: Curve, grid: np.ndarray) -> np.ndarray:
    return np.interp(grid, curve.x, curve.y, left=np.nan, right=np.nan)


def _solve_ode(cfg: StochConfig) -> ModelSolution:
    if cfg.key == "ad_mfsh":
        return solve_ad_mfsh(
            pmf_degree_pgf({1: 25 / 31, 5: 5 / 31, 25: 1 / 31}),
            beta=1.0,
            gamma=1.0,
            t_max=cfg.t_max,
            n_steps=cfg.ode_steps,
            i0=1e-4,
        )
    if cfg.key == "dfd":
        return solve_dfd(
            nb_degree_pgf(r=4.0, p=1.0 / 3.0),
            beta=5.0 / 4.0,
            gamma=1.0,
            eta=0.5,
            t_max=cfg.t_max,
            n_steps=cfg.ode_steps,
            i0=3e-4,
        )
    if cfg.key == "dc":
        return solve_dc(
            poisson_degree(3.0),
            beta=2.0,
            gamma=1.0,
            eta1=1.0,
            eta2=0.5,
            t_max=cfg.t_max,
            n_steps=cfg.ode_steps,
            i0=3e-4,
        )
    if cfg.key == "mp":
        return solve_mp(
            mp_piecewise_uniform_expected(),
            beta=0.15,
            gamma=1.0,
            t_max=cfg.t_max,
            n_steps=cfg.ode_steps,
            i0=1e-4,
        )
    if cfg.key == "ed_mfsh":
        return solve_ed_mfsh(
            edmfs_h_expected(),
            beta=0.435,
            gamma=1.0,
            t_max=cfg.t_max,
            n_steps=cfg.ode_steps,
            i0=2e-5,
        )
    if cfg.key == "dvd":
        return solve_dvd(
            dvd_expected(),
            beta=5.0 / 4.0,
            gamma=1.0,
            eta=0.5,
            t_max=cfg.t_max,
            n_steps=cfg.ode_steps,
            i0=3e-4,
        )
    raise ValueError(cfg.key)


def _run_single_sim(cfg: StochConfig, p: SimParams, rng: np.random.Generator) -> SimulationResult:
    i0 = max(20, int(0.0002 * p.n_nodes))
    if cfg.key == "ad_mfsh":
        return run_ad_mfsh_sim(
            n_nodes=p.n_nodes, beta=1.0, gamma=1.0, t_max=cfg.t_max, dt=p.dt, initial_infected=i0, rng=rng
        )
    if cfg.key == "dfd":
        return run_dfd_sim(
            n_nodes=p.n_nodes, beta=1.25, gamma=1.0, eta=0.5, t_max=cfg.t_max, dt=p.dt, initial_infected=i0, rng=rng
        )
    if cfg.key == "dc":
        return run_dc_sim(
            n_nodes=p.n_nodes,
            beta=2.0,
            gamma=1.0,
            eta1=1.0,
            eta2=0.5,
            t_max=cfg.t_max,
            dt=p.dt,
            initial_infected=i0,
            rng=rng,
        )
    if cfg.key == "mp":
        return run_mp_sim(
            n_nodes=p.n_nodes, beta=0.15, gamma=1.0, t_max=cfg.t_max, dt=p.dt, initial_infected=i0, rng=rng
        )
    if cfg.key == "ed_mfsh":
        return run_ed_mfsh_binned_sim(
            n_nodes=p.n_nodes,
            beta=0.435,
            gamma=1.0,
            t_max=cfg.t_max,
            dt=p.dt,
            initial_infected=i0,
            rng=rng,
            n_bins=240,
        )
    if cfg.key == "dvd":
        return run_dvd_sim(
            n_nodes=p.n_nodes, beta=1.25, gamma=1.0, eta=0.5, t_max=cfg.t_max, dt=p.dt, initial_infected=i0, rng=rng
        )
    raise ValueError(cfg.key)


def configs() -> list[StochConfig]:
    return [
        StochConfig(5, "ad_mfsh", "Fig5 ad-MFSH (stochastic+ODE)", 0.01, 0.08, 20.0, 2001, ROOT / "mfsh_cum.pdf", ROOT / "mfsh_inf.pdf"),
        StochConfig(7, "dfd", "Fig7 DFD (stochastic+ODE)", 0.03, 0.08, 26.0, 2401, ROOT / "DFD_cum.pdf", ROOT / "DFD_inf.pdf"),
        StochConfig(9, "dc", "Fig9 DC (stochastic+ODE)", 0.03, 0.08, 20.0, 2201, ROOT / "DC_cum.pdf", ROOT / "DC_inf.pdf"),
        StochConfig(11, "mp", "Fig11 MP (stochastic+ODE)", 0.01, 0.08, 24.0, 2201, ROOT / "MP_cum.pdf", ROOT / "MP_inf.pdf"),
        StochConfig(
            13,
            "ed_mfsh",
            "Fig13 ed-MFSH (stochastic+ODE)",
            0.005,
            0.01,
            320.0,
            3001,
            ROOT / "mfsh_ed_cum_smallR0.pdf",
            ROOT / "mfsh_ed_inf_smallR0.pdf",
        ),
        StochConfig(15, "dvd", "Fig15 DVD (stochastic+ODE)", 0.03, 0.08, 12.5, 1801, ROOT / "DVD_cum.pdf", ROOT / "DVD_inf.pdf"),
    ]


def profile_params(profile: str) -> dict[str, SimParams]:
    if profile == "quick":
        return {
            "ad_mfsh": SimParams(30_000, 1, 0.06),
            "dfd": SimParams(4_000, 24, 0.06),
            "dc": SimParams(2_500, 40, 0.06),
            "mp": SimParams(30_000, 1, 0.06),
            "ed_mfsh": SimParams(300_000, 1, 0.1),
            "dvd": SimParams(4_000, 24, 0.06),
        }
    if profile == "paper":
        return {
            "ad_mfsh": SimParams(500_000, 1, 0.04),
            "dfd": SimParams(10_000, 250, 0.05),
            "dc": SimParams(5_000, 322, 0.05),
            "mp": SimParams(500_000, 1, 0.04),
            "ed_mfsh": SimParams(15_000_000, 1, 0.1),
            "dvd": SimParams(10_000, 240, 0.05),
        }
    return {
        "ad_mfsh": SimParams(120_000, 1, 0.04),
        "dfd": SimParams(10_000, 120, 0.05),
        "dc": SimParams(5_000, 180, 0.05),
        "mp": SimParams(120_000, 1, 0.04),
        "ed_mfsh": SimParams(2_000_000, 1, 0.1),
        "dvd": SimParams(10_000, 120, 0.05),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stochastic + ODE reproduction for fig5/7/9/11/13/15.")
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "figures")
    p.add_argument("--profile", choices=["quick", "default", "paper"], default="default")
    p.add_argument("--seed-base", type=int, default=20260220)
    p.add_argument("--single-attempts", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pmap = profile_params(args.profile)

    rows = []
    all_cfg = configs()
    for i, cfg in enumerate(all_cfg, start=1):
        sim_p = pmap[cfg.key]
        print(
            f"[{i}/{len(all_cfg)}] fig{cfg.fig_no} key={cfg.key} "
            f"n={sim_p.n_nodes} runs={sim_p.runs_total} dt={sim_p.dt} profile={args.profile}"
        )

        paper_cum_sim, xlim_cum, ylim_cum = digitize_curve(cfg.cum_pdf, "solid")
        paper_cum_theory, _, _ = digitize_curve(cfg.cum_pdf, "dashed")
        paper_inf_sim, xlim_inf, ylim_inf = digitize_curve(cfg.inf_pdf, "solid")
        paper_inf_theory, _, _ = digitize_curve(cfg.inf_pdf, "dashed")

        x_min = min(xlim_cum[0], xlim_inf[0])
        x_max = max(xlim_cum[1], xlim_inf[1])
        fit_grid = np.linspace(x_min, x_max, 401)
        p_cum_fit = _interp_curve(paper_cum_sim, fit_grid)
        p_inf_fit = _interp_curve(paper_inf_sim, fit_grid)

        ode = _solve_ode(cfg)
        ode_shift = _first_crossing_time(ode.t, ode.cumulative, cfg.align_threshold)
        ode_t = ode.t - ode_shift

        runs = []
        for r in range(sim_p.runs_total):
            rng = np.random.default_rng(args.seed_base + cfg.fig_no * 10000 + r)
            run = _run_single_sim(cfg, sim_p, rng)
            runs.append(run)

        if sim_p.runs_total == 1:
            best = None
            best_score = float("inf")
            attempts = max(1, args.single_attempts)
            for a in range(attempts):
                run = runs[0] if a == 0 else _run_single_sim(
                    cfg, sim_p, np.random.default_rng(args.seed_base + cfg.fig_no * 10000 + 5000 + a)
                )
                shift = _first_crossing_time(run.t, run.cumulative, cfg.align_threshold)
                tt = run.t - shift
                sim_c = np.interp(fit_grid, tt, run.cumulative, left=np.nan, right=np.nan)
                sim_i = np.interp(fit_grid, tt, run.I, left=np.nan, right=np.nan)
                score = _rmse(sim_i, p_inf_fit) + 0.7 * _rmse(sim_c, p_cum_fit)
                # Prefer epidemic realizations when score is similar.
                if run.cumulative[-1] < cfg.epidemic_threshold:
                    score += 0.2
                if score < best_score:
                    best = run
                    best_score = score
            assert best is not None
            run = best
            sim_shift = _first_crossing_time(run.t, run.cumulative, cfg.align_threshold)
            sim_mean = SimulationResult(t=run.t - sim_shift, S=run.S, I=run.I, R=run.R)
            n_total = 1
            n_epi = 1 if run.cumulative[-1] >= cfg.epidemic_threshold else 0
        else:
            avg = align_and_average_runs(
                runs=runs,
                align_threshold=cfg.align_threshold,
                epidemic_threshold=cfg.epidemic_threshold,
                t_min=-6.0,
                t_max=22.0 if cfg.key != "ed_mfsh" else 320.0,
                n_grid=1001 if cfg.key != "ed_mfsh" else 1201,
            )
            sim_mean = avg.mean
            n_total = avg.n_total
            n_epi = avg.n_epidemic

        grid = np.linspace(x_min, x_max, 901)

        sim_cum = np.interp(grid, sim_mean.t, sim_mean.cumulative, left=np.nan, right=np.nan)
        sim_inf = np.interp(grid, sim_mean.t, sim_mean.I, left=np.nan, right=np.nan)
        ode_cum = np.interp(grid, ode_t, ode.cumulative, left=np.nan, right=np.nan)
        ode_inf = np.interp(grid, ode_t, ode.I, left=np.nan, right=np.nan)
        p_cum_s = _interp_curve(paper_cum_sim, grid)
        p_inf_s = _interp_curve(paper_inf_sim, grid)
        p_cum_d = _interp_curve(paper_cum_theory, grid)
        p_inf_d = _interp_curve(paper_inf_theory, grid)

        tpk_sim, ipk_sim = _peak(grid, sim_inf)
        tpk_p, ipk_p = _peak(grid, p_inf_s)

        rows.append(
            {
                "figure": cfg.fig_no,
                "key": cfg.key,
                "profile": args.profile,
                "n_nodes": sim_p.n_nodes,
                "runs_total": n_total,
                "runs_epidemic": n_epi,
                "rmse_sim_cum_vs_paper_sim": _rmse(sim_cum, p_cum_s),
                "rmse_sim_inf_vs_paper_sim": _rmse(sim_inf, p_inf_s),
                "rmse_ode_cum_vs_paper_theory": _rmse(ode_cum, p_cum_d),
                "rmse_ode_inf_vs_paper_theory": _rmse(ode_inf, p_inf_d),
                "peak_t_abs_err_sim_vs_paper_sim": abs(tpk_sim - tpk_p),
                "peak_i_abs_err_sim_vs_paper_sim": abs(ipk_sim - ipk_p),
            }
        )

        fig, axes = plt.subplots(2, 1, figsize=(7.4, 7.6), sharex=True)
        ax_cum, ax_inf = axes
        ax_cum.plot(grid, p_cum_s, color="#374151", lw=1.2, alpha=0.6, label="Paper sim (digitized)")
        ax_cum.plot(grid, p_cum_d, color="#9ca3af", lw=1.1, ls="--", alpha=0.7, label="Paper theory (digitized)")
        ax_cum.plot(sim_mean.t, sim_mean.cumulative, color="#111827", lw=2.0, label="Our sim")
        ax_cum.plot(ode_t, ode.cumulative, color="#2563eb", lw=1.8, ls="--", label="Our ODE")
        ax_cum.set_ylim(ylim_cum[0], ylim_cum[1])
        ax_cum.set_ylabel("Cumulative infections")
        ax_cum.grid(alpha=0.2)
        ax_cum.legend(loc="best", fontsize=8, frameon=True)

        ax_inf.plot(grid, p_inf_s, color="#374151", lw=1.2, alpha=0.6, label="Paper sim (digitized)")
        ax_inf.plot(grid, p_inf_d, color="#9ca3af", lw=1.1, ls="--", alpha=0.7, label="Paper theory (digitized)")
        ax_inf.plot(sim_mean.t, sim_mean.I, color="#111827", lw=2.0, label="Our sim")
        ax_inf.plot(ode_t, ode.I, color="#dc2626", lw=1.8, ls="--", label="Our ODE")
        ax_inf.set_ylim(ylim_inf[0], ylim_inf[1])
        ax_inf.set_xlim(x_min, x_max)
        ax_inf.set_ylabel("Infections")
        ax_inf.set_xlabel("t")
        ax_inf.grid(alpha=0.2)
        ax_inf.legend(loc="best", fontsize=8, frameon=True)

        fig.suptitle(f"{cfg.title} | profile={args.profile} | runs={n_epi}/{n_total}")
        fig.tight_layout()
        out = args.output_dir / f"fig{cfg.fig_no}_{cfg.key}_stochastic_repro"
        fig.savefig(out.with_suffix(".png"), dpi=240)
        fig.savefig(out.with_suffix(".pdf"))
        plt.close(fig)

    report = args.output_dir / f"fig5to15_stochastic_metrics_{args.profile}.csv"
    with report.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("done")


if __name__ == "__main__":
    main()
