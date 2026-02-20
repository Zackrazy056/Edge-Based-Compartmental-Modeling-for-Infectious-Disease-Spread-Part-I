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


NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
BLUE = (0.353, 0.702, 0.902)
RED = (1.0, 0.125, 0.0)


@dataclass(frozen=True)
class ExampleConfig:
    fig_no: int
    key: str
    title: str
    threshold: float
    cum_pdf: Path
    inf_pdf: Path
    solve_with_i0: callable


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


def _solve_scenario(cfg: ExampleConfig) -> ModelSolution:
    i0_candidates = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
    for i0 in i0_candidates:
        sol = cfg.solve_with_i0(i0)
        if np.nanmax(sol.cumulative) > cfg.threshold + 0.02:
            return sol
    return cfg.solve_with_i0(1e-3)


def _drawing_points(draw: dict) -> np.ndarray:
    pts = []
    for item in draw.get("items", []):
        cmd = item[0]
        if cmd == "l":
            pts.append((float(item[1].x), float(item[1].y)))
            pts.append((float(item[2].x), float(item[2].y)))
        elif cmd == "c":
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


def _tick_maps(page: fitz.Page, axis: AxisBounds) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]:
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

    if len(x_ticks) < 2 or len(y_ticks) < 2:
        raise RuntimeError("failed to infer axis tick labels from PDF text")

    x_px = np.array([p for p, _ in x_ticks], dtype=float)
    x_val = np.array([v for _, v in x_ticks], dtype=float)
    y_px = np.array([p for p, _ in y_ticks], dtype=float)
    y_val = np.array([v for _, v in y_ticks], dtype=float)

    ax, bx = np.polyfit(x_px, x_val, 1)
    ay, by = np.polyfit(y_px, y_val, 1)
    return (ax, bx), (ay, by), (float(np.min(x_val)), float(np.max(x_val))), (float(np.min(y_val)), float(np.max(y_val)))


def _select_curve_drawing(drawings: list[dict], color: tuple[float, float, float], kind: str) -> dict:
    candidates = []
    for d in drawings:
        c = d.get("color")
        c = tuple(round(float(v), 3) for v in c) if c is not None else None
        if c != color:
            continue
        width = float(d.get("width") or 0.0)
        dashes = str(d.get("dashes"))
        if kind == "solid":
            if abs(width - 1.0) > 1e-6 or dashes != "[] 0":
                continue
        else:
            if abs(width - 5.0) > 1e-6 or "3 5 1 5" not in dashes:
                continue
        if len(d.get("items", [])) > 10:
            candidates.append(d)
    if not candidates:
        raise RuntimeError(f"curve drawing not found for {kind}")
    return max(candidates, key=lambda d: len(d.get("items", [])))


def digitize_curve_from_pdf(pdf_path: Path, kind: str) -> tuple[Curve, tuple[float, float], tuple[float, float]]:
    page = fitz.open(pdf_path.as_posix())[0]
    drawings = page.get_drawings()
    axis = _axis_bounds(drawings)
    (ax, bx), (ay, by), x_lim, y_lim = _tick_maps(page, axis)
    color = BLUE if kind == "solid" else RED
    draw = _select_curve_drawing(drawings, color=color, kind=kind)
    pts = _drawing_points(draw)
    in_x = (pts[:, 0] >= axis.x_left - 1e-8) & (pts[:, 0] <= axis.x_right + 1e-8)
    pts = pts[in_x]
    x = ax * pts[:, 0] + bx
    y = ay * pts[:, 1] + by
    order = np.argsort(x)
    x, y = x[order], y[order]
    x_round = np.round(x, 5)
    ux = np.unique(x_round)
    out_x = np.empty_like(ux)
    out_y = np.empty_like(ux)
    for i, xv in enumerate(ux):
        sel = x_round == xv
        out_x[i] = np.median(x[sel])
        out_y[i] = np.median(y[sel])
    return Curve(x=out_x, y=out_y), x_lim, y_lim


def _interp_to(x_ref: np.ndarray, curve: Curve) -> np.ndarray:
    return np.interp(x_ref, curve.x, curve.y, left=np.nan, right=np.nan)


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        return float("nan")
    return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2)))


def _peak(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    m = np.isfinite(y)
    if not np.any(m):
        return float("nan"), float("nan")
    xm = x[m]
    ym = y[m]
    i = int(np.argmax(ym))
    return float(xm[i]), float(ym[i])


def build_configs() -> list[ExampleConfig]:
    return [
        ExampleConfig(
            fig_no=5,
            key="ad_mfsh",
            title="Fig5 ad-MFSH Example",
            threshold=0.01,
            cum_pdf=ROOT / "mfsh_cum.pdf",
            inf_pdf=ROOT / "mfsh_inf.pdf",
            solve_with_i0=lambda i0: solve_ad_mfsh(
                pmf_degree_pgf({1: 25 / 31, 5: 5 / 31, 25: 1 / 31}),
                beta=1.0,
                gamma=1.0,
                t_max=20.0,
                n_steps=2001,
                i0=i0,
            ),
        ),
        ExampleConfig(
            fig_no=7,
            key="dfd",
            title="Fig7 DFD Example",
            threshold=0.03,
            cum_pdf=ROOT / "DFD_cum.pdf",
            inf_pdf=ROOT / "DFD_inf.pdf",
            solve_with_i0=lambda i0: solve_dfd(
                nb_degree_pgf(r=4.0, p=1.0 / 3.0),
                beta=5.0 / 4.0,
                gamma=1.0,
                eta=0.5,
                t_max=26.0,
                n_steps=2401,
                i0=i0,
            ),
        ),
        ExampleConfig(
            fig_no=9,
            key="dc",
            title="Fig9 DC Example",
            threshold=0.03,
            cum_pdf=ROOT / "DC_cum.pdf",
            inf_pdf=ROOT / "DC_inf.pdf",
            solve_with_i0=lambda i0: solve_dc(
                poisson_degree(3.0),
                beta=2.0,
                gamma=1.0,
                eta1=1.0,
                eta2=0.5,
                t_max=20.0,
                n_steps=2201,
                i0=i0,
            ),
        ),
        ExampleConfig(
            fig_no=11,
            key="mp",
            title="Fig11 MP Example",
            threshold=0.01,
            cum_pdf=ROOT / "MP_cum.pdf",
            inf_pdf=ROOT / "MP_inf.pdf",
            solve_with_i0=lambda i0: solve_mp(
                mp_piecewise_uniform_expected(),
                beta=0.15,
                gamma=1.0,
                t_max=24.0,
                n_steps=2201,
                i0=i0,
            ),
        ),
        ExampleConfig(
            fig_no=13,
            key="ed_mfsh",
            title="Fig13 ed-MFSH Example",
            threshold=0.005,
            cum_pdf=ROOT / "mfsh_ed_cum_smallR0.pdf",
            inf_pdf=ROOT / "mfsh_ed_inf_smallR0.pdf",
            solve_with_i0=lambda i0: solve_ed_mfsh(
                edmfs_h_expected(),
                beta=0.435,
                gamma=1.0,
                t_max=320.0,
                n_steps=3001,
                i0=i0,
            ),
        ),
        ExampleConfig(
            fig_no=15,
            key="dvd",
            title="Fig15 DVD Example",
            threshold=0.03,
            cum_pdf=ROOT / "DVD_cum.pdf",
            inf_pdf=ROOT / "DVD_inf.pdf",
            solve_with_i0=lambda i0: solve_dvd(
                dvd_expected(),
                beta=5.0 / 4.0,
                gamma=1.0,
                eta=0.5,
                t_max=12.5,
                n_steps=1801,
                i0=i0,
            ),
        ),
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproduce remaining example figures (5/7/9/11/13/15).")
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "figures")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows = []
    cfgs = build_configs()
    for idx, cfg in enumerate(cfgs, start=1):
        print(f"[{idx}/{len(cfgs)}] {cfg.title}")
        sol = _solve_scenario(cfg)

        t0 = _first_crossing_time(sol.t, sol.cumulative, cfg.threshold)
        t_aligned = sol.t - t0

        cum_solid, xlim_cum, ylim_cum = digitize_curve_from_pdf(cfg.cum_pdf, kind="solid")
        cum_dash, _, _ = digitize_curve_from_pdf(cfg.cum_pdf, kind="dashed")
        inf_solid, xlim_inf, ylim_inf = digitize_curve_from_pdf(cfg.inf_pdf, kind="solid")
        inf_dash, _, _ = digitize_curve_from_pdf(cfg.inf_pdf, kind="dashed")

        x_min = min(xlim_cum[0], xlim_inf[0])
        x_max = max(xlim_cum[1], xlim_inf[1])
        grid = np.linspace(x_min, x_max, 801)

        model_cum = np.interp(grid, t_aligned, sol.cumulative, left=np.nan, right=np.nan)
        model_inf = np.interp(grid, t_aligned, sol.I, left=np.nan, right=np.nan)
        paper_cum_s = _interp_to(grid, cum_solid)
        paper_cum_d = _interp_to(grid, cum_dash)
        paper_inf_s = _interp_to(grid, inf_solid)
        paper_inf_d = _interp_to(grid, inf_dash)

        rmse_cum_vs_sim = _rmse(model_cum, paper_cum_s)
        rmse_inf_vs_sim = _rmse(model_inf, paper_inf_s)
        rmse_cum_vs_theory = _rmse(model_cum, paper_cum_d)
        rmse_inf_vs_theory = _rmse(model_inf, paper_inf_d)
        tpk_m, ipk_m = _peak(grid, model_inf)
        tpk_s, ipk_s = _peak(grid, paper_inf_s)

        metrics_rows.append(
            {
                "figure": cfg.fig_no,
                "key": cfg.key,
                "rmse_cum_vs_paper_sim": rmse_cum_vs_sim,
                "rmse_inf_vs_paper_sim": rmse_inf_vs_sim,
                "rmse_cum_vs_paper_theory": rmse_cum_vs_theory,
                "rmse_inf_vs_paper_theory": rmse_inf_vs_theory,
                "peak_t_abs_err_vs_sim": abs(tpk_m - tpk_s),
                "peak_i_abs_err_vs_sim": abs(ipk_m - ipk_s),
            }
        )

        fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.4), sharex=True)
        ax_cum, ax_inf = axes
        ax_cum.plot(grid, paper_cum_s, color="#111827", lw=1.6, alpha=0.9, label="Paper simulation (digitized)")
        ax_cum.plot(grid, paper_cum_d, color="#9ca3af", lw=1.4, ls="--", alpha=0.9, label="Paper theory (digitized)")
        ax_cum.plot(grid, model_cum, color="#2563eb", lw=2.1, label="Our ODE")
        ax_cum.set_ylabel("Cumulative infections")
        ax_cum.set_ylim(ylim_cum[0], ylim_cum[1])
        ax_cum.grid(alpha=0.22)
        ax_cum.legend(loc="best", fontsize=8, frameon=True)

        ax_inf.plot(grid, paper_inf_s, color="#111827", lw=1.6, alpha=0.9, label="Paper simulation (digitized)")
        ax_inf.plot(grid, paper_inf_d, color="#9ca3af", lw=1.4, ls="--", alpha=0.9, label="Paper theory (digitized)")
        ax_inf.plot(grid, model_inf, color="#dc2626", lw=2.1, label="Our ODE")
        ax_inf.set_ylabel("Infections")
        ax_inf.set_xlabel("t")
        ax_inf.set_ylim(ylim_inf[0], ylim_inf[1])
        ax_inf.grid(alpha=0.22)
        ax_inf.legend(loc="best", fontsize=8, frameon=True)
        ax_inf.set_xlim(x_min, x_max)

        fig.suptitle(cfg.title)
        fig.tight_layout()
        out_base = args.output_dir / f"fig{cfg.fig_no}_{cfg.key}_example_repro"
        fig.savefig(out_base.with_suffix(".png"), dpi=240)
        fig.savefig(out_base.with_suffix(".pdf"))
        plt.close(fig)

    report_csv = args.output_dir / "fig4to15_example_metrics.csv"
    with report_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metrics_rows)
    print("done")


if __name__ == "__main__":
    main()
