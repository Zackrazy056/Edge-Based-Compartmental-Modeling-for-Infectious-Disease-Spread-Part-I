from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import fitz
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ebcm_repro.models import empirical_degree_pgf, solve_cm, solve_mass_action
from ebcm_repro.simulation import build_configuration_model_edges, run_sir_tauleap, sample_degree_sequence


Color = tuple[float, float, float]

COLOR_BY_NAME: dict[str, Color] = {
    "blue": (0.353, 0.702, 0.902),
    "red": (1.0, 0.125, 0.0),
    "green": (0.0, 0.604, 0.502),
    "orange": (0.902, 0.604, 0.0),
    "ma": (0.804, 0.604, 0.702),
}
DIST_ORDER_FROM_LATEST_PEAK = ["homogeneous", "poisson", "bimodal", "truncated_powerlaw"]


@dataclass(frozen=True)
class AxisBounds:
    x_left: float
    x_right: float
    y_top: float
    y_bottom: float


@dataclass(frozen=True)
class Curve:
    t: np.ndarray
    y: np.ndarray


def _color_tuple(color_obj: tuple[float, ...] | None) -> tuple[float, ...] | None:
    if color_obj is None:
        return None
    return tuple(round(float(v), 3) for v in color_obj)


def _drawing_points(draw: dict) -> np.ndarray:
    pts: list[tuple[float, float]] = []
    for item in draw.get("items", []):
        cmd = item[0]
        if cmd == "l":
            p1, p2 = item[1], item[2]
            pts.append((float(p1.x), float(p1.y)))
            pts.append((float(p2.x), float(p2.y)))
        elif cmd == "c":
            # Bezier segment: include all control points to preserve geometry envelope.
            for p in item[1:5]:
                pts.append((float(p.x), float(p.y)))
    if not pts:
        return np.empty((0, 2), dtype=float)
    arr = np.asarray(pts, dtype=float)
    return arr


def _axis_bounds_from_drawings(drawings: list[dict]) -> AxisBounds:
    lines = []
    for d in drawings:
        if _color_tuple(d.get("color")) != (0.0, 0.0, 0.0):
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
        length = math.hypot(x2 - x1, y2 - y1)
        if length > 100:
            lines.append((x1, y1, x2, y2))

    x_vert = sorted({round(x1, 4) for x1, y1, x2, y2 in lines if abs(x1 - x2) < 1e-5})
    y_hori = sorted({round(y1, 4) for x1, y1, x2, y2 in lines if abs(y1 - y2) < 1e-5})
    if len(x_vert) < 2 or len(y_hori) < 2:
        raise RuntimeError("failed to detect axis bounds from vector lines")
    return AxisBounds(
        x_left=float(x_vert[0]),
        x_right=float(x_vert[-1]),
        y_top=float(y_hori[0]),
        y_bottom=float(y_hori[-1]),
    )


def _curve_from_drawings(
    drawings: list[dict],
    axis: AxisBounds,
    color: Color,
    width: float,
    dashes: str,
    y_max: float,
) -> Curve:
    matches = []
    for d in drawings:
        if _color_tuple(d.get("color")) != color:
            continue
        if d.get("dashes") != dashes:
            continue
        if abs(float(d.get("width") or 0.0) - width) > 1e-6:
            continue
        if len(d.get("items", [])) <= 5:
            continue
        matches.append(d)
    if len(matches) != 1:
        raise RuntimeError(f"expected 1 drawing match, got {len(matches)} for color={color}, width={width}, dashes={dashes}")

    pts = _drawing_points(matches[0])
    if pts.size == 0:
        raise RuntimeError("empty path points")
    in_x = (pts[:, 0] >= axis.x_left - 1e-8) & (pts[:, 0] <= axis.x_right + 1e-8)
    pts = pts[in_x]
    x = pts[:, 0]
    y = pts[:, 1]

    # Collapse duplicate x by median y, then map to data coordinates.
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    x_rounded = np.round(x, 4)
    uniq_x = np.unique(x_rounded)
    out_x = np.empty_like(uniq_x, dtype=float)
    out_y = np.empty_like(uniq_x, dtype=float)
    for i, ux in enumerate(uniq_x):
        sel = x_rounded == ux
        out_x[i] = float(np.median(x[sel]))
        out_y[i] = float(np.median(y[sel]))

    t = -5.0 + 20.0 * (out_x - axis.x_left) / (axis.x_right - axis.x_left)
    val = y_max * (axis.y_bottom - out_y) / (axis.y_bottom - axis.y_top)
    val = np.clip(val, 0.0, y_max)
    return Curve(t=t, y=val)


def _interp_curve(curve: Curve, t_grid: np.ndarray) -> np.ndarray:
    if curve.t.size == 0:
        return np.full_like(t_grid, np.nan)
    order = np.argsort(curve.t)
    t = curve.t[order]
    y = curve.y[order]
    t_u, idx = np.unique(t, return_index=True)
    y_u = y[idx]
    out = np.interp(t_grid, t_u, y_u, left=np.nan, right=np.nan)
    return out


def digitize_fig3_from_vector(cum_pdf: Path, inf_pdf: Path, t_grid: np.ndarray) -> dict:
    cum_drawings = fitz.open(cum_pdf.as_posix())[0].get_drawings()
    inf_drawings = fitz.open(inf_pdf.as_posix())[0].get_drawings()

    axis_cum = _axis_bounds_from_drawings(cum_drawings)
    axis_inf = _axis_bounds_from_drawings(inf_drawings)

    solid_inf_by_color = {
        cname: _curve_from_drawings(inf_drawings, axis_inf, color, width=1.0, dashes="[] 0", y_max=0.35)
        for cname, color in COLOR_BY_NAME.items()
        if cname != "ma"
    }

    peaks = {cname: float(curve.t[np.argmax(curve.y)]) for cname, curve in solid_inf_by_color.items()}
    color_sorted = sorted(peaks.keys(), key=lambda c: peaks[c], reverse=True)
    color_to_dist = {c: dist for c, dist in zip(color_sorted, DIST_ORDER_FROM_LATEST_PEAK)}

    result = {
        "axis": {
            "cum": axis_cum.__dict__,
            "inf": axis_inf.__dict__,
        },
        "color_to_distribution": color_to_dist,
        "target_grid_t": t_grid.tolist(),
        "targets": {},
    }

    for cname, dist in color_to_dist.items():
        solid_cum = _curve_from_drawings(cum_drawings, axis_cum, COLOR_BY_NAME[cname], width=1.0, dashes="[] 0", y_max=1.0)
        dash_cum = _curve_from_drawings(
            cum_drawings, axis_cum, COLOR_BY_NAME[cname], width=5.0, dashes="[ 6 6 ] 0", y_max=1.0
        )
        solid_inf = _curve_from_drawings(inf_drawings, axis_inf, COLOR_BY_NAME[cname], width=1.0, dashes="[] 0", y_max=0.35)
        dash_inf = _curve_from_drawings(
            inf_drawings, axis_inf, COLOR_BY_NAME[cname], width=5.0, dashes="[ 6 6 ] 0", y_max=0.35
        )
        result["targets"][dist] = {
            "solid_cum": _interp_curve(solid_cum, t_grid).tolist(),
            "dashed_cum": _interp_curve(dash_cum, t_grid).tolist(),
            "solid_inf": _interp_curve(solid_inf, t_grid).tolist(),
            "dashed_inf": _interp_curve(dash_inf, t_grid).tolist(),
        }

    ma_cum = _curve_from_drawings(cum_drawings, axis_cum, COLOR_BY_NAME["ma"], width=3.0, dashes="[ 1 3 ] 0", y_max=1.0)
    ma_inf = _curve_from_drawings(inf_drawings, axis_inf, COLOR_BY_NAME["ma"], width=3.0, dashes="[ 1 3 ] 0", y_max=0.35)
    result["targets"]["ma"] = {
        "cum": _interp_curve(ma_cum, t_grid).tolist(),
        "inf": _interp_curve(ma_inf, t_grid).tolist(),
    }
    return result


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


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if not np.any(m):
        return float("nan")
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))


def _peak_metrics(t: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    m = np.isfinite(y)
    if not np.any(m):
        return float("nan"), float("nan")
    tm = t[m]
    ym = y[m]
    i = int(np.argmax(ym))
    return float(tm[i]), float(ym[i])


def _simulate_scenario(
    scenario: str,
    seed: int,
    n_nodes: int,
    beta: float,
    gamma: float,
    t_max: float,
    dt: float,
    ode_steps: int,
    tp_nu: float,
    tp_cutoff: float,
    outbreak_attempts: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    for attempt in range(outbreak_attempts):
        rng = np.random.default_rng(seed + 10_000 * attempt)
        degrees = sample_degree_sequence(
            scenario,
            n_nodes=n_nodes,
            rng=rng,
            truncated_powerlaw_nu=tp_nu,
            truncated_powerlaw_cutoff=tp_cutoff,
        )
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
        if sim.cumulative[-1] <= 0.04:
            continue

        pgf = empirical_degree_pgf(degrees)
        i0 = initial_infected / n_nodes
        theta0 = _theta_from_i0(pgf, i0)
        cm = solve_cm(pgf=pgf, beta=beta, gamma=gamma, t_max=t_max, n_steps=ode_steps, theta0=theta0, r0=0.0)
        ma = solve_mass_action(beta_hat=beta * pgf.mean_degree, gamma=gamma, t_max=t_max, i0=i0, r0=0.0, n_steps=ode_steps)

        sim_shift = _first_crossing_time(sim.t, sim.cumulative, threshold=0.01)
        cm_shift = _first_crossing_time(cm.t, cm.cumulative, threshold=0.01)
        ma_shift = _first_crossing_time(ma.t, ma.cumulative, threshold=0.01)

        return (
            sim.t - sim_shift,
            sim.cumulative,
            sim.I,
            cm.t - cm_shift,
            cm.cumulative,
            cm.I,
            ma.t - ma_shift,
            ma.cumulative,
            ma.I,
        )
    raise RuntimeError(f"failed major outbreak for {scenario} with seed={seed}")


def _interp_on_grid(t: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    order = np.argsort(t)
    t = t[order]
    y = y[order]
    t_u, idx = np.unique(t, return_index=True)
    y_u = y[idx]
    return np.interp(grid, t_u, y_u, left=np.nan, right=np.nan)


def _score_vs_target(
    pred_cum: np.ndarray,
    pred_inf: np.ndarray,
    tgt_cum: np.ndarray,
    tgt_inf: np.ndarray,
    t_grid: np.ndarray,
) -> tuple[float, dict[str, float]]:
    rmse_cum = _rmse(pred_cum, tgt_cum)
    rmse_inf = _rmse(pred_inf, tgt_inf)
    tpk_p, ypk_p = _peak_metrics(t_grid, pred_inf)
    tpk_t, ypk_t = _peak_metrics(t_grid, tgt_inf)
    dt_peak = abs(tpk_p - tpk_t)
    dy_peak = abs(ypk_p - ypk_t)
    score = rmse_inf + 0.7 * rmse_cum + 0.12 * dt_peak + 2.0 * dy_peak
    metrics = {
        "rmse_cum": rmse_cum,
        "rmse_inf": rmse_inf,
        "peak_t_pred": tpk_p,
        "peak_t_target": tpk_t,
        "peak_t_abs_err": dt_peak,
        "peak_i_pred": ypk_p,
        "peak_i_target": ypk_t,
        "peak_i_abs_err": dy_peak,
        "score": score,
    }
    return score, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Digitize Fig3 (vector PDF) and calibrate reproduction errors.")
    parser.add_argument("--cum-pdf", type=Path, default=ROOT / "CM_comp_cum_legend.pdf")
    parser.add_argument("--inf-pdf", type=Path, default=ROOT / "CM_comp_I_legend.pdf")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "figures")
    parser.add_argument("--n-nodes", type=int, default=120000)
    parser.add_argument("--beta", type=float, default=0.6)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--t-max", type=float, default=24.0)
    parser.add_argument("--dt", type=float, default=0.04)
    parser.add_argument("--ode-steps", type=int, default=1401)
    parser.add_argument("--seed-start", type=int, default=100)
    parser.add_argument("--seed-count", type=int, default=12)
    parser.add_argument("--tp-nu-min", type=float, default=1.36)
    parser.add_argument("--tp-nu-max", type=float, default=1.46)
    parser.add_argument("--tp-nu-num", type=int, default=6)
    parser.add_argument("--tp-cutoff", type=float, default=40.0)
    parser.add_argument("--grid-step", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    t_grid = np.arange(-5.0, 15.0 + 0.5 * args.grid_step, args.grid_step, dtype=float)

    print("[1/4] digitizing target curves from vector PDFs")
    digitized = digitize_fig3_from_vector(args.cum_pdf, args.inf_pdf, t_grid=t_grid)
    digitized_json = args.output_dir / "fig3_digitized_targets.json"
    digitized_json.write_text(json.dumps(digitized, indent=2), encoding="utf-8")

    scenarios = DIST_ORDER_FROM_LATEST_PEAK
    report_rows = []
    best_curves = {}
    tp_nu_grid = np.linspace(args.tp_nu_min, args.tp_nu_max, args.tp_nu_num)

    print("[2/4] searching calibrated seeds (and truncated-powerlaw nu grid)")
    for scenario in scenarios:
        target = digitized["targets"][scenario]
        tgt_solid_cum = np.asarray(target["solid_cum"], dtype=float)
        tgt_solid_inf = np.asarray(target["solid_inf"], dtype=float)
        tgt_dash_cum = np.asarray(target["dashed_cum"], dtype=float)
        tgt_dash_inf = np.asarray(target["dashed_inf"], dtype=float)
        best = None

        nu_candidates: Iterable[float] = tp_nu_grid if scenario == "truncated_powerlaw" else [1.418]
        for tp_nu in nu_candidates:
            for seed in range(args.seed_start, args.seed_start + args.seed_count):
                try:
                    sim_t, sim_cum, sim_inf, cm_t, cm_cum, cm_inf, ma_t, ma_cum, ma_inf = _simulate_scenario(
                        scenario=scenario,
                        seed=seed,
                        n_nodes=args.n_nodes,
                        beta=args.beta,
                        gamma=args.gamma,
                        t_max=args.t_max,
                        dt=args.dt,
                        ode_steps=args.ode_steps,
                        tp_nu=float(tp_nu),
                        tp_cutoff=args.tp_cutoff,
                    )
                except RuntimeError:
                    continue

                sim_cum_g = _interp_on_grid(sim_t, sim_cum, t_grid)
                sim_inf_g = _interp_on_grid(sim_t, sim_inf, t_grid)
                cm_cum_g = _interp_on_grid(cm_t, cm_cum, t_grid)
                cm_inf_g = _interp_on_grid(cm_t, cm_inf, t_grid)
                ma_cum_g = _interp_on_grid(ma_t, ma_cum, t_grid)
                ma_inf_g = _interp_on_grid(ma_t, ma_inf, t_grid)

                sim_score, sim_metrics = _score_vs_target(sim_cum_g, sim_inf_g, tgt_solid_cum, tgt_solid_inf, t_grid)
                _, cm_metrics = _score_vs_target(cm_cum_g, cm_inf_g, tgt_dash_cum, tgt_dash_inf, t_grid)
                _, ma_metrics = _score_vs_target(
                    ma_cum_g,
                    ma_inf_g,
                    np.asarray(digitized["targets"]["ma"]["cum"], dtype=float),
                    np.asarray(digitized["targets"]["ma"]["inf"], dtype=float),
                    t_grid,
                )

                total_score = sim_score + 0.35 * cm_metrics["score"]
                cand = {
                    "scenario": scenario,
                    "seed": seed,
                    "tp_nu": float(tp_nu),
                    "score_total": float(total_score),
                    "sim_metrics": sim_metrics,
                    "cm_metrics": cm_metrics,
                    "ma_metrics": ma_metrics,
                    "curves": {
                        "sim_cum": sim_cum_g,
                        "sim_inf": sim_inf_g,
                        "cm_cum": cm_cum_g,
                        "cm_inf": cm_inf_g,
                        "ma_cum": ma_cum_g,
                        "ma_inf": ma_inf_g,
                    },
                }
                if best is None or cand["score_total"] < best["score_total"]:
                    best = cand

        if best is None:
            raise RuntimeError(f"no valid calibration candidate for {scenario}")

        best_curves[scenario] = best["curves"]
        report_rows.append(
            {
                "scenario": scenario,
                "seed": best["seed"],
                "tp_nu": best["tp_nu"],
                "sim_rmse_cum": best["sim_metrics"]["rmse_cum"],
                "sim_rmse_inf": best["sim_metrics"]["rmse_inf"],
                "sim_peak_t_abs_err": best["sim_metrics"]["peak_t_abs_err"],
                "sim_peak_i_abs_err": best["sim_metrics"]["peak_i_abs_err"],
                "cm_rmse_cum": best["cm_metrics"]["rmse_cum"],
                "cm_rmse_inf": best["cm_metrics"]["rmse_inf"],
                "cm_peak_t_abs_err": best["cm_metrics"]["peak_t_abs_err"],
                "cm_peak_i_abs_err": best["cm_metrics"]["peak_i_abs_err"],
                "score_total": best["score_total"],
            }
        )

    print("[3/4] writing calibration reports")
    report_json = args.output_dir / "fig3_calibration_report.json"
    report_json.write_text(json.dumps({"rows": report_rows}, indent=2), encoding="utf-8")

    report_csv = args.output_dir / "fig3_calibration_report.csv"
    with report_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(report_rows[0].keys()))
        writer.writeheader()
        writer.writerows(report_rows)

    print("[4/4] plotting overlays")
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.6), sharex=True)
    ax_cum, ax_inf = axes
    color_lookup = {
        "homogeneous": "#5ab4e6",
        "poisson": "#ff1f00",
        "bimodal": "#009a80",
        "truncated_powerlaw": "#e69a00",
    }
    for scenario in scenarios:
        c = color_lookup[scenario]
        tgt = digitized["targets"][scenario]
        ax_cum.plot(t_grid, np.asarray(tgt["solid_cum"]), color=c, lw=1.0, alpha=0.55)
        ax_cum.plot(t_grid, np.asarray(best_curves[scenario]["sim_cum"]), color=c, lw=2.0, ls="--")
        ax_inf.plot(t_grid, np.asarray(tgt["solid_inf"]), color=c, lw=1.0, alpha=0.55, label=scenario)
        ax_inf.plot(t_grid, np.asarray(best_curves[scenario]["sim_inf"]), color=c, lw=2.0, ls="--")

    ax_cum.plot(t_grid, np.asarray(digitized["targets"]["ma"]["cum"]), color="#9c4f96", lw=1.0, alpha=0.5)
    ax_inf.plot(t_grid, np.asarray(digitized["targets"]["ma"]["inf"]), color="#9c4f96", lw=1.0, alpha=0.5, label="ma")
    ax_cum.set_ylabel("Cumulative infections")
    ax_inf.set_ylabel("Infections")
    ax_inf.set_xlabel("t")
    ax_cum.set_xlim(-5, 15)
    ax_cum.set_ylim(0, 1.0)
    ax_inf.set_ylim(0, 0.35)
    ax_inf.legend(loc="upper left", fontsize=8, frameon=True)
    ax_cum.grid(alpha=0.18)
    ax_inf.grid(alpha=0.18)
    ax_cum.set_title("Fig3 Digitized Target (thin) vs Calibrated Reproduction (dashed)")
    fig.tight_layout()
    fig.savefig(args.output_dir / "fig3_digitized_calibrated_overlay.png", dpi=240)
    fig.savefig(args.output_dir / "fig3_digitized_calibrated_overlay.pdf")
    plt.close(fig)
    print("done")


if __name__ == "__main__":
    main()
