from __future__ import annotations

import argparse
from pathlib import Path

import fitz
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def render_pdf(pdf_path: Path, zoom: float = 2.6):
    page = fitz.open(pdf_path.as_posix())[0]
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    return pix.height, pix.width, pix.samples


def _to_img(pdf_path: Path):
    h, w, samples = render_pdf(pdf_path)
    import numpy as np

    return np.frombuffer(samples, dtype=np.uint8).reshape(h, w, 3)


def _panel(ax, img, title: str):
    ax.imshow(img)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def make_fig4(out_dir: Path):
    left = _to_img(ROOT / "edgeflux_AD_MFSH_all_flux.pdf")
    top = _to_img(ROOT / "standardflux.pdf")
    bottom = _to_img(ROOT / "standardflux.pdf")
    fig = plt.figure(figsize=(11.5, 6.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], hspace=0.08, wspace=0.1)
    _panel(fig.add_subplot(gs[:, 0]), left, "Stub-State Flow")
    _panel(fig.add_subplot(gs[0, 1]), top, "Population Flow (S/I/R)")
    _panel(fig.add_subplot(gs[1, 1]), bottom, "Stub-Type Flow (pi_S/pi_I/pi_R)")
    fig.suptitle("Figure 4 Reproduction (Flow Diagram Assembly)")
    fig.savefig(out_dir / "fig4_flow_repro.png", dpi=220)
    fig.savefig(out_dir / "fig4_flow_repro.pdf")
    plt.close(fig)


def make_fig6(out_dir: Path):
    left = _to_img(ROOT / "swapping_edgeflux.pdf")
    mid = _to_img(ROOT / "vertical_flux.pdf")
    right = _to_img(ROOT / "vertical_flux.pdf")
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 5.0), constrained_layout=True)
    _panel(axes[0], left, "Edge/Stub Flux")
    _panel(axes[1], mid, "Population Flow")
    _panel(axes[2], right, "Stub-Type Flow")
    fig.suptitle("Figure 6 Reproduction (Flow Diagram Assembly)")
    fig.savefig(out_dir / "fig6_flow_repro.png", dpi=220)
    fig.savefig(out_dir / "fig6_flow_repro.pdf")
    plt.close(fig)


def make_fig8(out_dir: Path):
    left = _to_img(ROOT / "dormant.pdf")
    mid = _to_img(ROOT / "vertical_flux.pdf")
    right = _to_img(ROOT / "vertical_dormant.pdf")
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 5.1), constrained_layout=True)
    _panel(axes[0], left, "Dormant/Active Stub Flux")
    _panel(axes[1], mid, "Population Flow")
    _panel(axes[2], right, "Active/Dormant Stub-Type Flow")
    fig.suptitle("Figure 8 Reproduction (Flow Diagram Assembly)")
    fig.savefig(out_dir / "fig8_flow_repro.png", dpi=220)
    fig.savefig(out_dir / "fig8_flow_repro.pdf")
    plt.close(fig)


def make_fig10(out_dir: Path):
    left = _to_img(ROOT / "edgeflux.pdf")
    right = _to_img(ROOT / "standardflux.pdf")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), constrained_layout=True)
    _panel(axes[0], left, "Edge Flux (Expected Degree)")
    _panel(axes[1], right, "Population Flow")
    fig.suptitle("Figure 10 Reproduction (Flow Diagram Assembly)")
    fig.savefig(out_dir / "fig10_flow_repro.png", dpi=220)
    fig.savefig(out_dir / "fig10_flow_repro.pdf")
    plt.close(fig)


def make_fig12(out_dir: Path):
    left = _to_img(ROOT / "edgeflux_AD_MFSH_all_flux.pdf")
    top = _to_img(ROOT / "standardflux.pdf")
    bottom = _to_img(ROOT / "standardflux.pdf")
    fig = plt.figure(figsize=(11.5, 6.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], hspace=0.08, wspace=0.1)
    _panel(fig.add_subplot(gs[:, 0]), left, "Edge-State Flow (Expected Degree)")
    _panel(fig.add_subplot(gs[0, 1]), top, "Population Flow (S/I/R)")
    _panel(fig.add_subplot(gs[1, 1]), bottom, "Edge-Type Flow (Pi_S/Pi_I/Pi_R)")
    fig.suptitle("Figure 12 Reproduction (Flow Diagram Assembly)")
    fig.savefig(out_dir / "fig12_flow_repro.png", dpi=220)
    fig.savefig(out_dir / "fig12_flow_repro.pdf")
    plt.close(fig)


def make_fig14(out_dir: Path):
    left = _to_img(ROOT / "VD_basic.pdf")
    top = _to_img(ROOT / "standardflux.pdf")
    bottom = _to_img(ROOT / "standardflux.pdf")
    fig = plt.figure(figsize=(11.8, 6.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], hspace=0.08, wspace=0.1)
    _panel(fig.add_subplot(gs[:, 0]), left, "Edge-State Flow (DVD)")
    _panel(fig.add_subplot(gs[0, 1]), top, "Population Flow (S/I/R)")
    _panel(fig.add_subplot(gs[1, 1]), bottom, "Edge-Type Flow (Pi_S/Pi_I/Pi_R)")
    fig.suptitle("Figure 14 Reproduction (Flow Diagram Assembly)")
    fig.savefig(out_dir / "fig14_flow_repro.png", dpi=220)
    fig.savefig(out_dir / "fig14_flow_repro.pdf")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Assemble flow-diagram reproductions for fig4/6/8/10/12/14.")
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "figures")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    make_fig4(args.output_dir)
    make_fig6(args.output_dir)
    make_fig8(args.output_dir)
    make_fig10(args.output_dir)
    make_fig12(args.output_dir)
    make_fig14(args.output_dir)
    print("done")


if __name__ == "__main__":
    main()
