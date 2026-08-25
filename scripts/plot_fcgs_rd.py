#!/usr/bin/env python3
"""Rate-distortion figures for the FCGS lambda sweep: LPIPS vs bitstream size.

One line per dataset, one marker per lambda. Two variants are written: with
error bars (dispersion across the scenes of each dataset) and without.

    python3 scripts/plot_fcgs_rd.py --csv /d01/luis/runs_fcgs/metrics.csv \
        --outdir /d01/luis/figures
"""

import argparse
import csv
import os
import statistics
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages

SCENE_TO_DATASET = {
    "bicycle": "MipNeRF360", "bonsai": "MipNeRF360", "counter": "MipNeRF360",
    "flowers": "MipNeRF360", "garden": "MipNeRF360", "kitchen": "MipNeRF360",
    "room": "MipNeRF360", "stump": "MipNeRF360", "treehill": "MipNeRF360",
    "train": "TanksAndTemples", "truck": "TanksAndTemples",
    "drjohnson": "DeepBlending", "playroom": "DeepBlending",
}
DATASET_LABEL = {
    "MipNeRF360": "Mip-NeRF 360",
    "TanksAndTemples": "Tanks & Temples",
    "DeepBlending": "Deep Blending",
}
# Categorical slots 1-3 of the reference palette: the documented trio that
# validates on the all-pairs list (CVD dE 9.2, normal-vision 24.0, light mode).
SERIES_COLOR = {
    "MipNeRF360": "#2a78d6",      # slot 1, blue
    "TanksAndTemples": "#eb6834",  # slot 2, orange
    "DeepBlending": "#1baf7a",     # slot 3, aqua
}
MARKERS = {"MipNeRF360": "o", "TanksAndTemples": "s", "DeepBlending": "^"}
ORDER = ["MipNeRF360", "TanksAndTemples", "DeepBlending"]
LMD_ORDER = ["1e-4", "2e-4", "4e-4", "8e-4", "16e-4"]

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SURFACE = "#ffffff"
GRID = "#d9d8d4"

BASE_FONT = 20


def load(csv_path):
    """-> {lmd: {dataset: {'size': [per-scene means], 'lpips': [...]}}}"""
    per_scene = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            scene = row["scene"]
            if scene not in SCENE_TO_DATASET or not row.get("LPIPS"):
                continue
            lmd = row["method"].replace("FCGS_", "")
            per_scene[lmd][scene]["lpips"].append(float(row["LPIPS"]))
            per_scene[lmd][scene]["size"].append(float(row["size_MB"]))

    out = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for lmd, scenes in per_scene.items():
        for scene, m in scenes.items():
            ds = SCENE_TO_DATASET[scene]
            out[lmd][ds]["lpips"].append(statistics.mean(m["lpips"]))
            out[lmd][ds]["size"].append(statistics.mean(m["size"]))
            out[lmd][ds]["scenes"].append(scene)
    return out


def load_authors(results_dir):
    """The numbers the FCGS authors publish in results/<dataset>/<scene>.csv.

    Two operating points per scene (highrate / lowrate). Sizes are bytes, so
    they get the same MiB conversion the run pipeline uses.

    These come from the authors' own trained 3DGS models at their evaluation
    resolution -- different gaussian counts and different ground-truth images
    from the local models -- so the two families are reference curves side by
    side, not a like-for-like comparison.
    """
    per_ds = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for scene, ds in SCENE_TO_DATASET.items():
        path = os.path.join(results_dir, ds, f"{scene}.csv")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for row in csv.DictReader(f):
                rate = row["Submethod"].replace("FCGS-", "")
                per_ds[ds][rate]["lpips"].append(float(row["LPIPS"]))
                per_ds[ds][rate]["size"].append(
                    float(row["Size [Bytes]"]) / 1024 / 1024)
    return per_ds


def authors_points(authors, ds):
    """(x, y, xerr, yerr, n) per rate point, ordered high-rate first."""
    pts = []
    for rate in ("highrate", "lowrate"):
        m = authors.get(ds, {}).get(rate)
        if not m:
            continue
        pts.append((
            statistics.mean(m["size"]),
            statistics.mean(m["lpips"]),
            statistics.stdev(m["size"]) if len(m["size"]) > 1 else 0.0,
            statistics.stdev(m["lpips"]) if len(m["lpips"]) > 1 else 0.0,
            len(m["size"]),
        ))
    return pts


def series_points(data, ds):
    """Per-lambda (x, y, xerr, yerr, n_scenes) for one dataset."""
    pts = []
    for lmd in LMD_ORDER:
        if lmd not in data or ds not in data[lmd] or not data[lmd][ds]["lpips"]:
            continue
        sizes = data[lmd][ds]["size"]
        lp = data[lmd][ds]["lpips"]
        pts.append((
            statistics.mean(sizes),
            statistics.mean(lp),
            statistics.stdev(sizes) if len(sizes) > 1 else 0.0,
            statistics.stdev(lp) if len(lp) > 1 else 0.0,
            len(sizes),
            lmd,
        ))
    return pts


def draw(data, authors, path_png, path_pdf, errorbars, figsize, combined_pdf=None):
    plt.rcParams.update({
        "font.size": BASE_FONT,
        "axes.titlesize": BASE_FONT + 2,
        "axes.labelsize": BASE_FONT,
        "xtick.labelsize": BASE_FONT - 2,
        "ytick.labelsize": BASE_FONT - 2,
        "legend.fontsize": BASE_FONT - 2,
        "font.family": "DejaVu Sans",
    })
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for ds in ORDER:
        pts = series_points(data, ds)
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        color = SERIES_COLOR[ds]
        if errorbars:
            # Kept deliberately recessive: the across-scene spread is an order of
            # magnitude wider than the lambda sweep itself, so at full weight it
            # buries the very curve the figure is about.
            ax.errorbar(
                xs, ys,
                xerr=[p[2] for p in pts], yerr=[p[3] for p in pts],
                fmt="none", ecolor=color, elinewidth=1.5, capsize=4,
                capthick=1.5, alpha=0.35, zorder=2,
            )
        ax.plot(xs, ys, "-", color=color, linewidth=2, zorder=3)
        # 2px surface ring so overlapping markers stay separable
        ax.plot(
            xs, ys, MARKERS[ds], color=color, markersize=11,
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=4,
            label=DATASET_LABEL[ds],
        )

    # The authors' published operating points. Colour still carries the dataset;
    # the dashed line and hollow marker carry the source, so the two families
    # are never told apart by colour alone.
    for ds in ORDER:
        pts = authors_points(authors, ds)
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        color = SERIES_COLOR[ds]
        if errorbars:
            ax.errorbar(
                xs, ys,
                xerr=[p[2] for p in pts], yerr=[p[3] for p in pts],
                fmt="none", ecolor=color, elinewidth=1.5, capsize=4,
                capthick=1.5, alpha=0.3, zorder=2,
            )
        ax.plot(xs, ys, "--", color=color, linewidth=2, alpha=0.9, zorder=3)
        ax.plot(
            xs, ys, MARKERS[ds], color=color, markersize=11,
            markerfacecolor=SURFACE, markeredgewidth=2.5, zorder=4,
        )

    # Direct labels: the relief rule for the low-contrast slot, and identity
    # that never depends on colour alone. Anchored at the low-rate end and
    # nudged clear of the line so they never sit on top of a marker.
    # Anchored to the right of each curve's high-rate end and given room in the
    # margin. Offsets measured from a data point in *points* do not survive the
    # two figures having very different y ranges, so the labels go where the
    # curves provably are not: past their right-hand end.
    ax.set_xlim(right=ax.get_xlim()[1] + 0.16 * (ax.get_xlim()[1] - ax.get_xlim()[0]))
    for ds in ORDER:
        pts = series_points(data, ds)
        if not pts:
            continue
        x, y = pts[0][0], pts[0][1]
        ax.annotate(
            DATASET_LABEL[ds], xy=(x, y), xytext=(18, 0),
            textcoords="offset points", ha="left", va="center",
            color=TEXT_PRIMARY, fontsize=BASE_FONT - 4, fontweight="bold",
            zorder=6,
        )

    # Only the two ends of the sweep are annotated. The x axis already encodes
    # rate, so a lambda on every marker was pure collision with no new
    # information.
    pts = series_points(data, "MipNeRF360")
    if pts:
        # The low-rate end sits just right of the authors' hollow marker, so its
        # label is pushed right rather than centred on top of it.
        for point, (dx, dy, ha) in ((pts[0], (0, 20, "center")),
                                    (pts[-1], (16, 14, "left"))):
            x, y, _, _, _, lmd = point
            ax.annotate(
                f"$\\lambda$={lmd}", xy=(x, y), xytext=(dx, dy),
                textcoords="offset points", ha=ha,
                color=TEXT_SECONDARY, fontsize=BASE_FONT - 5, zorder=6,
            )

    ax.set_xlabel("Tamanho do bitstream (MB)  $\\downarrow$", color=TEXT_PRIMARY)
    ax.set_ylabel("LPIPS  $\\downarrow$", color=TEXT_PRIMARY)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY)

    # Two legends, because two things are encoded: colour = dataset,
    # line style = which measurement the curve comes from.
    leg = ax.legend(loc="lower right", frameon=True, facecolor=SURFACE,
                    edgecolor=GRID, framealpha=0.95)
    for text in leg.get_texts():
        text.set_color(TEXT_PRIMARY)
    ax.add_artist(leg)

    style_handles = [
        Line2D([], [], color=TEXT_SECONDARY, linestyle="-", linewidth=2,
               marker="o", markersize=10, markeredgecolor=SURFACE,
               markeredgewidth=2, label="Este trabalho (5 $\\lambda$)"),
        Line2D([], [], color=TEXT_SECONDARY, linestyle="--", linewidth=2,
               marker="o", markersize=10, markerfacecolor=SURFACE,
               markeredgewidth=2.5, label="Autores do FCGS (2 taxas)"),
    ]
    # Placed above the axes: inside the plot it covered the Deep Blending
    # dashed line, and there is no interior region wide enough for it.
    leg2 = ax.legend(handles=style_handles, loc="lower left",
                     bbox_to_anchor=(0.0, 1.01), ncol=2, frameon=False,
                     handlelength=3.0, columnspacing=2.5)
    for text in leg2.get_texts():
        text.set_color(TEXT_PRIMARY)

    if errorbars:
        # Below the axes, not beside the legend: at font 20 the two collide on
        # the same line.
        fig.text(
            0.99, 0.012, "Barras: $\\pm$1 desvio-padrão entre as cenas do conjunto",
            ha="right", va="bottom", color=TEXT_SECONDARY,
            fontsize=BASE_FONT - 5,
        )

    fig.tight_layout()
    fig.savefig(path_png, dpi=200, facecolor=SURFACE)
    print(f"wrote {path_png}")

    # Full-bleed PDF: the axes are stretched to the page, no surrounding margin
    # beyond what the labels need.
    with PdfPages(path_pdf) as pdf:
        pdf.savefig(fig, facecolor=SURFACE, bbox_inches=None)
    print(f"wrote {path_pdf}")
    if combined_pdf is not None:
        combined_pdf.savefig(fig, facecolor=SURFACE, bbox_inches=None)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/d01/luis/runs_fcgs/metrics.csv")
    ap.add_argument("--outdir", default="/d01/luis/figures")
    ap.add_argument("--results_dir", default="/d01/luis/FCGS/results",
                    help="the authors' published per-scene CSVs")
    # A4 landscape, so the figure fills the whole PDF page.
    ap.add_argument("--width", type=float, default=11.69)
    ap.add_argument("--height", type=float, default=8.27)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    data = load(args.csv)
    authors = load_authors(args.results_dir)

    for ds in ORDER:
        for x, y, xe, ye, n, lmd in series_points(data, ds):
            print(f"{ds:<16} lmd={lmd:<6} size={x:7.2f}+-{xe:6.2f} "
                  f"LPIPS={y:.4f}+-{ye:.4f}  ({n} cenas)")

    combined_path = os.path.join(args.outdir, "fcgs_graficos.pdf")
    with PdfPages(combined_path) as combined:
        draw(data, authors,
             os.path.join(args.outdir, "fcgs_lpips_vs_size_errorbars.png"),
             os.path.join(args.outdir, "fcgs_lpips_vs_size_errorbars.pdf"),
             errorbars=True, figsize=(args.width, args.height),
             combined_pdf=combined)
        draw(data, authors,
             os.path.join(args.outdir, "fcgs_lpips_vs_size.png"),
             os.path.join(args.outdir, "fcgs_lpips_vs_size.pdf"),
             errorbars=False, figsize=(args.width, args.height),
             combined_pdf=combined)
    print(f"wrote {combined_path} (2 páginas)")


if __name__ == "__main__":
    main()
