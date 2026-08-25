#!/usr/bin/env python3
"""Emit the FCGS rows for latex_table.tex, one per lambda.

Aggregation matches the rest of the table: per scene the seeds are averaged,
then the scenes of a dataset are averaged. A scene whose seeds all failed
contributes nothing; if a whole dataset is missing, its cells become "--", and
any dataset averaged over an incomplete scene set is reported so the caller can
footnote it.

    python3 scripts/make_latex_rows.py --csv /d01/luis/runs_fcgs/metrics.csv \
        --patch /d01/luis/latex_table.tex
"""

import argparse
import csv
import re
import statistics
from collections import defaultdict

SCENE_TO_DATASET = {
    "bicycle": "MipNeRF360", "bonsai": "MipNeRF360", "counter": "MipNeRF360",
    "flowers": "MipNeRF360", "garden": "MipNeRF360", "kitchen": "MipNeRF360",
    "room": "MipNeRF360", "stump": "MipNeRF360", "treehill": "MipNeRF360",
    "train": "TanksAndTemples", "truck": "TanksAndTemples",
    "drjohnson": "DeepBlending", "playroom": "DeepBlending",
}
# Column order of the table: Tanks&Temples, Mip-NeRF360, Deep Blending
DATASET_COLS = ["TanksAndTemples", "MipNeRF360", "DeepBlending"]
N_SCENES = {"TanksAndTemples": 2, "MipNeRF360": 9, "DeepBlending": 2}
LMD_ORDER = ["1e-4", "2e-4", "4e-4", "8e-4", "16e-4"]
LMD_TEX = {
    "1e-4": r"1{\times}10^{-4}", "2e-4": r"2{\times}10^{-4}",
    "4e-4": r"4{\times}10^{-4}", "8e-4": r"8{\times}10^{-4}",
    "16e-4": r"16{\times}10^{-4}",
}


# Values published by the FCGS authors, in the column order of the table
# (Tanks&Temples, Mip-NeRF360, Deep Blending), reproduced from
# results/<dataset>/<scene>.csv.
#
# Sizes are restated in MiB (2^20 bytes) so the whole table shares one unit:
# the authors publish MB = 10^6 bytes, while every locally measured row comes
# from compress.py, which divides by 1024**2. The published decimal figures
# were 33,6 / 67,2 / 54,5 and 18,8 / 36,3 / 30,1 -- 4.86% larger purely from
# the unit. Quality metrics are the published values, verified against the
# per-scene CSVs. Compression time is the range the paper reports; decode time
# is not given.
AUTHOR_ROWS = [
    (r"FCGS~\cite{chen2024fcgs} (alta taxa)",
     ["23,62", "0,839", "0,184", "32,0",
      "27,39", "0,806", "0,226", "64,0",
      "29,58", "0,899", "0,248", "52,0", "11--36", "--"]),
    (r"FCGS~\cite{chen2024fcgs} (baixa taxa)",
     ["23,48", "0,833", "0,193", "17,9",
      "27,05", "0,798", "0,237", "34,6",
      "29,27", "0,893", "0,257", "28,7", "11--36", "--"]),
]


def br(value, decimals):
    """Brazilian decimal comma, matching the rest of the table."""
    if value is None:
        return "--"
    return f"{value:.{decimals}f}".replace(".", ",")


def load(csv_path):
    per_scene = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in csv.DictReader(open(csv_path)):
        scene = row["scene"]
        if scene not in SCENE_TO_DATASET or not row.get("PSNR"):
            continue
        lmd = row["method"].replace("FCGS_", "")
        for m in ("PSNR", "SSIM", "LPIPS", "size_MB", "time_total", "time_ac_decode"):
            if row.get(m):
                per_scene[lmd][scene][m].append(float(row[m]))
    return per_scene


def build_row(per_scene, lmd):
    scenes = per_scene[lmd]
    cells, notes = [], []
    for ds in DATASET_COLS:
        members = [s for s in scenes if SCENE_TO_DATASET[s] == ds]
        if not members:
            cells += ["--"] * 4
            notes.append(f"{ds}: sem dados")
            continue
        if len(members) < N_SCENES[ds]:
            missing = [s for s, d in SCENE_TO_DATASET.items()
                       if d == ds and s not in members]
            notes.append(f"{ds}: {len(members)}/{N_SCENES[ds]} cenas "
                         f"(faltam {', '.join(sorted(missing))})")
        agg = {}
        for m in ("PSNR", "SSIM", "LPIPS", "size_MB"):
            vals = [statistics.mean(scenes[s][m]) for s in members if scenes[s][m]]
            agg[m] = statistics.mean(vals) if vals else None
        cells += [br(agg["PSNR"], 2), br(agg["SSIM"], 3),
                  br(agg["LPIPS"], 3), br(agg["size_MB"], 1)]

    comp = [statistics.mean(v["time_total"]) for v in scenes.values() if v["time_total"]]
    dec = [statistics.mean(v["time_ac_decode"]) for v in scenes.values() if v["time_ac_decode"]]
    cells.append(br(statistics.mean(comp), 0) if comp else "--")
    cells.append(br(statistics.mean(dec), 0) if dec else "--")

    label = f"FCGS (reproduzido, $\\lambda{{=}}{LMD_TEX[lmd]}$)"
    return f"{label} & " + " & ".join(cells) + r" \\", notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/d01/luis/runs_fcgs/metrics.csv")
    ap.add_argument("--patch", default=None,
                    help="latex_table.tex to rewrite the FCGS block in")
    args = ap.parse_args()

    per_scene = load(args.csv)
    rows = [f"{lbl} & " + " & ".join(cells) + r" \\"
            for lbl, cells in AUTHOR_ROWS]
    rows.append(r"\midrule")
    all_notes = []
    for lmd in LMD_ORDER:
        if lmd not in per_scene:
            continue
        row, notes = build_row(per_scene, lmd)
        rows.append(row)
        for n in notes:
            all_notes.append(f"  lambda={lmd}  {n}")

    block = "\n".join(rows)
    print(block)
    if all_notes:
        print("\n% cobertura incompleta:")
        for n in all_notes:
            print("%" + n)

    if args.patch:
        text = open(args.patch).read()
        # Replace the placeholder FCGS row (or a previously generated block)
        # Spans the whole FCGS block, including the \midrule this generator
        # puts between the authors' rows and ours -- otherwise a second run
        # would replace only the first rows and duplicate the rest.
        pattern = re.compile(
            r"(?m)^FCGS[^\n]*\\\\\n(?:^(?:FCGS[^\n]*\\\\|\\midrule)\n)*"
        )
        if not pattern.search(text):
            raise SystemExit("no FCGS row found in the table to replace")
        # lambda replacement: the LaTeX block is full of backslashes that re.sub
        # would otherwise read as group escapes.
        text = pattern.sub(lambda _: block + "\n", text, count=1)
        open(args.patch, "w").write(text)
        print(f"\npatched {args.patch}")


if __name__ == "__main__":
    main()
