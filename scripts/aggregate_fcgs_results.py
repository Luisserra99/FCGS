#!/usr/bin/env python3
"""Aggregate the FCGS seed sweep into per-dataset numbers.

Reads the CSV written by c3dgs/scripts/extract_metrics.py and reduces it in two
stages, the way the paper table reports things:

    per (lmd, scene): mean over the seeds that succeeded
    per (lmd, dataset): mean over the scenes of that dataset

A scene where every seed failed contributes nothing and is reported as missing,
so a dataset average is always labelled with how many scenes it covers.
"""

import argparse
import csv
import json
import statistics
from collections import defaultdict

SCENE_TO_DATASET = {
    "bicycle": "MipNeRF360", "bonsai": "MipNeRF360", "counter": "MipNeRF360",
    "flowers": "MipNeRF360", "garden": "MipNeRF360", "kitchen": "MipNeRF360",
    "room": "MipNeRF360", "stump": "MipNeRF360", "treehill": "MipNeRF360",
    "train": "TanksAndTemples", "truck": "TanksAndTemples",
    "drjohnson": "DeepBlending", "playroom": "DeepBlending",
}
DATASETS = ["TanksAndTemples", "MipNeRF360", "DeepBlending"]
SCENES_OF = defaultdict(list)
for _s, _d in SCENE_TO_DATASET.items():
    SCENES_OF[_d].append(_s)

METRICS = ["PSNR", "SSIM", "LPIPS", "size_MB", "time_total", "time_ac_decode"]
LMD_ORDER = ["1e-4", "2e-4", "4e-4", "8e-4", "16e-4"]


def lmd_label(method):
    return method.replace("FCGS_", "")


def load(csv_path):
    """rows -> {lmd: {scene: {metric: [values over seeds]}}}"""
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    seeds = defaultdict(lambda: defaultdict(list))
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            lmd = lmd_label(row["method"])
            scene = row["scene"]
            if scene not in SCENE_TO_DATASET:
                continue
            seeds[lmd][scene].append(row["seed"])
            for m in METRICS:
                if row.get(m, "") != "":
                    data[lmd][scene][m].append(float(row[m]))
    return data, seeds


def scene_means(data):
    """{lmd: {scene: {metric: (mean, std, n)}}}"""
    out = defaultdict(dict)
    for lmd, scenes in data.items():
        for scene, metrics in scenes.items():
            agg = {}
            for m, vals in metrics.items():
                if not vals:
                    continue
                agg[m] = (
                    statistics.mean(vals),
                    statistics.stdev(vals) if len(vals) > 1 else 0.0,
                    len(vals),
                )
            out[lmd][scene] = agg
    return out


def dataset_means(per_scene):
    """{lmd: {dataset: {metric: (mean_over_scenes, std_over_scenes,
                                mean_seed_std, n_scenes, n_expected)}}}"""
    out = defaultdict(dict)
    for lmd, scenes in per_scene.items():
        for ds in DATASETS:
            present = [s for s in SCENES_OF[ds] if s in scenes]
            agg = {}
            for m in METRICS:
                vals = [scenes[s][m][0] for s in present if m in scenes[s]]
                seed_stds = [scenes[s][m][1] for s in present if m in scenes[s]]
                if not vals:
                    continue
                agg[m] = (
                    statistics.mean(vals),
                    statistics.stdev(vals) if len(vals) > 1 else 0.0,
                    statistics.mean(seed_stds) if seed_stds else 0.0,
                    len(vals),
                    len(SCENES_OF[ds]),
                )
            agg["_scenes_present"] = present
            agg["_scenes_missing"] = [s for s in SCENES_OF[ds] if s not in scenes]
            out[lmd][ds] = agg
    return out


def overall_times(per_scene):
    """Mean compression / decompression time over every scene of a lmd."""
    out = {}
    for lmd, scenes in per_scene.items():
        comp = [v["time_total"][0] for v in scenes.values() if "time_total" in v]
        dec = [v["time_ac_decode"][0] for v in scenes.values() if "time_ac_decode" in v]
        out[lmd] = (
            statistics.mean(comp) if comp else None,
            statistics.mean(dec) if dec else None,
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="/d01/luis/runs_fcgs/metrics.csv")
    ap.add_argument("-o", "--output", default="/d01/luis/runs_fcgs/aggregated.json")
    args = ap.parse_args()

    data, seeds = load(args.csv)
    per_scene = scene_means(data)
    per_ds = dataset_means(per_scene)
    times = overall_times(per_scene)

    lmds = [l for l in LMD_ORDER if l in per_scene]

    print("=== seeds per (lmd, scene) ===")
    all_scenes = [s for d in DATASETS for s in SCENES_OF[d]]
    print(f"{'scene':<11}" + "".join(f"{l:>7}" for l in lmds))
    for s in all_scenes:
        print(f"{s:<11}" + "".join(f"{len(seeds[l].get(s, [])):>7}" for l in lmds))

    print("\n=== per-dataset means ===")
    for lmd in lmds:
        print(f"\n--- lmd={lmd}  comp={times[lmd][0]:.1f}s  decomp={times[lmd][1]:.1f}s ---")
        for ds in DATASETS:
            a = per_ds[lmd][ds]
            if "PSNR" not in a:
                print(f"  {ds:<16} (no data)")
                continue
            n, exp = a["PSNR"][3], a["PSNR"][4]
            miss = f"  MISSING: {','.join(a['_scenes_missing'])}" if a["_scenes_missing"] else ""
            print(f"  {ds:<16} PSNR {a['PSNR'][0]:6.2f}  SSIM {a['SSIM'][0]:.3f}  "
                  f"LPIPS {a['LPIPS'][0]:.3f}  {a['size_MB'][0]:6.1f} MB  "
                  f"[{n}/{exp} cenas]{miss}")

    payload = {
        "lmds": lmds,
        "per_scene": {l: {s: {m: list(v) for m, v in mm.items()}
                          for s, mm in per_scene[l].items()} for l in lmds},
        "per_dataset": {l: {d: {k: (list(v) if isinstance(v, tuple) else v)
                                for k, v in per_ds[l][d].items()} for d in DATASETS}
                        for l in lmds},
        "times": {l: list(times[l]) for l in lmds},
    }
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
