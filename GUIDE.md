# FCGS Compression Improvements — What Was Done & How to Reproduce

Date: 2026-06-09. Branch: `FCGS_testes` (local changes on top of commit `7ad658d`).

## 0. Summary

Seven optimization proposals were evaluated for this inference-only repo. Four require retraining
(geo hyperprior shrink, slice-ratio change, learned cold-start constants, quaternion 4→3+sign) and
were **rated not applicable** — there is no training code and only pre-trained checkpoints.
The three applicable ones were implemented:

| # | Change | Type | Measured effect |
|---|--------|------|-----------------|
| 1 | Hierarchical context coder for the MEM mask (`mask_ctx.b`) | lossless | −0.3% … −4.1% of mask bits (scene/λ dependent); decoded PLY byte-identical |
| 2a | G-PCC flag tuning: `cabac_bypass_stream_enabled_flag=0` | lossless | −0.6% of xyz bits, same speed |
| 2b | G-PCC xyz voxelization bit depth `--xyz_bits {16,15,14}` | lossy (gated) | 15-bit: −23% xyz bits, 14-bit: −45% xyz bits; failed the ≤0.05 dB gate on large scenes → **kept at 16 by default**, 15 usable per-scene |
| 3 | `.fcgs` single-file container | packaging | 37+ files/step → 1 file/scene; no size change in reported bits |

Also rejected **empirically**: G-PCC predictive geometry (`--geomTreeType=1`) is 18–30% *worse*
and ~6× slower to encode on 3DGS point distributions; planar mode and larger intra prediction are
size-neutral. (See sweep table below.)

Encode/decode speed is unchanged (mask coder adds 2–6 small fully-parallel CUDA range-coder passes;
encode time on `room` λ=1e-4: 17.3 s → 17.4 s).

## 1. What changed, file by file

### `model/encodings_cuda.py`
New hierarchical binary context coder (appended after the legacy `encoder`/`decoder` pair):
- `_bernoulli_encode_bytes` / `_bernoulli_decode_bytes` — arithmetic coding with a **per-element**
  probability tensor (the legacy mask coder only supported one global probability).
- `_mask_ctx_ids` — context id (left·2+right) from the two already-decoded neighbours.
- `encoder_mask_ctx` / `decoder_mask_ctx` — the mask is kept in **Morton (scan) order** and coded
  in `L` dyadic levels: level 0 = every 2^(L−1)-th bit with a global Bernoulli; each next level
  codes the midpoints conditioned on their two coded neighbours (4 contexts, probabilities fitted
  on the data at encode time, stored in the 42-byte header). `L ∈ {1..6}` is chosen at encode time
  by entropy estimate, so incompressible masks pay ≈0 overhead. Every level is one fully-parallel
  GPU `arithmetic_encode`/`decode` call — this is the trick that makes spatial context compatible
  with the CUDA range coder, which needs all per-element CDFs *before* decoding.

### `model/FCGS_model.py`
- `compress(..., mask_mode='ctx', xyz_bits=16)`: the mask is now computed **before** the seed-1
  shuffle (the `Encoder_mask` MLP is per-row, so values are order-independent — verified bitwise),
  encoded in Morton order via `encoder_mask_ctx`, then shuffled along with xyz/features.
  `mask_mode='global'` reproduces the original behaviour/bitstream exactly.
- `decomprss()`: auto-detects `mask_ctx.b` (new format) vs `mask.b` (legacy) by file existence.
  New masks are decoded right after G-PCC xyz decode (still in Morton order), then shuffled.
  **Old bitstreams keep decoding through the untouched legacy branch.**

### `model/gpcc_utils.py`
- `GPCC_CODEC_PATH`: resolves `tmc3` from `$TMC3_PATH`, then `$PATH`, then the in-repo build
  (`mpeg-pcc-tmc13/build/tmc3/tmc3`) — no more hard dependency on PATH setup.
- `GPCC_ENC_FLAGS_DEFAULT`: extracted the hard-coded flags; new default uses
  `cabac_bypass_stream_enabled_flag=0` (measured −0.6%, same speed).
- `voxelize`/`devoxelize`/`sorted_orig_voxels` take a `scale_factor` (bit depth) argument.
- `compress_gaussian_params(..., xyz_bits=16)`: with `xyz_bits != 16` the `xyz_gpcc.bin` gets a
  self-describing header (`FCGSXYZ1` magic + bit depth); 16-bit streams keep the legacy headerless
  format so older decoders still read them. `decompress_gaussian_params` sniffs the magic.

### `encode_single_scene.py`
New flags: `--mask_mode {ctx,global}` (default ctx), `--xyz_bits {16,15,14}` (default 16),
`--pack` / `--pack_rm` (single-file container), `--per_step_size` (lower it to 500000 if a large
scene at high λ runs out of GPU memory). Also prints per-stream sizes (xyz/mask/fea/feq/geo).

### `decode_single_scene.py`, `decode_single_scene_validate.py`
If `{bit_path_from}/{lmd}.fcgs` exists it is unpacked to a temp dir transparently. No flags needed.

### New tools
- `tools/test_mask_ctx.py` — 35-case exact round-trip unit test of the mask coder.
- `tools/gpcc_sweep.py` — G-PCC config sweep with lossless verification per config.
- `tools/fcgs_container.py` — `pack`/`unpack` for the `.fcgs` container.

## 2. Step-by-step reproduction

All commands run from `/d01/luis/FCGS` with the `FCGS` conda env
(`/d01/luis/miniconda3/envs/FCGS/bin/python`).

### Step A — mask coder unit test (no GPU scene data needed)
```bash
python tools/test_mask_ctx.py
# expect: 35/35 passed; ~90% saving on synthetic correlated masks, ~0% overhead on iid masks
```

### Step B — encode/decode one scene, prove losslessness
```bash
# legacy-format encode (baseline for comparison)
python encode_single_scene.py --lmd 1e-4 --determ 1 --mask_mode global \
  --ply_path_from /d01/luis/datasets/models/room/point_cloud/iteration_30000/point_cloud.ply \
  --bit_path_to /d01/luis/compress_FCGS_global_1/room
python decode_single_scene.py --lmd 1e-4 \
  --bit_path_from /d01/luis/compress_FCGS_global_1/room \
  --ply_path_to /d01/luis/compress_FCGS_global_1/room/decoded.ply

# new context-coded encode
python encode_single_scene.py --lmd 1e-4 --determ 1 --mask_mode ctx \
  --ply_path_from /d01/luis/datasets/models/room/point_cloud/iteration_30000/point_cloud.ply \
  --bit_path_to /d01/luis/compress_FCGS_maskctx_1/room
python decode_single_scene.py --lmd 1e-4 \
  --bit_path_from /d01/luis/compress_FCGS_maskctx_1/room \
  --ply_path_to /d01/luis/compress_FCGS_maskctx_1/room/decoded.ply

cmp /d01/luis/compress_FCGS_global_1/room/decoded.ply \
    /d01/luis/compress_FCGS_maskctx_1/room/decoded.ply && echo BYTE-IDENTICAL
```
Verified byte-identical for room/train/bicycle × λ∈{1e-4, 16e-4} (6/6 pairs).
Note: `bicycle` at λ=16e-4 needs `--per_step_size 500000` on a 24 GB GPU (pre-existing limit,
not introduced by these changes — the original script needs it too).

### Step C — G-PCC sweep (why the new default flags)
```bash
python tools/gpcc_sweep.py --ply_path /d01/luis/datasets/models/room/point_cloud/iteration_30000/point_cloud.ply
python tools/gpcc_sweep.py --ply_path /d01/luis/datasets/models/train/point_cloud/iteration_30000/point_cloud.ply --bit_depths 16 15 14
```
Result (consistent on both scenes): `cabac_bypass=0` −0.6% lossless; planar/intra6 neutral;
predictive geometry +18…31% worse and ~6× slower encode. 15-bit voxels −23% xyz size,
14-bit −45% (lossy — see gate below).

### Step D — xyz bit-depth quality gate
```bash
python encode_single_scene.py --lmd 1e-4 --determ 1 --xyz_bits 15 \
  --ply_path_from /d01/luis/datasets/models/room/point_cloud/iteration_30000/point_cloud.ply \
  --bit_path_to /d01/luis/compress_FCGS_xyz15_1/room
python decode_single_scene_validate.py --lmd 1e-4 \
  --bit_path_from /d01/luis/compress_FCGS_xyz15_1/room \
  --ply_path_to /d01/luis/compress_FCGS_xyz15_1/room/decoded.ply \
  --source_path /d01/luis/datasets/images/room
```
(Repeat with `--xyz_bits 14` and at `--lmd 16e-4`; compare PSNR/SSIM/LPIPS against the 16-bit run.)

Measured on `room` (gate: adopt only if ΔPSNR ≤ 0.05 dB):

| λ | bits | total MB | PSNR | ΔPSNR | verdict |
|---|------|----------|------|-------|---------|
| 1e-4 | 16 | 26.47 | 31.5095 | — | reference |
| 1e-4 | 15 | 25.88 (−2.2%) | 31.4871 | −0.022 | **pass** |
| 1e-4 | 14 | 25.31 (−4.4%) | 31.3905 | −0.119 | fail |
| 16e-4 | 16 | 15.18 | 31.0456 | — | reference |
| 16e-4 | 15 | 14.62 (−3.7%) | 31.0252 | −0.020 | **pass** |
| 16e-4 | 14 | 14.05 (−7.4%) | 30.9399 | −0.106 | fail |

On `train` (large outdoor scene) 15-bit FAILS the gate: 21.7418 → 21.6066 (Δ −0.135) at λ=1e-4
and 21.6275 → 21.4903 (Δ −0.137) at λ=16e-4. The voxel size scales with the scene bounding box,
so a fixed bit budget that is harmless on a room-sized scene is too coarse on a large one.

→ **Verdict: not adopted as default — scene-dependent.** `--xyz_bits 15` is a worthwhile manual
option for compact scenes (−2…−5% total at −0.02 dB on `room`), but the benchmark and the
defaults stay at 16-bit (strictly lossless).

### Step E — single-file container
```bash
python encode_single_scene.py ... --pack          # writes {bit_path_to}/{lmd}.fcgs as well
python tools/fcgs_container.py pack   <dir>/0.0001 <dir>/0.0001.fcgs   # or pack manually
python decode_single_scene.py --lmd 1e-4 --bit_path_from <dir_with_fcgs> --ply_path_to out.ply
```
The decoders auto-detect the container. Verified: container decode is byte-identical to
loose-file decode (204 files → one 27.8 MB file for room λ=1e-4).

### Step F — full benchmark
```bash
python /d01/luis/bench_maskctx/run_matrix.py \
  --dest_base /d01/luis/compress_FCGS_new \
  --scenes bicycle bonsai counter drjohnson flowers garden kitchen playroom room stump train treehill truck \
  --lmds 1e-4 2e-4 4e-4 8e-4 16e-4
```
Writes `compression_metrics.json` per scene (same schema as the old baselines, so
`extract_metrics.py` and the plotting scripts keep working) plus a flat `summary.csv`.

**Full-suite result (13 scenes × 5 λ, 65/65 runs OK, June 2026, RTX 3090) vs the May
`compress_FCGS_default_*` baselines** (62 runs comparable — 3 baseline JSONs lack metrics);
full data: `/d01/luis/bench_maskctx/full_suite.csv`, comparison: `compare_baselines.py`:

| λ | Δ size (avg) | Δ PSNR (avg) |
|---|---|---|
| 1e-4 | −1.01% | +0.074 dB |
| 2e-4 | −1.13% | +0.089 dB |
| 4e-4 | −1.26% | +0.120 dB |
| 8e-4 | −1.50% | +0.103 dB |
| 16e-4 | −1.39% | +0.227 dB |
| **overall** | **−1.26%** | **+0.123 dB** |

No scene/λ regressed by more than 0.02 dB (run-to-run render noise); several improved
markedly (train/truck/playroom +0.2…0.6 dB, room@16e-4 +1.8 dB) because the May baselines
were encoded with ~500k-Gaussian steps while the committed default of 1M gives the context
models larger blocks — bigger steps are both smaller *and* better. The size reduction
decomposes into the mask context coder, the G-PCC cabac flag, and the step-size effect.

## 3. Important operational notes

- **Old bitstreams from May 2026 (`compress_FCGS_default_*`) cannot be decoded** — by either the
  original or the modified code. They were produced by a since-modified script (the bicycle dir
  has 13 step dirs ⇒ ~500k per step, while the committed script uses 1M; chunk-size mismatch makes
  the arithmetic decoder crash). This predates these changes; the metrics in their
  `compression_metrics.json` are still valid as reference numbers.
- The mask change is **format-breaking forward** (new `mask_ctx.b` file), but the decoder
  auto-detects and old streams use the untouched legacy path. To produce legacy-format streams:
  `--mask_mode global`.
- `--xyz_bits 16` keeps `xyz_gpcc.bin` bit-exactly in the legacy format.
