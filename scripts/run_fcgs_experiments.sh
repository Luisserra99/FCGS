#!/usr/bin/env bash
#
# Batch runner for the FCGS compression experiments.
#
# Runs every (lambda, scene, seed) combination end to end -- encode, decode,
# evaluate -- and leaves results.json / times.json per run in the same shape the
# C3DGS runs use, so c3dgs/scripts/extract_metrics.py aggregates both methods
# into a single CSV.
#
# The run is checkpointed: completed scene/seed combinations are recorded in
# <RESULTS_ROOT>/<METHOD_LABEL>/progress.tsv and skipped on a re-run, so an
# interrupted or partially failed batch can simply be started again.
#
#   ./scripts/run_fcgs_experiments.sh
#   SCENES_OVERRIDE="train truck" LMDS_OVERRIDE="1e-4" ./scripts/run_fcgs_experiments.sh
#   DRY_RUN=true ./scripts/run_fcgs_experiments.sh
#
# ============================================================================
# CONFIG
# ============================================================================

# --- paths -----------------------------------------------------------------
FCGS_DIR=${FCGS_DIR:-/d01/luis/FCGS}
MODELS_ROOT=${MODELS_ROOT:-/d01/luis/datasets/models}
IMAGES_ROOT=${IMAGES_ROOT:-/d01/luis/datasets/images}
RESULTS_ROOT=${RESULTS_ROOT:-/d01/luis/runs_fcgs}
CONDA_BASE=${CONDA_BASE:-/d01/luis/miniconda3}
CONDA_ENV=${CONDA_ENV:-FCGS}
# extract_metrics.py / stats_significance.py live with the C3DGS scripts and are
# method-agnostic, so the FCGS runs are aggregated with the very same tools.
C3DGS_SCRIPTS=${C3DGS_SCRIPTS:-/d01/luis/c3dgs/scripts}
# stats_significance.py needs scipy, which the FCGS env does not have.
ANALYSIS_PYTHON=${ANALYSIS_PYTHON:-/usr/bin/python3}

# --- the experiment grid ----------------------------------------------------
SCENES=(bicycle bonsai counter drjohnson flowers garden kitchen \
        playroom room stump train treehill truck)
SEEDS=(0 1 2 3 4)
LMDS=(1e-4 2e-4 4e-4 8e-4 16e-4)
ITERATION=${ITERATION:-30000}      # which trained point cloud to compress

# --- codec knobs ------------------------------------------------------------
NR=${NR:-3}                        # normalization radius
DETERM=${DETERM:-1}                # deterministic codec (see docs/atomic_statement.md)
PER_STEP_SIZE=${PER_STEP_SIZE:-1000000}          # gaussians per independent step
PER_STEP_FALLBACKS=(500000 250000)               # retried in order after a failure

# --- execution control ------------------------------------------------------
KEEP_PLY=${KEEP_PLY:-false}        # keep the decoded .ply (bicycle alone is ~1.4 GB)
RESUME=${RESUME:-true}             # skip runs already recorded as DONE
FORCE_RERUN=${FORCE_RERUN:-false}  # ignore checkpoints and redo everything
DRY_RUN=${DRY_RUN:-false}          # print the commands without running them
RUN_ANALYSIS=${RUN_ANALYSIS:-true} # aggregate to metrics.csv at the end
RUN_SIGNIFICANCE=${RUN_SIGNIFICANCE:-false}  # paired tests across the lambdas
BASELINE_LABEL=${BASELINE_LABEL:-FCGS_1e-4}  # only used by the significance step
# Where torch caches the VGG16 weights LPIPS needs; validated below.
TORCH_HOME=${TORCH_HOME:-$HOME/.cache/torch}

# Optional space-separated overrides, e.g. SCENES_OVERRIDE="train truck"
[ -n "${SCENES_OVERRIDE:-}" ] && read -r -a SCENES <<< "$SCENES_OVERRIDE"
[ -n "${SEEDS_OVERRIDE:-}" ]  && read -r -a SEEDS  <<< "$SEEDS_OVERRIDE"
[ -n "${LMDS_OVERRIDE:-}" ]   && read -r -a LMDS   <<< "$LMDS_OVERRIDE"
[ -n "${PER_STEP_FALLBACKS_OVERRIDE:-}" ] && read -r -a PER_STEP_FALLBACKS <<< "$PER_STEP_FALLBACKS_OVERRIDE"

# ============================================================================
# END CONFIG -- no edits needed below
# ============================================================================

# Deliberately no `set -e`: a single failing scene must not abort a batch that
# runs for a day. Failures are recorded and the loop moves on.
set -uo pipefail

# printf %g follows the locale, and this box is pt_BR: without this the lambda
# directory would come out as "0,0001" while python writes "0.0001".
export LC_ALL=C

SUMMARY="$RESULTS_ROOT/run_summary.log"
FAILURES="$RESULTS_ROOT/failures.log"

mkdir -p "$RESULTS_ROOT"
touch "$SUMMARY" "$FAILURES"

log() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$SUMMARY"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# --- environment ------------------------------------------------------------
[ -d "$FCGS_DIR" ] || die "FCGS_DIR not found: $FCGS_DIR"
[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ] || die "conda.sh not found under $CONDA_BASE"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV" || die "cannot activate conda env '$CONDA_ENV'"
PYTHON_BIN=$(command -v python) || die "no python in the '$CONDA_ENV' env"
log "using $PYTHON_BIN"

# The previous batch died with CUDA OOM on bicycle after three successful steps:
# the failure was fragmentation, not a genuine shortage.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}

# LPIPS pulls the VGG16 ImageNet weights through torch.hub on first use. If
# TORCH_HOME is unwritable the run dies *after* the decode, throwing away the
# timing measurement, so fall back to a writable cache instead.
if ! mkdir -p "$TORCH_HOME/hub/checkpoints" 2>/dev/null; then
  echo "WARNING: TORCH_HOME '$TORCH_HOME' is not writable; falling back to $HOME/.cache/torch" >&2
  TORCH_HOME=$HOME/.cache/torch
  mkdir -p "$TORCH_HOME/hub/checkpoints" || die "cannot create a torch cache at $TORCH_HOME"
fi
export TORCH_HOME
log "TORCH_HOME=$TORCH_HOME"

cd "$FCGS_DIR" || die "cannot cd to $FCGS_DIR"

# --- preflight --------------------------------------------------------------
# tmc3 (GPCC) is invoked through os.system inside model/gpcc_utils.py, where a
# failure surfaces only as an assertion halfway through the encode.
TMC3=$(grep -m1 -oP "gpcc_codec_path='\K[^']+" model/gpcc_utils.py)
[ -x "$TMC3" ] || die "tmc3 not executable: '$TMC3' (see model/gpcc_utils.py)"

missing=()
for lmd in "${LMDS[@]}"; do
  ckpt=$(printf './checkpoints/checkpoint_%g.pkl' "$lmd")
  [ -f "$ckpt" ] || missing+=("no checkpoint for lmd=$lmd ($ckpt)")
done
for scene in "${SCENES[@]}"; do
  ply="$MODELS_ROOT/$scene/point_cloud/iteration_$ITERATION/point_cloud.ply"
  [ -f "$MODELS_ROOT/$scene/cfg_args" ] || missing+=("$scene: no models/$scene/cfg_args")
  [ -f "$ply" ]                         || missing+=("$scene: no $ply")
  [ -d "$IMAGES_ROOT/$scene" ]          || missing+=("$scene: no images/$scene")
done
if [ ${#missing[@]} -gt 0 ]; then
  printf 'ERROR: %s\n' "${missing[@]}" >&2
  die "resolve the missing inputs above, or trim SCENES / LMDS"
fi

# Fail fast on a broken environment rather than after N doomed runs.
if ! "$PYTHON_BIN" -c "import bitstream_io, lpips; from model.FCGS_model import FCGS" >/dev/null 2>&1; then
  echo "ERROR: the FCGS modules do not import. Details:" >&2
  "$PYTHON_BIN" -c "import bitstream_io, lpips; from model.FCGS_model import FCGS" 2>&1 | tail -5 >&2
  die "fix the environment before launching a batch"
fi

total=$(( ${#LMDS[@]} * ${#SCENES[@]} * ${#SEEDS[@]} ))
log "=== FCGS batch: $total runs (${#LMDS[@]} lmd x ${#SCENES[@]} scenes x ${#SEEDS[@]} seeds) ==="
log "per_step_size=$PER_STEP_SIZE fallbacks=${PER_STEP_FALLBACKS[*]} determ=$DETERM nr=$NR keep_ply=$KEEP_PLY"

done_n=0 skip_n=0 fail_n=0 idx=0

for lmd in "${LMDS[@]}"; do
  METHOD_LABEL="FCGS_${lmd}"
  METHOD_DIR="$RESULTS_ROOT/$METHOD_LABEL"
  PROGRESS="$METHOD_DIR/progress.tsv"
  mkdir -p "$METHOD_DIR"
  touch "$PROGRESS"
  lmd_dirname=$(printf '%g' "$lmd")   # matches str(float(lmd)) on the python side

  log "--- lmd=$lmd -> $METHOD_DIR ---"

  for scene in "${SCENES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      idx=$((idx + 1))
      run_key="${scene}:${seed}"
      out="$METHOD_DIR/$scene/seed_$seed"
      bits="$out/bitstreams"
      ply="$out/decoded.ply"
      ply_src="$MODELS_ROOT/$scene/point_cloud/iteration_$ITERATION/point_cloud.ply"

      # A run counts as complete only if it was recorded DONE *and* left a
      # non-empty results.json, so a truncated run is never mistaken for one.
      marker=$(printf 'DONE\t%s\t' "$run_key")
      if [ "$FORCE_RERUN" != true ] && [ "$RESUME" = true ] \
         && grep -qF -- "$marker" "$PROGRESS" && [ -s "$out/results.json" ]; then
        skip_n=$((skip_n + 1))
        log "[$idx/$total] skip $METHOD_LABEL $run_key (done)"
        continue
      fi

      log "[$idx/$total] run $METHOD_LABEL $run_key -> $out"
      if [ "$DRY_RUN" = true ]; then
        printf '  %s encode_single_scene.py --lmd %s --seed %s --per_step_size %s --ply_path_from %q --bit_path_to %q --out_dir %q\n' \
          "$PYTHON_BIN" "$lmd" "$seed" "$PER_STEP_SIZE" "$ply_src" "$bits" "$out"
        printf '  %s decode_single_scene_validate.py --lmd %s --seed %s --bit_path_from %q --ply_path_to %q --out_dir %q --model_path %q --source_path %q\n' \
          "$PYTHON_BIN" "$lmd" "$seed" "$bits" "$ply" "$out" "$MODELS_ROOT/$scene" "$IMAGES_ROOT/$scene"
        continue
      fi

      # A previous attempt may have left partial steps behind; the decoder
      # counts step directories, so a stale one would silently corrupt the run.
      rm -rf "$bits" "$out/results.json" "$out/times.json" "$out/encode_meta.json"
      mkdir -p "$out"
      : > "$out/run.log"

      # --- encode, retrying with smaller steps if CUDA runs out of memory ----
      status=1
      used_step_size=""
      tried=""
      for pss in "$PER_STEP_SIZE" "${PER_STEP_FALLBACKS[@]}"; do
        # A fallback equal to a size already tried would just burn another
        # failed encode (e.g. PER_STEP_SIZE=500000 with the default fallbacks).
        case " $tried " in *" $pss "*) continue ;; esac
        tried="$tried $pss"
        echo "=== encode $run_key (per_step_size=$pss) ===" >> "$out/run.log"
        "$PYTHON_BIN" encode_single_scene.py \
          --lmd "$lmd" \
          --nr "$NR" \
          --determ "$DETERM" \
          --seed "$seed" \
          --per_step_size "$pss" \
          --ply_path_from "$ply_src" \
          --bit_path_to "$bits" \
          --out_dir "$out" 2>&1 | tr '\r' '\n' | tee -a "$out/run.log"
        status=${PIPESTATUS[0]}
        if [ "$status" -eq 0 ]; then
          used_step_size=$pss
          break
        fi
        log "[$idx/$total] encode failed at per_step_size=$pss (exit=$status), retrying smaller"
        rm -rf "$bits"
      done

      if [ "$status" -ne 0 ]; then
        ts=$(date -Is)
        printf 'FAIL\t%s\t%s\t%s\tencode\n' "$run_key" "$ts" "$out" >> "$PROGRESS"
        printf '%s\t%s\t%s\tencode exit=%s\t%s\n' "$ts" "$METHOD_LABEL" "$run_key" "$status" "$out/run.log" >> "$FAILURES"
        fail_n=$((fail_n + 1))
        log "[$idx/$total] FAIL $run_key (encode, exit=$status) -> $out/run.log"
        continue
      fi
      [ "$used_step_size" != "$PER_STEP_SIZE" ] && \
        log "[$idx/$total] note: $run_key encoded with per_step_size=$used_step_size"

      # --- decode + evaluate -------------------------------------------------
      echo "=== decode+eval $run_key ===" >> "$out/run.log"
      # --seed is passed explicitly even though encode_meta.json carries it:
      # a mismatch here must fail loudly, not decode a corrupted model.
      "$PYTHON_BIN" decode_single_scene_validate.py \
        --lmd "$lmd" \
        --nr "$NR" \
        --seed "$seed" \
        --bit_path_from "$bits" \
        --ply_path_to "$ply" \
        --out_dir "$out" \
        --model_path "$MODELS_ROOT/$scene" \
        --source_path "$IMAGES_ROOT/$scene" 2>&1 | tr '\r' '\n' | tee -a "$out/run.log"
      status=${PIPESTATUS[0]}

      [ "$KEEP_PLY" = true ] || rm -f "$ply"

      ts=$(date -Is)
      if [ "$status" -eq 0 ] && [ -s "$out/results.json" ]; then
        printf 'DONE\t%s\t%s\t%s\tpss=%s\n' "$run_key" "$ts" "$out" "$used_step_size" >> "$PROGRESS"
        done_n=$((done_n + 1))
        log "[$idx/$total] ok   $METHOD_LABEL $run_key"
      else
        printf 'FAIL\t%s\t%s\t%s\tdecode\n' "$run_key" "$ts" "$out" >> "$PROGRESS"
        printf '%s\t%s\t%s\tdecode exit=%s\t%s\n' "$ts" "$METHOD_LABEL" "$run_key" "$status" "$out/run.log" >> "$FAILURES"
        fail_n=$((fail_n + 1))
        log "[$idx/$total] FAIL $run_key (decode, exit=$status) -> $out/run.log"
      fi
    done
  done
done

log "=== FCGS batch done: $done_n ok, $skip_n skipped, $fail_n failed ==="
[ "$fail_n" -gt 0 ] && log "failures listed in $FAILURES; re-run this script to retry them"

# --- analysis ---------------------------------------------------------------
if [ "$RUN_ANALYSIS" != true ] || [ "$DRY_RUN" = true ]; then
  exit 0
fi

# Every method subdirectory holding results becomes one labelled root, so the
# aggregate covers all lambdas run so far, not just this invocation.
roots=() labels=()
for d in "$RESULTS_ROOT"/*/; do
  [ -d "$d" ] || continue
  [ -n "$(find "$d" -name results.json -print -quit 2>/dev/null)" ] || continue
  roots+=("$d")
  labels+=("$(basename "$d")")
done

if [ ${#roots[@]} -eq 0 ]; then
  log "no results.json found under $RESULTS_ROOT; skipping analysis"
  exit 0
fi

CSV="$RESULTS_ROOT/metrics.csv"
log "aggregating ${#roots[@]} configuration(s): ${labels[*]}"
"$PYTHON_BIN" "$C3DGS_SCRIPTS/extract_metrics.py" "${roots[@]}" \
  --labels "${labels[@]}" -o "$CSV" 2>&1 | tee -a "$SUMMARY"

if [ "$RUN_SIGNIFICANCE" = true ]; then
  if [ -x "$ANALYSIS_PYTHON" ] && printf '%s\n' "${labels[@]}" | grep -qx -- "$BASELINE_LABEL"; then
    "$ANALYSIS_PYTHON" "$C3DGS_SCRIPTS/stats_significance.py" "$CSV" \
      --baseline "$BASELINE_LABEL" \
      --latex "$RESULTS_ROOT/significance_table.tex" 2>&1 | tee -a "$SUMMARY"
  else
    log "skipping significance tests (need $ANALYSIS_PYTHON and baseline '$BASELINE_LABEL')"
  fi
fi

log "analysis written to $CSV"
