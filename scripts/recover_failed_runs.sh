#!/usr/bin/env bash
#
# Replay decode + evaluation for runs whose encode succeeded but whose
# evaluation died (typically "Unable to find a valid cuDNN algorithm to run
# convolution": LPIPS could not get a workspace because the decode's working
# set was still resident on the GPU).
#
# The bitstreams are already on disk, so nothing is re-encoded -- this is far
# cheaper than re-running those combinations from scratch.
#
#   ./scripts/recover_failed_runs.sh
#   DRY_RUN=true ./scripts/recover_failed_runs.sh
#
set -uo pipefail
export LC_ALL=C

FCGS_DIR=${FCGS_DIR:-/d01/luis/FCGS}
MODELS_ROOT=${MODELS_ROOT:-/d01/luis/datasets/models}
IMAGES_ROOT=${IMAGES_ROOT:-/d01/luis/datasets/images}
RESULTS_ROOT=${RESULTS_ROOT:-/d01/luis/runs_fcgs}
CONDA_BASE=${CONDA_BASE:-/d01/luis/miniconda3}
CONDA_ENV=${CONDA_ENV:-FCGS}
C3DGS_SCRIPTS=${C3DGS_SCRIPTS:-/d01/luis/c3dgs/scripts}
NR=${NR:-3}
KEEP_PLY=${KEEP_PLY:-false}
DRY_RUN=${DRY_RUN:-false}
TORCH_HOME=${TORCH_HOME:-$HOME/.cache/torch}

SUMMARY="$RESULTS_ROOT/recovery.log"
touch "$SUMMARY"
log() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$SUMMARY"; }

# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV" || { echo "cannot activate $CONDA_ENV" >&2; exit 1; }
PYTHON_BIN=$(command -v python)
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}
mkdir -p "$TORCH_HOME/hub/checkpoints" 2>/dev/null || TORCH_HOME=$HOME/.cache/torch
export TORCH_HOME
cd "$FCGS_DIR" || exit 1

# Collect every run that has an encode but no evaluation.
pending=()
for meta in "$RESULTS_ROOT"/FCGS_*/*/seed_*/encode_meta.json; do
  [ -e "$meta" ] || continue
  out=$(dirname "$meta")
  [ -s "$out/results.json" ] && continue
  [ -d "$out/bitstreams" ] || continue
  pending+=("$out")
done

total=${#pending[@]}
log "=== recovery: $total run(s) with an encode but no results.json ==="
[ "$total" -eq 0 ] && exit 0

ok=0 fail=0 idx=0
for out in "${pending[@]}"; do
  idx=$((idx + 1))
  seed_dir=$(basename "$out")          # seed_N
  scene=$(basename "$(dirname "$out")")
  method=$(basename "$(dirname "$(dirname "$out")")")   # FCGS_<lmd>
  lmd=${method#FCGS_}
  seed=${seed_dir#seed_}
  progress="$RESULTS_ROOT/$method/progress.tsv"

  log "[$idx/$total] $method $scene:$seed"
  if [ "$DRY_RUN" = true ]; then continue; fi

  echo "=== recovery decode+eval $scene:$seed ===" >> "$out/run.log"
  "$PYTHON_BIN" decode_single_scene_validate.py \
    --lmd "$lmd" \
    --nr "$NR" \
    --seed "$seed" \
    --bit_path_from "$out/bitstreams" \
    --ply_path_to "$([ "$KEEP_PLY" = true ] && echo "$out/decoded.ply" || echo "")" \
    --out_dir "$out" \
    --model_path "$MODELS_ROOT/$scene" \
    --source_path "$IMAGES_ROOT/$scene" 2>&1 | tr '\r' '\n' | tee -a "$out/run.log" \
    | grep -E "Evaluation results|Error|Traceback" || true
  status=${PIPESTATUS[0]}

  ts=$(date -Is)
  if [ "$status" -eq 0 ] && [ -s "$out/results.json" ]; then
    printf 'DONE\t%s:%s\t%s\t%s\trecovered\n' "$scene" "$seed" "$ts" "$out" >> "$progress"
    ok=$((ok + 1))
    log "[$idx/$total] ok   $method $scene:$seed"
  else
    fail=$((fail + 1))
    log "[$idx/$total] FAIL $method $scene:$seed (exit=$status)"
  fi
done

log "=== recovery done: $ok recovered, $fail still failing ==="

roots=() labels=()
for d in "$RESULTS_ROOT"/FCGS_*/; do
  [ -n "$(find "$d" -name results.json -print -quit 2>/dev/null)" ] || continue
  roots+=("$d"); labels+=("$(basename "$d")")
done
"$PYTHON_BIN" "$C3DGS_SCRIPTS/extract_metrics.py" "${roots[@]}" \
  --labels "${labels[@]}" -o "$RESULTS_ROOT/metrics.csv" 2>&1 | tail -3 | tee -a "$SUMMARY"
