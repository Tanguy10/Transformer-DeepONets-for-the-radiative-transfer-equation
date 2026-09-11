#!/usr/bin/env bash
# Multi-resolution campaign -- Table III and Figs. 8-11 of the paper.
#
# Five models trained on the SAME mixed-resolution dataset (a third of the snapshots at
# 150^2, a third at 75^2, a third at 25^2), then tested with the input pair supplied at
# each resolution in turn, the prediction always requested on the 150^2 solver mesh:
#
#   res-TBD       Transformer-Based DeepONet, native ingestion at any resolution
#   res-PCA-150   PCA-DeepONet, every input resampled up to the 150^2 solver mesh
#   res-PCA-50    PCA-DeepONet, everything brought down to a 50^2 working grid
#   res-AE-150    AE-DeepONet, same two working grids
#   res-AE-50
#
# The four fixed-dimension models carry the AFFINE head (`--affine`, no softplus), so
# their reconstructor is literally the R of Theorem 3.4 -- the class the paper compares
# the attention decoder against.
#
# Derived datasets, rebuilt on first use next to the main one:
#   mixed_res_dataset_phi.pt     native, grouped 150/75/25   -> Transformer-Based DeepONet
#   interp150_dataset_phi.pt     everything upsampled to 150 -> PCA/AE @150
#   fixedres50_dataset_phi.pt    everything at 50            -> PCA/AE @50
#
# 70/15/15 split (seed 42, same as the parent repo): trained on the 4200 train
# snapshots, best checkpoint picked on the 900 CLEAN val ones, everything REPORTED
# measured on the 901 test ones.
#
#   setsid nohup bash src/robustness_resolution/run_campaign.sh \
#       > src/robustness_resolution/logs/campaign.log 2>&1 &
#
# Every model is trained in TWO stages, the settings of Appendix A: the encoder is first
# pretrained on the reconstruction of (T, kappa) -- on the mixed data, at the model's own
# working resolution -- then the operator is trained with that encoder FINE-TUNED at
# 1e-4 (a PCA has no weights, so it stays as fitted). Pass --no-finetune_encoder to
# train the reconstructor alone.
#
# Restart: one .done_<tag> marker per step, just relaunch the same file after a cut.
# Do NOT edit it while it runs (bash re-reads it by offset).
# Expected wall time on a single GPU: the encoder stages add ~1000 epochs each, and
# fine-tuning puts the encoder back inside the graph, so count roughly twice the frozen
# campaign -- of the order of a day and a half.
# Override the interpreter with PY=..., or restrict the grid with STEPS="pca150 ...".

set -u
SUB=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)   # .../src/robustness_resolution
SRC=$(dirname "$SUB")
ROOT=$(dirname "$SRC")
PY=${PY:-python}
LOGD="$SUB/logs"
export PYTHONPATH="$SRC"
export PYTHONUNBUFFERED=1
mkdir -p "$LOGD"
cd "$ROOT" || exit 1

log(){ echo "[res $(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run(){   # run <tag> <command...>
  local TAG="$1"; shift
  if [ -f "$LOGD/.done_$TAG" ]; then log "SKIP  $TAG"; return 0; fi
  local t0=$SECONDS
  log "START $TAG"
  "$@" > "$LOGD/$TAG.log" 2>&1
  local rc=$?
  local dt=$(( (SECONDS - t0) / 60 ))
  if [ $rc -eq 0 ]; then
    touch "$LOGD/.done_$TAG"; log "END   $TAG (OK, $dt min)"
  else
    log "END   $TAG (FAILED rc=$rc, $dt min) -- see $LOGD/$TAG.log"
  fi
  return $rc
}

# The GPU is compute-bound: two concurrent runs each go twice as slow, so wait for it
# to free up -- but IGNORE processes stuck in UNINTERRUPTIBLE disk sleep (state D). A
# process frozen on the solver-H5 automount (cf. utils.solver_h5) keeps its GPU memory
# reserved without ever resuming: counting it as a tenant would block the campaign
# forever. It burns no SM, so leaving it there only costs a few hundred MB.
gpu_busy(){
  local p st out=""
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
    st=$(ps -p "$p" -o stat= 2>/dev/null | tr -d ' ')
    case "$st" in ""|D*) continue ;; esac
    out="$out $p"
  done
  echo "$out"
}

while [ -n "$(gpu_busy)" ]; do
  log "GPU busy (pid$(gpu_busy) ) -- waiting 120 s"
  sleep 120
done

log "=== MULTI-RESOLUTION CAMPAIGN -- query (x,y,phi), rel-L2 loss ==="

# ---------------------------------------------------------------------------
#  Trainings -- cheapest first (they validate the chain in 2 h)
# ---------------------------------------------------------------------------
run "pca150" $PY "$SUB/src/train.py" --model pca --affine
run "ae150"  $PY "$SUB/src/train.py" --model ae  --affine
run "pca50"  $PY "$SUB/src/train.py" --model pca --affine --input_res 50
run "ae50"   $PY "$SUB/src/train.py" --model ae  --affine --input_res 50
run "tbd"    $PY "$SUB/src/train.py" --model tbd

MODELS=(--model tbd:res-TBD --model pca:res-PCA-150 --model ae:res-AE-150
        --model pca:res-PCA-50:50 --model ae:res-AE-50:50)

# ---------------------------------------------------------------------------
#  Evaluations -- on TEST, the input pair supplied at each resolution in turn
# ---------------------------------------------------------------------------
# Table III + the per-resolution snapshots of Figs. 9-11
run "eval_mixed" $PY "$SUB/src/evaluate.py" --split test --test mixed --tag mixed \
                     "${MODELS[@]}"
# Sanity check: the same models on the clean 150^2 test block
run "eval_clean" $PY "$SUB/src/evaluate.py" --split test "${MODELS[@]}" --snapshots 2

# The input fields themselves (T and kappa, one row each) at the three resolutions: Fig. 8
run "fields_fig" $PY "$SUB/src/evaluate.py" --fields 3155

log "=== DONE ==="
cat "$LOGD/eval_mixed.log"
