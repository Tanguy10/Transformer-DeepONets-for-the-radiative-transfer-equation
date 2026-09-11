#!/usr/bin/env bash
set -u
SUB=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
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

# Ignore processes in D state: stuck on an automount, they hold GPU memory forever.
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

run "pca150" $PY "$SUB/src/train.py" --model pca --affine
run "ae150"  $PY "$SUB/src/train.py" --model ae  --affine
run "pca50"  $PY "$SUB/src/train.py" --model pca --affine --input_res 50
run "ae50"   $PY "$SUB/src/train.py" --model ae  --affine --input_res 50
run "tbd"    $PY "$SUB/src/train.py" --model tbd

MODELS=(--model tbd:res-TBD --model pca:res-PCA-150 --model ae:res-AE-150
        --model pca:res-PCA-50:50 --model ae:res-AE-50:50)

run "eval_mixed" $PY "$SUB/src/evaluate.py" --split test --test mixed --tag mixed \
                     "${MODELS[@]}"
run "eval_clean" $PY "$SUB/src/evaluate.py" --split test "${MODELS[@]}" --snapshots 2

run "fields_fig" $PY "$SUB/src/evaluate.py" --fields 3155

log "=== DONE ==="
cat "$LOGD/eval_mixed.log"
