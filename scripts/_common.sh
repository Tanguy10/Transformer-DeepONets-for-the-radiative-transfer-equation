set -u
SCRIPTS=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(dirname "$SCRIPTS")
SRC="$ROOT/src"
CKPT="$SRC/checkpoints"
LOGD="$SRC/logs"
PY=${PY:-python}
export PYTHONPATH="$SRC"
export PYTHONUNBUFFERED=1
cd "$ROOT" || exit 1

log(){ echo "[$(basename "$0" .sh) $(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run(){   # run <model> <tag> <command...>
  local M="$1" TAG="$2"; shift 2
  mkdir -p "$LOGD/$M"
  if [ -f "$LOGD/$M/.done_$TAG" ]; then log "SKIP  $M/$TAG"; return 0; fi
  local t0=$SECONDS
  log "START $M/$TAG"
  "$@" > "$LOGD/$M/$TAG.log" 2>&1
  local rc=$?
  local dt=$(( (SECONDS - t0) / 60 ))
  if [ $rc -eq 0 ]; then
    touch "$LOGD/$M/.done_$TAG"; log "END   $M/$TAG (OK, $dt min)"
  else
    log "END   $M/$TAG (FAILED rc=$rc, $dt min) -- see $LOGD/$M/$TAG.log"
  fi
  return $rc
}

# Ignore processes in D state: stuck on an automount, they hold GPU memory forever.
gpu_busy(){
  local p st out=""
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    st=$(ps -p "$p" -o stat= 2>/dev/null | tr -d ' ')
    case "$st" in ""|D*) continue ;; esac
    out="$out $p"
  done
  echo "$out"
}

wait_for_gpu(){
  while [ -n "$(gpu_busy)" ]; do
    log "GPU busy (pid$(gpu_busy) ) -- waiting 300 s"
    sleep 300
  done
}

# Per-snapshot relative L2 on the test split, read back from the metrics files.
report(){
  printf '\n%-24s %8s %8s\n' "model" "I-L2" "G-L2"
  local M F I G
  for M in "$@"; do
    F="$SRC/results/$M/${M}_metrics_test.txt"
    [ -f "$F" ] || { printf '%-24s %8s %8s\n' "$M" "--" "--"; continue; }
    I=$(awk '/^  I  L2/ {print $4}' "$F"); G=$(awk '/^  G  L2/ {print $4}' "$F")
    printf '%-24s %8s %8s\n' "$M" "${I:-—}" "${G:-—}"
  done
}
