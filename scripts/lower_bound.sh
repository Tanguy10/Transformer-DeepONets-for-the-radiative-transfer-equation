#!/usr/bin/env bash
. "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

REFRESH=${REFRESH:-none}      # none | spectrum | errors | all
MODELS=${MODELS:-"PCA-p128-affine PCA-p256-affine PCA-p512-affine AE-p128-affine AE-p256-affine AE-p512-affine TBD-N4-D32 TBD-N2-D64 TBD-N1-D128 TBD-N2-D128 TBD-N4-D128"}

wait_for_gpu
log "=== LOWER BOUND (Thm 3.4) vs the measured errors ==="

$PY src/lower_bound/run.py --list

run "lower_bound" "1_run" \
    $PY src/lower_bound/run.py --refresh "$REFRESH" --models $MODELS \
       --out "$SRC/lower_bound/bound_comparison.pdf"

log "=== DONE ==="
cat "$LOGD/lower_bound/1_run.log"
