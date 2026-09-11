#!/usr/bin/env bash
. "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

SPLIT=${SPLIT:-test}
FIGURES=${FIGURES:-all}

for F in "$CKPT/PCA-p512-softplus.pth" "$CKPT/encoder_pca_PCA-p512.pkl" \
         "$CKPT/AE-p512-softplus.pth" "$CKPT/operator_tbd_TBD-N4-D128.pt"; do
  [ -f "$F" ] || { log "MISSING $F -- run scripts/train_paper_models.sh first"; exit 1; }
done

wait_for_gpu
log "=== PSD, split $SPLIT: truth vs PCA-p512-softplus / AE-p512-softplus / TBD-N4-D128 ==="

run "psd" "1_run" \
    $PY src/power_spectrum/run.py --split "$SPLIT" --figures "$FIGURES" \
       --refresh pca,ae,tbd

log "=== DONE ==="
grep -E "G-L2" "$LOGD/psd/1_run.log" 2>/dev/null
ls -l "$SRC/power_spectrum/results"
