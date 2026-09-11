#!/usr/bin/env bash
# Figure 6: power spectral density of the incident radiation G over the test split, for
# the solver and for the best model of each family.
#
#   truth   FVDOM
#   pca     PCA-p512-softplus   500 PCA components per field
#   ae      AE-p512-softplus    latent 500 -> the same branch input as the PCA
#   tbd     TBD-N4-D128    best of the (N, D) grid, as PCA and AE are of theirs
#
# PCA and AE at the same p and the same branch width: what separates their two panels is
# the encoder, nothing else.
#
#   setsid nohup bash scripts/power_spectrum.sh > src/logs/psd.log 2>&1 &
#
# Cost: ~10 min, dominated by the 901 Transformer-Based DeepONet inferences.
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
