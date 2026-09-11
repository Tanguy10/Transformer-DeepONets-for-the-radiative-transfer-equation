#!/usr/bin/env bash
. "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

EPOCHS_OP=${EPOCHS_OP:-2000}
TBD_GRID=${TBD_GRID:-"1,128"}
MODELS=""

wait_for_gpu
log "=== TBD AFFINE HEAD (no output softplus), $EPOCHS_OP operator epochs ==="

for ND in $TBD_GRID; do
  N=${ND%,*}; D=${ND#*,}
  BASE="TBD-N${N}-D${D}"
  M="${BASE}-affine"
  [ -f "$CKPT/encoder_tbd_${BASE}.pt" ] || {
    log "MISSING encoder_tbd_${BASE}.pt -- run scripts/train_paper_models.sh first"; continue; }
  run "$M" "1_train" \
      $PY src/train.py --model tbd --name "$M" --stage operator \
         --epochs $EPOCHS_OP --affine
  run "$M" "2_eval" $PY src/evaluate.py --name "$M"
  MODELS="$MODELS $M"
done

log "=== DONE -- the softplus reference is TBD-N* (Table II) ==="
report $MODELS
