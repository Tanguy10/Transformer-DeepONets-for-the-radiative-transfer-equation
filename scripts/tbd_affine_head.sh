#!/usr/bin/env bash
# What the output softplus buys the Transformer-Based DeepONet.
#
# Same encoders, same operator budget as Table II, one change: the head. The runs below
# predict I = D_I(z, y) * I_scale instead of softplus(D_I(z, y)) * I_scale, so I may go
# negative. The reconstructor stays a cross-attention -- NOT the affine R of Thm 3.4 --
# hence these runs keep their dagger in the bound figure; the question they answer is
# whether TBD-N1-D128 still goes under the bound (0.80 train / 1.03 test) once the
# monotone output map that already lets a branch.trunk model cross it is removed.
#
# The encoder is NOT refitted: `<run>-affine` reads encoder_tbd_<run>.pt, exactly as the
# two heads of the PCA and AE runs share theirs.
#
#   setsid nohup bash scripts/tbd_affine_head.sh > src/logs/tbd_affine.log 2>&1 &
#
# TBD_GRID picks the configurations (default: the one that crosses the bound).
# Operator cost, measured on the softplus campaign: N4-D32 6.5 h, N2-D64 7.8 h,
# N1-D128 10.7 h, N2-D128 10.7 h, N4-D128 10.9 h.
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
