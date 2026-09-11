#!/usr/bin/env bash
. "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

DOT="$SRC/dot_decoder/dot_deeponet.py"
ENC=${ENC:-TBD-N1-D128}
EPOCHS=${EPOCHS:-2000}

[ -f "$CKPT/encoder_tbd_${ENC}.pt" ] || {
  log "MISSING encoder_tbd_${ENC}.pt -- run scripts/train_paper_models.sh first"; exit 1; }

wait_for_gpu
log "=== JUNCTION ABLATION on the encoder of $ENC, $EPOCHS epochs ==="

train(){   # train <name> <extra flags...>
  local M="$1"; shift
  run "$M" "1_train" $PY "$DOT" --mode train --name "$M" \
      --pretrained_encoder "$ENC" --op_epochs $EPOCHS "$@"
  run "$M" "2_eval"  $PY "$DOT" --mode eval  --name "$M"
}

train abl-dot
train abl-dot-notrunk       --trunk_hidden
train abl-dot-branch        --branch_hidden 512 512 512 512
train abl-dot-branch-raw    --branch_hidden 512 512 512 512 --no-rff
train abl-dot-branch-affine --branch_hidden 512 512 512 512 --affine

log "=== DONE -- the cross-attention reference is TBD-N1-D128 (Table II) ==="
for M in TBD-N1-D128 abl-dot abl-dot-notrunk abl-dot-branch abl-dot-branch-raw \
         abl-dot-branch-affine; do
  grep -H "TEST .*I-L2" "$LOGD/$M/2_eval.log" 2>/dev/null || true
done
