#!/usr/bin/env bash
# Appendix C: what the cross-attention reconstructor buys, isolated.
#
# Every run below keeps the Transformer-Based DeepONet ENCODER of TBD-N1-D128 and
# replaces its cross-attention decoder by the branch.trunk junction of a vanilla
# DeepONet, I = act(<z(u), t(y)> + b) * I_scale, where z(u) is the flattened latent.
# Only the junction changes, so the difference with TBD-N1-D128 (Table II) is the
# reconstructor and nothing else.
#
#   abl-dot                  <z, t(y)> + b, trunk MLP on random Fourier features
#   abl-dot-notrunk          no trunk at all: t(y) IS the RFF projection
#   abl-dot-branch           a branch MLP A: z -> beta first (the full vanilla junction)
#   abl-dot-branch-raw       same, trunk MLP on the raw (x, y, phi) instead of the RFF
#   abl-dot-branch-affine    same, affine head -- literally the R of Theorem 3.4
#
#   setsid nohup bash scripts/ablation_junction.sh > src/logs/ablation.log 2>&1 &
#
# Cost: five 2000-epoch operator runs on a frozen-then-fine-tuned encoder, ~4 h each.
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
