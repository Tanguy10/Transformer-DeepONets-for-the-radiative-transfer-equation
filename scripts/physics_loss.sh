#!/usr/bin/env bash
# Appendix B: the RTE residual as a REGULARISER on top of the data loss, for the three
# families -- everything else identical to the data-only runs of
# scripts/train_paper_models.sh, so w_phys is the ONLY thing that changes.
#
#   PCA-p128-softplus-phys   PCA-DeepONet, p = 128
#   AE-p128-softplus-phys    AE-DeepONet, p = 128, encoder fine-tuned at 1e-4
#   TBD-N4-D128-phys         Transformer-Based DeepONet, N = 4, D = 128
#
# The ENCODERS ARE NOT RETRAINED: each run starts from the encoder of its data-only
# twin (copied for pca/ae, --pretrained_encoder for the Transformer-Based DeepONet).
# Two reasons -- it saves three 1000-epoch pretrainings, and it keeps the comparison
# clean, since a refit encoder would be a second difference on top of w_phys.
#
# Collocation: 50000 points RESAMPLED every epoch for the DeepONets (a fixed pool would
# let the residual be memorised rather than regularised), 64 snapshots per physics step;
# 2048 points per step for the Transformer-Based DeepONet, whose cross-attention decoder
# cannot hold a graph over 8192. Both caps are OOM limits measured on a single GPU.
#
# Result: w_phys = 0.1 costs about 0.3 points of I-L2 and buys about 15 points of rho,
# the normalised RTE residual. It does not buy accuracy.
#
#   setsid nohup bash scripts/physics_loss.sh > src/logs/physics_loss.log 2>&1 &
#
# Cost, measured: 6.2 s/epoch for PCA, 12.7 s/epoch for AE, ~23 s/epoch for the
# Transformer-Based DeepONet -> 3.5 h + 7 h + 13 h, about a day, sequential.
. "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

W=${W_PHYS:-0.1}
EPOCHS=${EPOCHS:-2000}
N_COLLOC=${N_COLLOC:-50000}
BATCH_PDE=${BATCH_PDE:-64}
N_QUERY_PHYS=${N_QUERY_PHYS:-2048}

wait_for_gpu
log "=== PHYSICS LOSS: data + $W * RTE residual, $EPOCHS epochs ==="

# --- PCA-DeepONet p128 (a PCA basis has no trainable weight: nothing to fine-tune)
M="PCA-p128-softplus-phys"
cp -n "$CKPT/encoder_pca_PCA-p128.pkl" "$CKPT/encoder_pca_${M}.pkl"
run "$M" "1_train" \
    $PY src/train.py --model pca --name "$M" --stage operator --p_basis 128 \
       --epochs $EPOCHS --w_phys "$W" --n_colloc $N_COLLOC --batch_size_pde $BATCH_PDE
run "$M" "2_eval" $PY src/evaluate.py --name "$M" --physics --phys_snapshots 200

# --- AE-DeepONet p128, encoder fine-tuned (the default, at 1e-4)
M="AE-p128-softplus-phys"
cp -n "$CKPT/encoder_autoencoder_AE-p128.pt" "$CKPT/encoder_autoencoder_${M}.pt"
run "$M" "1_train" \
    $PY src/train.py --model ae --name "$M" --stage operator --p_basis 128 \
       --epochs $EPOCHS --w_phys "$W" --n_colloc $N_COLLOC --batch_size_pde $BATCH_PDE \
       --no-freeze_encoder --encoder_lr 1e-4
run "$M" "2_eval" $PY src/evaluate.py --name "$M" --physics --phys_snapshots 200

# --- Transformer-Based DeepONet N4-D128, encoder fine-tuned
M="TBD-N4-D128-phys"
run "$M" "1_train" \
    $PY src/train.py --model tbd --name "$M" --stage operator -N 4 -D 128 \
       --epochs $EPOCHS --w_phys "$W" --n_query_phys $N_QUERY_PHYS \
       --no-freeze_encoder --encoder_lr 1e-4 \
       --pretrained_encoder "$CKPT/encoder_tbd_TBD-N4-D128.pt"
run "$M" "2_eval" $PY src/evaluate.py --name "$M" --physics --phys_snapshots 200

log "=== DONE -- data-only twin on the left, hybrid on the right ==="
printf '%-24s %8s %8s %8s\n' "model" "I-L2" "G-L2" "rho"
for M in PCA-p128-softplus PCA-p128-softplus-phys AE-p128-softplus AE-p128-softplus-phys \
         TBD-N4-D128 TBD-N4-D128-phys; do
  F="$SRC/results/$M/${M}_metrics_test.txt"
  [ -f "$F" ] || continue
  I=$(awk '/^  I  L2/ {print $4}' "$F"); G=$(awk '/^  G  L2/ {print $4}' "$F")
  R=$(awk '/rho = RMS/ {print $5}' "$F")
  printf '%-24s %8s %8s %8s\n' "$M" "${I:-—}" "${G:-—}" "${R:-—}"
done
