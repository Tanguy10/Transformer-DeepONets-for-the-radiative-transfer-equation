#!/usr/bin/env bash
. "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

W=${W_PHYS:-0.1}
EPOCHS=${EPOCHS:-2000}
N_COLLOC=${N_COLLOC:-50000}
BATCH_PDE=${BATCH_PDE:-64}
N_QUERY_PHYS=${N_QUERY_PHYS:-2048}

wait_for_gpu
log "=== PHYSICS LOSS: data + $W * RTE residual, $EPOCHS epochs ==="

M="PCA-p128-softplus-phys"
cp -n "$CKPT/encoder_pca_PCA-p128.pkl" "$CKPT/encoder_pca_${M}.pkl"
run "$M" "1_train" \
    $PY src/train.py --model pca --name "$M" --stage operator --p_basis 128 \
       --epochs $EPOCHS --w_phys "$W" --n_colloc $N_COLLOC --batch_size_pde $BATCH_PDE
run "$M" "2_eval" $PY src/evaluate.py --name "$M" --physics --phys_snapshots 200

M="AE-p128-softplus-phys"
cp -n "$CKPT/encoder_autoencoder_AE-p128.pt" "$CKPT/encoder_autoencoder_${M}.pt"
run "$M" "1_train" \
    $PY src/train.py --model ae --name "$M" --stage operator --p_basis 128 \
       --epochs $EPOCHS --w_phys "$W" --n_colloc $N_COLLOC --batch_size_pde $BATCH_PDE \
       --no-freeze_encoder --encoder_lr 1e-4
run "$M" "2_eval" $PY src/evaluate.py --name "$M" --physics --phys_snapshots 200

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
