#!/usr/bin/env bash
# Figure 7: the Theorem 3.4 lower bound on the error of an affine reconstructor of rank
# p, on the training set and on the test set, with the models of Table II placed on it.
#
# Every model sitting in src/checkpoints/ enters the comparison: its width p and its
# family are read off the checkpoint, and whether its reconstructor is affine is read
# off the checkpoint too (cf. deeponet.load_model). Nothing is listed by hand.
#
#   setsid nohup bash scripts/lower_bound.sh > src/logs/lower_bound.log 2>&1 &
#
# Cost: one eigenvalue problem of the size of the dataset (M x M, M <= 4200) plus one
# pass of each model over train and test -- about 40 min, then minutes on the cache.
. "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

REFRESH=${REFRESH:-none}      # none | spectrum | errors | all
# The eleven models of the comparison: the DeepONets in the class of Thm 3.4 (affine
# head, no softplus) and the Transformer-Based DeepONets as they are trained everywhere
# else. The other heads, and the `-phys` runs of Appendix B, do not belong on this
# figure: a head that is not the one being bounded, or another objective, would put two
# different questions on the same axes.
MODELS=${MODELS:-"PCA-p128-affine PCA-p256-affine PCA-p512-affine AE-p128-affine AE-p256-affine AE-p512-affine TBD-N4-D32 TBD-N2-D64 TBD-N1-D128 TBD-N2-D128 TBD-N4-D128"}

wait_for_gpu
log "=== LOWER BOUND (Thm 3.4) vs the measured errors ==="

$PY src/lower_bound/run.py --list

run "lower_bound" "1_run" \
    $PY src/lower_bound/run.py --refresh "$REFRESH" --models $MODELS \
       --out "$SRC/lower_bound/bound_comparison.pdf"

log "=== DONE ==="
cat "$LOGD/lower_bound/1_run.log"
