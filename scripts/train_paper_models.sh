#!/usr/bin/env bash
# Tables I and II of the paper: the seventeen fixed-resolution models.
#
#   PCA-p{128,256,512}           PCA-DeepONet, softplus head
#   PCA-p{128,256,512}-affine    the same, affine head (no softplus, Thm 3.4 to the letter)
#   AE-p{128,256,512}            AE-DeepONet, softplus head
#   AE-p{128,256,512}-affine     the same, affine head
#   TBD-N4-D32, TBD-N2-D64,      Transformer-Based DeepONet, the (N, D) grid
#   TBD-N1-D128, TBD-N2-D128,
#   TBD-N4-D128
#
# Identical budgets everywhere -- encoder 1000 epochs, operator 2000 epochs, batch 64
# (Transformer-Based DeepONet) / 256 (DeepONet), rel-L2 data loss, no physics term --
# which is exactly the default of src/train.py, so nothing but the run name and the
# latent size has to be passed.
#
# Learning rates, identical for every run:
#   * main network (branch+trunk, or the coordinate decoder): 1e-3 with
#     ExponentialLR(0.9977), i.e. 0.9977^2000 ~ 0.010 -- 1e-3 decaying to 1e-5;
#   * encoder while fine-tuned (AE and Transformer-Based DeepONet): 1e-4, same decay;
#   * encoder reconstruction pretraining (AE, Transformer-Based DeepONet): 1e-3 for
#     1000 epochs.
#
# The last step is Table I: the two encoders alone, side by side, on the test split.
#
#   setsid nohup bash scripts/train_paper_models.sh > src/logs/paper_models.log 2>&1 &
#
# Restart: one .done_<tag> marker per step plus intra-step --resume. Just relaunch the
# same file after a cut. Do NOT edit it while it runs (bash re-reads it by offset).
#
# Expected wall time on a single GPU: PCA ~5 h, AE ~16 h, Transformer ~52 h, plus the
# six affine heads ~14 h -- about four days for the whole grid.
# Override the interpreter with PY=..., or restrict the grid with FAMILIES="pca tbd".
. "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

EPOCHS_ENC=1000
EPOCHS_OP=2000
P_BASIS=${P_BASIS:-"128 256 512"}
TBD_GRID=${TBD_GRID:-"4,32 2,64 1,128 2,128 4,128"}
FAMILIES=${FAMILIES:-"pca ae tbd"}
MODELS=""

wait_for_gpu
log "=== PAPER MODELS: query (x,y,phi), rel-L2 loss, data-only, $EPOCHS_OP operator epochs ==="

# =============================================================================
#  Tables I-II, A & B. DeepONet -- PCA and AE encoders, three p, two heads
# =============================================================================
for FAM in $FAMILIES; do
  case $FAM in
    pca) BASE=PCA; ENC_EXT=pkl; ENC_KIND=pca ;;
    ae)  BASE=AE;  ENC_EXT=pt;  ENC_KIND=autoencoder ;;
    *)   continue ;;
  esac

  # The encoder does not depend on p (that is internal to the DeepONet): it is trained
  # once and copied, so every p starts from the SAME weights and diverges only through
  # fine-tuning. --encoder_epochs is a no-op for PCA (a direct fit).
  set -- $P_BASIS; P0=$1
  run "${BASE}-p${P0}" "0_encoder" \
      $PY src/train.py --model $FAM --name "${BASE}-p${P0}" --stage encoder \
         --encoder_epochs $EPOCHS_ENC || exit 1
  SRC_ENC="$CKPT/encoder_${ENC_KIND}_${BASE}-p${P0}.$ENC_EXT"
  for P in $P_BASIS; do
    DST="$CKPT/encoder_${ENC_KIND}_${BASE}-p${P}.$ENC_EXT"
    if [ -e "$SRC_ENC" ] && [ ! -e "$DST" ]; then
      cp -p "$SRC_ENC" "$DST" && log "encoder copied: $(basename "$DST")"
    fi
  done

  for P in $P_BASIS; do
    # Both heads share the encoder of the ${BASE}-p${P} configuration
    # (cf. encoders.encoder_run): the output activation is their only difference.
    for HEAD in softplus affine; do
      M="${BASE}-p${P}-${HEAD}"
      run "$M" "1_train" \
          $PY src/train.py --model $FAM --name "$M" --stage operator \
             --epochs $EPOCHS_OP --p_basis $P $([ "$HEAD" = affine ] && echo --affine)
      run "$M" "2_eval" $PY src/evaluate.py --name "$M"
      MODELS="$MODELS $M"
    done
  done
done

# =============================================================================
#  Table II, C. Transformer-Based DeepONet -- the (N, D) grid
# =============================================================================
case " $FAMILIES " in *" tbd "*)
  for ND in $TBD_GRID; do
    N=${ND%,*}; D=${ND#*,}
    M="TBD-N${N}-D${D}"
    run "$M" "0_encoder" \
        $PY src/train.py --model tbd --name "$M" --stage encoder \
           -N "$N" -D "$D" --encoder_epochs $EPOCHS_ENC
    run "$M" "1_train" \
        $PY src/train.py --model tbd --name "$M" --stage operator --epochs $EPOCHS_OP
    run "$M" "2_eval" $PY src/evaluate.py --name "$M"
    MODELS="$MODELS $M"
  done
;; esac

# =============================================================================
#  Table I. The two encoders alone, on the same test split
# =============================================================================
run "encoders" "3_recon" \
    $PY src/evaluate.py --encoders pca:PCA-p128,ae:AE-p128 --name encoders --split test

log "=== DONE ==="
report $MODELS
echo
cat "$SRC/results/encoders/encoders_recon_test.txt" 2>/dev/null
