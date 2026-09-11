#!/usr/bin/env bash
. "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

EPOCHS_ENC=1000
EPOCHS_OP=2000
P_BASIS=${P_BASIS:-"128 256 512"}
TBD_GRID=${TBD_GRID:-"4,32 2,64 1,128 2,128 4,128"}
FAMILIES=${FAMILIES:-"pca ae tbd"}
MODELS=""

wait_for_gpu
log "=== PAPER MODELS: query (x,y,phi), rel-L2 loss, data-only, $EPOCHS_OP operator epochs ==="

for FAM in $FAMILIES; do
  case $FAM in
    pca) BASE=PCA; ENC_EXT=pkl; ENC_KIND=pca ;;
    ae)  BASE=AE;  ENC_EXT=pt;  ENC_KIND=autoencoder ;;
    *)   continue ;;
  esac

  # One encoder per family, copied to every p: all ranks start from the same weights.
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

run "encoders" "3_recon" \
    $PY src/evaluate.py --encoders pca:PCA-p128,ae:AE-p128 --name encoders --split test

log "=== DONE ==="
report $MODELS
echo
cat "$SRC/results/encoders/encoders_recon_test.txt" 2>/dev/null
