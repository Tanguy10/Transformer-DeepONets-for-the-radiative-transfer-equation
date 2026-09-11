#!/usr/bin/env bash
. "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

STEPS=${STEPS:-"paper_models power_spectrum lower_bound multi_resolution physics_loss ablation_junction"}

declare -A WHAT=(
  [paper_models]="Tables I-II: the 17 fixed-resolution models"
  [power_spectrum]="Figure 6: power spectral density of G"
  [lower_bound]="Figure 7: the Theorem 3.4 lower bound"
  [multi_resolution]="Table III, Figures 8-11: the mixed-resolution campaign"
  [physics_loss]="Appendix B: the RTE residual as a regulariser"
  [ablation_junction]="Appendix C: cross-attention vs the branch.trunk junction"
)
declare -A FILE=([paper_models]=train_paper_models)

for S in $STEPS; do
  F="$SCRIPTS/${FILE[$S]:-$S}.sh"
  [ -f "$F" ] || { log "no such step: $S"; exit 1; }
  log "########## $S -- ${WHAT[$S]:-}"
  bash "$F" || log "$S FAILED -- continuing"
done

log "########## ALL DONE"
