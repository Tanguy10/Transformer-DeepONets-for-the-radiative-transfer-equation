#!/usr/bin/env bash
# Table III and Figures 8-11: the five models trained on the mixed-resolution dataset.
# The campaign itself lives with the sub-experiment; this is the entry point.
#
#   setsid nohup bash scripts/multi_resolution.sh > src/logs/multi_resolution.log 2>&1 &
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/../src/robustness_resolution/run_campaign.sh"
