#!/bin/bash

RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PARENT_DIR=$(dirname "$SCRIPT_DIR")

PROCESSED_DATA_DIR="$PARENT_DIR/resources/datasets/processed"

RESULTS_DIR="$PARENT_DIR/resources/results"

DATASET_NAMES=($(find "$PROCESSED_DATA_DIR" -mindepth 1 -maxdepth 1 -type d -exec basename {} \;))

FACTUAL_EXPLAINER_TYPES=("pg_explainer" "tgnnexplainer")
COUNTERFACTUAL_EXPLAINER_TYPES=("greedy" "searching" "cody")
EXPLAINER_TYPES=("${FACTUAL_EXPLAINER_TYPES[@]}" "${COUNTERFACTUAL_EXPLAINER_TYPES[@]}")

SAMPLER_TYPES=("random" "temporal" "spatio-temporal" "pretrained" "1-best")
ALL_SAMPLER_TYPES=("${SAMPLER_TYPES[@]}" "all")


function value_in_array() {
  local tested_item="$1"
  shift
  local options_array=("$@")

  for element in "${options_array[@]}"; do
    if [ "$element" = "$tested_item" ]; then
      return 0
    fi
  done
  return 1
}


function test_exists() {
  local tested_item="$1"
  shift
  local options_array=("$@")
  if value_in_array "$tested_item" "${options_array[@]}"; then
    return
  else
    echo -e "${RED}\"$tested_item\" is not a valid name!${NC}
Possible options are: [${CYAN}${options_array[*]}${NC}]"
    show_help
  fi
}