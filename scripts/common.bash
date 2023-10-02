#!/bin/bash

RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PARENT_DIR=$(dirname "$SCRIPT_DIR")

PROCESSED_DATA_DIR="$PARENT_DIR/resources/datasets/processed"

RESULTS_DIR="$PARENT_DIR/resources/results"

DATASET_NAMES=($(find "$PROCESSED_DATA_DIR" -mindepth 1 -maxdepth 1 -type d -exec basename {} \;))

EXPLAINER_TYPES=("pg_explainer" "tgnnexplainer" "greedy" "searching")

SAMPLER_TYPES=("random" "recent" "closest" "pretrained")


function test_exists() {
  local tested_item="$1"
  shift
  local options_array=("$@")

  match_found=false
  for element in "${options_array[@]}"; do
    if [ "$element" = "$tested_item" ]; then
      match_found=true
      break
    fi
  done

  if [ "$match_found" = true ]; then
    return
  else
    echo -e "${RED}\"$tested_item\" is not a valid name!${NC}
Possible options are: [${CYAN}${options_array[*]}${NC}]"
    show_help
  fi
}