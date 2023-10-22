#!/bin/bash

RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PARENT_DIR=$(dirname "$SCRIPT_DIR")

PROCESSED_DATA_DIR="$PARENT_DIR/resources/datasets/processed"

DATASET_NAMES=($(find "$PROCESSED_DATA_DIR" -mindepth 1 -maxdepth 1 -type d -exec basename {} \;))

EXPLAINER_TYPES=("pg_explainer" "tgnnexplainer" "greedy" "cftgnnexplainer")

SAMPLER_TYPES=("random" "recent" "closest" "pretrained" "1-best")

function evaluate() {
    for explainer in "${EXPLAINER_TYPES[@]}"; do
      case "$explainer" in
        pg_explainer|tgnnexplainer)
          bash "$SCRIPT_DIR/evaluate.bash" "$1" "$explainer"
          ;;
        *)
          for sampler in "${SAMPLER_TYPES[@]}"; do
            bash "$SCRIPT_DIR/evaluate.bash" "$1" "$explainer" "$sampler"
          done
          ;;
      esac
    done
}


function show_help() {
  echo -e "
Script for evaluating all explainer models on one dataset

Usage: bash $SCRIPT_DIR/run_evaluation.bash ${RED}DATASET-NAME${NC}

For the ${RED}DATASET-NAME${NC} parameter provide the name of any of the preprocessed datasets.
Possible values: ${CYAN}[${DATASET_NAMES[*]}]${NC}
"
exit 1
}

if [ $# -eq 0 ]; then
  show_help
else
  test_exists "$1" "${DATASET_NAMES[@]}"
  evaluate "$1"
fi