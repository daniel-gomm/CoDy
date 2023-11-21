#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

source "$SCRIPT_DIR/common.bash"

EXPLAINER_TYPES=("pg_explainer" "tgnnexplainer" "greedy" "cody")

function evaluate() {
    for explainer in "${EXPLAINER_TYPES[@]}"; do
      case "$explainer" in
        pg_explainer|tgnnexplainer)
          bash "$SCRIPT_DIR/evaluate.bash" "$1" "$explainer"
          ;;
        *)
          bash "$SCRIPT_DIR/evaluate.bash" "$1" "$explainer" "all"
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