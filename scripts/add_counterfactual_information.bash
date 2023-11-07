#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

source "$SCRIPT_DIR/common.bash"

show_help() {
  echo -e "
Evaluation script

Usage: bash $SCRIPT_DIR/add_counterfactual_information.bash ${RED}DATASET-NAME RESULTS-PATH${NC}

For the ${RED}DATASET-NAME${NC} parameter provide the name of any of the preprocessed datasets.
Possible values: ${CYAN}[${DATASET_NAMES[*]}]${NC}
For the ${RED}RESULTS-PATH${NC} parameter provide the path to the results file of a TGNNExplainer evaluation you want to test.
"
exit 1
}

if [ $# -lt 2 ]; then
  show_help
else
  test_exists "$1" "${DATASET_NAMES[@]}"
  TGN_PATH="$PARENT_DIR/resources/models/$1/$1-$1.pth"
  python "$SCRIPT_DIR/evaluate_factual_subgraphs.py" -d "$PROCESSED_DATA_DIR/$1" --bipartite --cuda --model "$TGN_PATH" --results "$2"
fi