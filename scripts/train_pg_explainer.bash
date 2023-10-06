#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

source "$SCRIPT_DIR/common.bash"

train_pg_explainer() {
  TGN_PATH="$PARENT_DIR/resources/models/$1/$1-$1.pth"
  MODEL_PATH="$PARENT_DIR/resources/models/$1/pg_explainer"
  if [ ! -d "$MODEL_PATH" ]; then
    mkdir -p "$MODEL_PATH"
    echo "Created new directory for the final PGExplainer model and model checkpoints for the $1 dataset at $MODEL_PATH"
  fi
  echo "Training PGExplainer model for the $1 dataset..."
  python "$SCRIPT_DIR/train_pgexplainer.py" -d "$PROCESSED_DATA_DIR/$1" --bipartite --cuda --model_path "$MODEL_PATH" --epochs 100 --model $TGN_PATH --candidates_size 30
}


show_help() {
  echo -e "
PGExplainer training script

Usage: bash $SCRIPT_DIR/train_pg_explainer.bash ${RED}DATASET-NAME${NC}

For the ${RED}DATASET-NAME${NC} parameter provide the name of any of the preprocessed datasets.
Possible values: ${CYAN}[${DATASET_NAMES[*]}]${NC}
"
exit 1
}

if [ $# -eq 0 ]; then
  show_help
else
  test_exists "$1" "${DATASET_NAMES[@]}"
  train_pg_explainer "$1"
fi
