#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

source "$SCRIPT_DIR/common.bash"

train_tgn() {
  MODEL_PATH="$PARENT_DIR/resources/models/$1"
  if [ ! -d "$MODEL_PATH" ]; then
    mkdir -p "$MODEL_PATH"
    echo "Created new directory for the final model and model checkpoints for the $1 dataset at $MODEL_PATH"
  fi
  if [ "$2" = "--bipartite" ]; then
    echo "Training TGN model for the $1 dataset (bipartite)..."
    python "$SCRIPT_DIR/train_tgnn.py" -d "$PROCESSED_DATA_DIR/$1" --bipartite --cuda --model_path "$MODEL_PATH/" -e 30
  else
    echo "Training TGN model for the $1 dataset..."
    python "$SCRIPT_DIR/train_tgnn.py" -d "$PROCESSED_DATA_DIR/$1" --cuda --model_path "$MODEL_PATH/" -e 30
  fi
}


show_help() {
  echo -e "
Train TGN Model script

Usage: bash $SCRIPT_DIR/train_tgn_model.bash ${RED}DATASET-NAME${NC} ${RED}--bipartite${NC}

For the ${RED}DATASET-NAME${NC} parameter provide the name of any of the preprocessed datasets.
Possible values: ${CYAN}[${DATASET_NAMES[*]}]${NC}

Provide the ${RED}--bipartite${NC} flag if the dataset is bipartite
"
exit 1
}

if [ $# -eq 0 ]; then
  show_help
else
  test_exists "$1" "${DATASET_NAMES[@]}"
  train_tgn "$1" "$2"
fi
