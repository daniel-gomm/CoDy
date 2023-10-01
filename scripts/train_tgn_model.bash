#!/bin/bash

RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PARENT_DIR=$(dirname "$SCRIPT_DIR")

PROCESSED_DATA_DIR="$PARENT_DIR/resources/datasets/processed"

DATASET_NAMES=($(find "$PROCESSED_DATA_DIR" -mindepth 1 -maxdepth 1 -type d -exec basename {} \;))

train_tgn() {
  MODEL_PATH="$PARENT_DIR/resources/models/$1"
  if [ ! -d "$MODEL_PATH" ]; then
    mkdir -p "$MODEL_PATH"
    echo "Created new directory for the final model and model checkpoints for the $1 dataset at $MODEL_PATH"
  fi
  echo "Training TGN model for the $1 dataset..."
  python "$SCRIPT_DIR/train_tgnn.py" -d "$PROCESSED_DATA_DIR/$1" --bipartite --cuda --model_path "$MODEL_PATH" -e 30
}


show_help() {
  echo -e "
Train TGN Model script

Usage: bash $SCRIPT_DIR/train_tgn_model.bash ${RED}DATASET-NAME${NC}

For the ${RED}DATASET-NAME${NC} parameter provide the name of any of the preprocessed datasets.
Possible values: ${CYAN}[${DATASET_NAMES[*]}]${NC}
"
exit 1
}

test_exists() {
  match_found=false
  for element in "${DATASET_NAMES[@]}"; do
    if [ "$element" = "$1" ]; then
      match_found=true
      break
    fi
  done

  if [ "$match_found" = true ]; then
    return
  else
    echo -e "${RED}\"$1\" is not the name of a valid dataset!${NC}"
    show_help
  fi
}

if [ $# -eq 0 ]; then
  show_help
else
  test_exists "$1"
  train_tgn "$1"
fi
