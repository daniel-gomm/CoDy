#!/bin/bash

RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PARENT_DIR=$(dirname "$SCRIPT_DIR")

PROCESSED_DATA_DIR="$PARENT_DIR/resources/datasets/processed"

DATASET_NAMES=($(find "$PROCESSED_DATA_DIR" -mindepth 1 -maxdepth 1 -type d -exec basename {} \;))

train_pg_explainer() {
  TGN_PATH="$PARENT_DIR/resources/models/$1/$1-$1.pth"
  MODEL_PATH="$PARENT_DIR/resources/models/$1/pg_explainer"
  if [ ! -d "$MODEL_PATH" ]; then
    mkdir -p "$MODEL_PATH"
    echo "Created new directory for the final PGExplainer model and model checkpoints for the $1 dataset at $MODEL_PATH"
  fi
  echo "Training PGExplainer model for the $1 dataset..."
  python "$SCRIPT_DIR/train_pgexplainer.py" -d "$PROCESSED_DATA_DIR/$1" --bipartite --cuda --model_path "$MODEL_PATH" --epochs 1 --model $TGN_PATH --candidates_size 30
}


show_help() {
  echo -e "
Train PGExplainer training script

Usage: bash $SCRIPT_DIR/train_pg_explainer.bash ${RED}DATASET-NAME${NC}

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
  train_pg_explainer "$1"
fi
