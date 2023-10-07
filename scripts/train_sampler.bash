#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

source "$SCRIPT_DIR/common.bash"

train_sampler() {
  TGN_PATH="$PARENT_DIR/resources/models/$1/$1-$1.pth"
  MODEL_PATH="$PARENT_DIR/resources/models/$1/sampler"
  if [ ! -d "$MODEL_PATH" ]; then
    mkdir -p "$MODEL_PATH"
    echo "Created new directory for the final Sampler model and model checkpoints for the $1 dataset at $MODEL_PATH"
  fi
  echo "Training Sampler model for the $1 dataset..."
  python "$SCRIPT_DIR/train_sampler.py" -d "$PROCESSED_DATA_DIR/$1" --bipartite --cuda --dynamic --model_save_path "$MODEL_PATH" --train_examples 500 --depth 2 --sample_size 20 --epochs 500 --model $TGN_PATH --candidates_size 75
}


show_help() {
  echo -e "
Train Sampler script

Usage: bash $SCRIPT_DIR/train_sampler.bash ${RED}DATASET-NAME${NC}

For the ${RED}DATASET-NAME${NC} parameter provide the name of any of the preprocessed datasets.
Possible values: ${CYAN}[${DATASET_NAMES[*]}]${NC}
"
exit 1
}

if [ $# -eq 0 ]; then
  show_help
else
  test_exists "$1" "${DATASET_NAMES[@]}"
  train_sampler "$1"
fi