#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source "$SCRIPT_DIR/common.bash"

DATASET_NAME="$1"


function show_help() {
  echo -e "
Script for preprocessing a dataset

Usage: bash $SCRIPT_DIR/preprocess_data.bash ${RED}DATASET-NAME${NC} ${RED}--bipartite${NC}

For the ${RED}DATASET-NAME${NC} parameter provide the name of any of the raw datasets that should be processed
Provide the ${RED}--bipartite${NC} flag if the processed dataset is bipartite
"
exit 1
}


if [ $# -eq 0 ]; then
  show_help
fi

DATA_DIR="$PARENT_DIR/resources/datasets"
RAW_DATA_DIR="$DATA_DIR/raw"
PROCESSED_DATA_DIR="$DATA_DIR/processed"

RAW_DATA_FILE="$RAW_DATA_DIR/$DATASET_NAME.csv"
DATASET_DIR="$PROCESSED_DATA_DIR/$DATASET_NAME"

if [ ! -d "$DATASET_DIR" ]; then
  mkdir -p "$DATASET_DIR"
  echo "Created new directory for the $DATASET_NAME dataset at $DATASET_DIR"
fi

if [ "$2" = "--bipartite" ]; then
  echo "Preprocessing the $DATASET_NAME dataset as bipartite graph..."
  python "$SCRIPT_DIR/preprocess_dataset.py" -f "$RAW_DATA_FILE" -t "$DATASET_DIR" --bipartite
else
  echo "Preprocessing the $DATASET_NAME dataset..."
  python "$SCRIPT_DIR/preprocess_dataset.py" -f "$RAW_DATA_FILE" -t "$DATASET_DIR"
fi

