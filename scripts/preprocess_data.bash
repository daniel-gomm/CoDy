#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source "$SCRIPT_DIR/common.bash"

DATA_DIR="$PARENT_DIR/resources/datasets"
RAW_DATA_DIR="$DATA_DIR/raw"
PROCESSED_DATA_DIR="$DATA_DIR/processed"


for file in "$RAW_DATA_DIR"/*; do
  if [ -f "$file" ]; then
    DATASET_FILENAME=${file##*/}
    DATASET_NAME=${DATASET_FILENAME%.csv}
    DATASET_DIR="$PROCESSED_DATA_DIR/$DATASET_NAME"

    if [ ! -d "$DATASET_DIR" ]; then
      mkdir -p "$DATASET_DIR"
      echo "Created new directory for the $DATASET_NAME dataset at $DATASET_DIR"
    fi
    echo "Preprocessing the $DATASET_NAME dataset..."
    python "$SCRIPT_DIR/preprocess_dataset.py" -f "$file" -t "$DATASET_DIR" --bipartite
  fi
done

echo "Finished preprocessing all datasets in $RAW_DATA_DIR"
