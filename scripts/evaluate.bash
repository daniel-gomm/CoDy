#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

source "$SCRIPT_DIR/common.bash"

evaluate_explainer() {
  TGN_PATH="$PARENT_DIR/resources/models/$1/$1-$1.pth"
  EVAL_RESULTS_DIR="$RESULTS_DIR/$1"
  RESULTS_SAVE_DIR="$EVAL_RESULTS_DIR/$2"
  EXPLAINED_IDS_PATH="$EVAL_RESULTS_DIR/evaluation_event_ids.npy"
  if [ ! -d "$EVAL_RESULTS_DIR" ]; then
    mkdir -p "$EVAL_RESULTS_DIR"
    echo "Created new directory for the evaluation results for the $1 dataset at $MODEL_PATH"
  fi
  mkdir -p "$RESULTS_SAVE_DIR"

  echo "Starting evaluation for dataset $1 with explainer $2"
  case $2 in
  pg_explainer)
    PG_EXP_MODEL_PATH="$PARENT_DIR/resources/models/$1/pg_explainer/$1_final.pth"
      if [ "$4" = "--bipartite" ];then
        python "$SCRIPT_DIR/evaluate_factual_explainer.py" -d "$PROCESSED_DATA_DIR/$1" --bipartite --cuda --explainer_model_path "$PG_EXP_MODEL_PATH" --model "$TGN_PATH" --candidates_size 30 --explainer pg_explainer --number_of_explained_events 200 --explained_ids "$EXPLAINED_IDS_PATH" --results "$RESULTS_SAVE_DIR/results_$1_$2.csv" --max_time "$3"
      else
        python "$SCRIPT_DIR/evaluate_factual_explainer.py" -d "$PROCESSED_DATA_DIR/$1" --cuda --explainer_model_path "$PG_EXP_MODEL_PATH" --model "$TGN_PATH" --candidates_size 30 --explainer pg_explainer --number_of_explained_events 200 --explained_ids "$EXPLAINED_IDS_PATH" --results "$RESULTS_SAVE_DIR/results_$1_$2.csv" --max_time "$3"
      fi
    ;;
  tgnnexplainer)
    PG_EXP_MODEL_PATH="$PARENT_DIR/resources/models/$1/pg_explainer/$1_final.pth"
    if [ "$4" = "--bipartite" ];then
      python "$SCRIPT_DIR/evaluate_factual_explainer.py" -d "$PROCESSED_DATA_DIR/$1" --bipartite --cuda --explainer_model_path "$PG_EXP_MODEL_PATH" --model "$TGN_PATH" --candidates_size 30 --explainer t_gnnexplainer --number_of_explained_events 200 --explained_ids "$EXPLAINED_IDS_PATH" --results "$RESULTS_SAVE_DIR/results_$1_$2.csv" --rollout 500 --mcts_save_dir "$RESULTS_SAVE_DIR/" --max_time "$3"
    else
        python "$SCRIPT_DIR/evaluate_factual_explainer.py" -d "$PROCESSED_DATA_DIR/$1" --cuda --explainer_model_path "$PG_EXP_MODEL_PATH" --model "$TGN_PATH" --candidates_size 30 --explainer t_gnnexplainer --number_of_explained_events 200 --explained_ids "$EXPLAINED_IDS_PATH" --results "$RESULTS_SAVE_DIR/results_$1_$2.csv" --rollout 500 --mcts_save_dir "$RESULTS_SAVE_DIR/" --max_time "$3"
    fi
    ;;
  greedy)
    echo "Selected sampler $3"
    SAMPLER_MODEL_PATH="$PARENT_DIR/resources/models/$1/sampler/$1_dynamic_sampler.pth"
    if [ "$5" = "--bipartite" ];then
      python "$SCRIPT_DIR/evaluate_cf_explainer.py" -d "$PROCESSED_DATA_DIR/$1" --bipartite --cuda --model "$TGN_PATH" --explainer greedy --number_of_explained_events 200 --explained_ids "$EXPLAINED_IDS_PATH" --results "$RESULTS_SAVE_DIR" --dynamic --predict_for_each_sample --sample_size 10 --candidates_size 64 --sampler "$3" --sampler_model_path "$SAMPLER_MODEL_PATH" --max_time "$4" --optimize
    else
      python "$SCRIPT_DIR/evaluate_cf_explainer.py" -d "$PROCESSED_DATA_DIR/$1" --cuda --model "$TGN_PATH" --explainer greedy --number_of_explained_events 200 --explained_ids "$EXPLAINED_IDS_PATH" --results "$RESULTS_SAVE_DIR" --dynamic --predict_for_each_sample --sample_size 10 --candidates_size 64 --sampler "$3" --sampler_model_path "$SAMPLER_MODEL_PATH" --max_time "$4" --optimize
    fi
    ;;
  cody)
    echo "Selected sampler $3"
    SAMPLER_MODEL_PATH="$PARENT_DIR/resources/models/$1/sampler/$1_dynamic_sampler.pth"
    if [ "$5" = "--bipartite" ];then
      python "$SCRIPT_DIR/evaluate_cf_explainer.py" -d "$PROCESSED_DATA_DIR/$1" --bipartite --cuda --model "$TGN_PATH" --explainer cody --number_of_explained_events 200 --explained_ids "$EXPLAINED_IDS_PATH" --results "$RESULTS_SAVE_DIR" --dynamic --predict_for_each_sample --sample_size 10 --candidates_size 64 --sampler "$3" --sampler_model_path "$SAMPLER_MODEL_PATH" --max_time "$4" --max_steps 300 --optimize
    else
      python "$SCRIPT_DIR/evaluate_cf_explainer.py" -d "$PROCESSED_DATA_DIR/$1" --cuda --model "$TGN_PATH" --explainer cody --number_of_explained_events 200 --explained_ids "$EXPLAINED_IDS_PATH" --results "$RESULTS_SAVE_DIR" --dynamic --predict_for_each_sample --sample_size 10 --candidates_size 64 --sampler "$3" --sampler_model_path "$SAMPLER_MODEL_PATH" --max_time "$4" --max_steps 300 --optimize
    fi
    ;;
  *)
    show_help
    ;;
  esac
}


show_help() {
  echo -e "
Evaluation script

Usage: bash $SCRIPT_DIR/evaluate.bash ${RED}DATASET-NAME EXPLAINER-NAME [SAMPLER-NAME] --bipartite${NC}

For the ${RED}DATASET-NAME${NC} parameter provide the name of any of the preprocessed datasets.
Possible values: ${CYAN}[${DATASET_NAMES[*]}]${NC}
For the ${RED}EXPLAINER-NAME${NC} parameter provide the name of any of the possible explainers.
Possible values: ${CYAN}[${EXPLAINER_TYPES[*]}]${NC}
Optional: For the ${RED}SAMPLER-NAME${NC} parameter provide the name of any of the possible samplers (Counterfactual Explainers only).
Possible values: ${CYAN}[${SAMPLER_TYPES[*]}]${NC}

Provide the ${RED}--bipartite${NC} flag if the dataset is bipartite
"
exit 1
}

if [ $# -lt 2 ]; then
  show_help
else
  test_exists "$1" "${DATASET_NAMES[@]}"
  test_exists "$2" "${EXPLAINER_TYPES[@]}"
  time="600" # 10 Hours default value
  if value_in_array "$2" "${COUNTERFACTUAL_EXPLAINER_TYPES[@]}"; then
    if [ $# -gt 3 ]; then
      time="$4"
      echo "Concluding evaluation after maximum time of $time minutes"
    fi
    test_exists "$3" "${ALL_SAMPLER_TYPES[@]}"
    echo "Evaluating explainer $2 with sampler $3 on dataset $1"
    evaluate_explainer "$1" "$2" "$3" "$time" "$5"
  elif value_in_array "$2" "${FACTUAL_EXPLAINER_TYPES[@]}"; then
    if [ $# -gt 2 ]; then
      time="$3"
      echo "Concluding evaluation after maximum time of $time minutes"
    fi
    echo "Evaluating explainer $2 on dataset $1"
    evaluate_explainer "$1" "$2" "$time" "$4"
  else
    show_help
  fi
fi
