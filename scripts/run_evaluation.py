import argparse
import os
import logging
from typing import List

import numpy as np
import pandas as pd

from common import (add_dataset_arguments, add_wrapper_model_arguments, create_dataset_from_args,
                    create_tgn_wrapper_from_args, parse_args)

from CFTGNNExplainer.connector.bridge import DynamicTGNNBridge
from CFTGNNExplainer.explainer.evaluation import EvaluationExplainer, EvaluationCounterFactualExample, \
    EvaluationGreedyCFExplainer
from CFTGNNExplainer.utils import ProgressBar

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def evaluate(explainer: EvaluationExplainer, explained_event_ids: np.ndarray):
    explanations = []

    progress_bar = ProgressBar(len(explained_event_ids), prefix='Evaluating explainer')
    last_event_id = np.min(explained_event_ids) - 1

    for event_id in explained_event_ids:
        progress_bar.update_postfix(f'Generating original score for event {event_id}')
        original_prediction = explainer.get_evaluation_original_prediction(event_id, last_event_id)
        explainer.tgnn_bridge.reset_model()
        progress_bar.update_postfix(f'Generating explanation for event {event_id}')
        explanation = explainer.evaluate_explanation(event_id, original_prediction)
        explanations.append(explanation)
        last_event_id = event_id - 1
        progress_bar.next()

    return explanations


def export_explanations(explanations: List[EvaluationCounterFactualExample], filepath: str):
    explanations_dicts = [explanation.to_dict() for explanation in explanations]
    explanations_df = pd.DataFrame(explanations_dicts)
    explanations_df.to_csv(filepath)
    logger.info(f'Saved evaluation results to {filepath}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Explainer Evaluation')
    add_dataset_arguments(parser)
    add_wrapper_model_arguments(parser)
    parser.add_argument('--explained_ids', required=True, type=str,
                        help='Path to the file containing all the event ids that should be explained')
    parser.add_argument('-r', '--results', required=True, type=str,
                        help='Filepath for the evaluation results')
    parser.add_argument('--explainer', required=True, type=str, help='Which explainer to evaluate',
                        choices=['greedy', 'searching_random', 'searching_recent', 'searching_closest'])

    args = parse_args(parser)

    dataset = create_dataset_from_args(args)

    tgn_wrapper = create_tgn_wrapper_from_args(args, dataset)

    # load event ids to explain
    event_ids_filepath = args.explained_ids

    if os.path.exists(event_ids_filepath):
        event_ids_to_explain = np.load(event_ids_filepath)
    else:
        logger.info('No event ids to explain provided. Generating new ones...')
        event_ids_to_explain = dataset.extract_random_event_ids(section='validation')
        event_ids_to_explain = np.array(event_ids_to_explain)
        np.save(event_ids_filepath, event_ids_to_explain)

    explainer = args.explainer

    match explainer:
        case 'greedy':
            explainer = EvaluationGreedyCFExplainer(DynamicTGNNBridge(tgn_wrapper))
        case 'searching_random', 'searching_recent', 'searching_closest':
            raise NotImplementedError
        case _:
            raise NotImplementedError

    explanations = evaluate(explainer, event_ids_to_explain)
    export_explanations(explanations, args.results)
