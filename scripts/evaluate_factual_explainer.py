import argparse
import json
import logging
import os.path
import time
from typing import List

import numpy as np
import pandas as pd
import torch

from cody.data import TrainTestDatasetParameters
from cody.embedding import StaticEmbedding
from cody.implementations.tgn import to_data_object
from cody.utils import ProgressBar
from TTGN.model.tgn import TGN
from TTGN.utils.utils import get_neighbor_finder

from common import add_dataset_arguments, add_wrapper_model_arguments, create_dataset_from_args, parse_args, \
    get_event_ids_from_file, column_to_int_array, column_to_float_array

from cody.implementations.ttgn import TTGNWrapper
from cody.explainer.baseline.pgexplainer import TPGExplainer, FactualExplanation
from cody.explainer.baseline.tgnnexplainer import TGNNExplainer, TGNNExplainerExplanation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def evaluate(evaluated_explainer: TGNNExplainer | TPGExplainer, explained_event_ids: np.ndarray,
             max_time_seconds: int = 72 * 60):
    explanation_list = []

    progress_bar = ProgressBar(len(explained_event_ids), prefix='Evaluating explainer')
    start_time = time.time()

    for event_id in explained_event_ids:
        if time.time() - start_time > max_time_seconds:
            logger.info("Time limit reached. Finishing evaluation...")
            break
        progress_bar.update_postfix(f'Generating explanation for event {event_id}')
        try:
            explanation = evaluated_explainer.explain(event_id)
            sparsity_list, fidelity_list, fidelity_best = evaluated_explainer.evaluate_fidelity(explanation)
            explanation.statistics['sparsity'] = sparsity_list
            explanation.statistics['fidelity'] = fidelity_list
            explanation.statistics['best fidelity'] = fidelity_best
            explanation_list.append(explanation)
        except RuntimeError:
            progress_bar.write(f'Could not find any candidates to explain {event_id}')
        progress_bar.next()
    progress_bar.close()
    return explanation_list


def export_explanations(explanation_list: List[FactualExplanation | TGNNExplainerExplanation], filepath: str):
    explanations_dicts = [explanation.to_dict() for explanation in explanation_list]
    explanations_df = pd.DataFrame(explanations_dicts)
    parquet_file_path = filepath.rstrip('csv') + 'parquet'
    if os.path.exists(parquet_file_path):
        existing_results = pd.read_parquet(parquet_file_path)
        explanations_df = pd.concat([existing_results, explanations_df], axis='rows')
    elif os.path.exists(filepath):
        existing_results = pd.read_csv(filepath)
        existing_results = existing_results.iloc[:, 1:]
        existing_results['results'] = existing_results['results'].str.replace("\'", '\"')
        existing_results['results'] = existing_results['results'].apply(lambda x: json.loads(x))
        column_to_int_array(existing_results, 'candidates')
        column_to_float_array(existing_results, 'sparsity')
        column_to_float_array(existing_results, 'fidelity')
        column_to_float_array(existing_results, 'best fidelity')
        explanations_df = pd.concat([existing_results, explanations_df], axis='rows')
    try:
        explanations_df.to_parquet(filepath.rstrip('csv') + 'parquet')
        logger.info(f'Saved evaluation results to {parquet_file_path}')
    except ImportError:
        logger.info('Failed to export to parquet format. Install pyarrow to export to parquet format '
                    '(pip install pyarrow)')
    explanations_df.to_csv(filepath)
    logger.info(f'Saved evaluation results to {filepath}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Factual Explainer Evaluation')
    add_dataset_arguments(parser)
    add_wrapper_model_arguments(parser)
    parser.add_argument('--explained_ids', required=True, type=str,
                        help='Path to the file containing all the event ids that should be explained')
    parser.add_argument('--wrong_predictions_only', action='store_true',
                        help='Provide if evaluation should focus on wrong predictions only')
    parser.add_argument('-r', '--results', required=True, type=str,
                        help='Filepath for the evaluation results')
    parser.add_argument('--candidates_size', type=int, default=50,
                        help='Number of candidates from which the samples are selected')
    parser.add_argument('--explainer', required=True, type=str, help='Which explainer to evaluate',
                        choices=['pg_explainer', 't_gnnexplainer'])
    parser.add_argument('--explainer_model_path', type=str, required=True,
                        help='Path to the model file of the PG-Explainer model')
    parser.add_argument('--rollout', type=int, default=500,
                        help='Number of rollouts to perform in the MCTS')
    parser.add_argument('--mcts_save_dir', type=str,
                        help='Path to which the results of the mcts are written to')
    parser.add_argument('--number_of_explained_events', type=int, default=1000,
                        help='Number of event ids to explain. Only has an effect if the explained_ids file has not '
                             'been initialized yet')
    parser.add_argument('--max_time', type=int, default=2400,
                        help='Maximal runtime (minutes)')

    args = parse_args(parser)

    dataset = create_dataset_from_args(args, TrainTestDatasetParameters(0.2, 0.6, 0.8, args.number_of_explained_events,
                                                                        500, 500))

    device = 'cpu'
    if args.cuda:
        device = 'cuda'

    tgn = TGN(
        neighbor_finder=get_neighbor_finder(to_data_object(dataset), uniform=False),
        node_features=dataset.node_features,
        edge_features=dataset.edge_features,
        device=torch.device(device),
        use_memory=True,
        memory_update_at_start=False,
        memory_dimension=172,
        embedding_module_type='graph_attention',
        message_function='identity',
        aggregator_type='last',
        memory_updater_type='gru',
        use_destination_embedding_in_message=False,
        use_source_embedding_in_message=False,
        dyrep=False,
        n_neighbors=20
    )

    tgn.to(device)

    tgn_wrapper = TTGNWrapper(tgn, dataset, num_hops=2, model_name=dataset.name, device=device, n_neighbors=20,
                              explanation_candidates_size=args.candidates_size,
                              batch_size=32, checkpoint_path=args.model)

    event_ids_to_explain = get_event_ids_from_file(args.explained_ids, logger, args.wrong_predictions_only,
                                                   tgn_wrapper)

    embedding = StaticEmbedding(dataset, tgn_wrapper)

    pg_explainer = TPGExplainer(tgn_wrapper, embedding=embedding, device=tgn_wrapper.device)

    match args.explainer:
        case 'pg_explainer':
            explainer = pg_explainer
        case 't_gnnexplainer':
            explainer = TGNNExplainer(tgn_wrapper, embedding, pg_explainer, results_dir=args.mcts_save_dir,
                                      device=tgn_wrapper.device, rollout=args.rollout, mcts_saved_dir=None,
                                      save_results=True)
        case _:
            raise NotImplementedError

    if os.path.exists(args.results):
        previous_results = pd.read_csv(args.results)
        encountered_event_ids = previous_results['explained_event_id'].to_numpy()
        logger.info(f'Resuming evaluation. '
                    f'Already processed {len(encountered_event_ids)}/{len(event_ids_to_explain)} events.')
        event_ids_to_explain = event_ids_to_explain[~np.isin(event_ids_to_explain, encountered_event_ids)]

    explanations = evaluate(explainer, event_ids_to_explain, args.max_time * 60)
    export_explanations(explanations, args.results)
