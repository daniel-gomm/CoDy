import argparse
import logging
from typing import List

import numpy as np
import pandas as pd
import torch

from CFTGNNExplainer.data import TrainTestDatasetParameters
from CFTGNNExplainer.embedding import StaticEmbedding
from CFTGNNExplainer.implementations.tgn import to_data_object
from CFTGNNExplainer.utils import ProgressBar
from TTGN.model.tgn import TGN
from TTGN.utils.utils import get_neighbor_finder

from common import add_dataset_arguments, add_wrapper_model_arguments, create_dataset_from_args, parse_args, \
    get_event_ids_from_file

from CFTGNNExplainer.implementations.ttgn import TTGNWrapper
from CFTGNNExplainer.explainer.baseline.pgexplainer import TPGExplainer, FactualExplanation
from CFTGNNExplainer.explainer.baseline.tgnnexplainer import TGNNExplainer, TGNNExplainerExplanation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def evaluate(evaluated_explainer: TGNNExplainer | TPGExplainer, explained_event_ids: np.ndarray):
    explanation_list = []

    progress_bar = ProgressBar(len(explained_event_ids), prefix='Evaluating explainer')

    for event_id in explained_event_ids:
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

    return explanation_list


def export_explanations(explanation_list: List[FactualExplanation | TGNNExplainerExplanation], filepath: str):
    explanations_dicts = [explanation.to_dict() for explanation in explanation_list]
    explanations_df = pd.DataFrame(explanations_dicts)
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

    event_ids_to_explain = get_event_ids_from_file(args.explained_ids, dataset, logger, args.wrong_predictions_only,
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

    explanations = evaluate(explainer, event_ids_to_explain)
    export_explanations(explanations, args.results)
