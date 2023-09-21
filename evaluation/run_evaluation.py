import argparse
import sys
import os
import glob
import logging
from typing import List

import numpy as np
import pandas as pd
import torch

from CFTGNNExplainer.connector.bridge import DynamicTGNNBridge
from CFTGNNExplainer.explainer.evaluation import EvaluationExplainer, EvaluationCounterFactualExample, \
    EvaluationGreedyCFExplainer
from CFTGNNExplainer.utils import ProgressBar
from TGN.model.tgn import TGN
from TGN.utils.utils import get_neighbor_finder
from CFTGNNExplainer.data.dataset import ContinuousTimeDynamicGraphDataset, TrainTestDatasetParameters
from CFTGNNExplainer.connector.tgnnwrapper import TGNWrapper

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

    parser.add_argument('--explained_ids', required=True, type=str,
                        help='Path to the file containing all the event ids that should be explained')
    parser.add_argument('-d', '--dataset', required=True, type=str, help='Path to the dataset folder')
    parser.add_argument('-m', '--model', required=True, default=None, type=str,
                        help='Path to the model checkpoint to use')
    parser.add_argument('-r', '--results', required=True, type=str,
                        help='Filepath for the evaluation results')
    parser.add_argument('--explainer', required=True, type=str, help='Which explainer to evaluate',
                        choices=['greedy', 'searching_random', 'searching_recent', 'searching_closest'])
    parser.add_argument('--directed', action='store_true', help='Provide if the graph is directed')
    parser.add_argument('--bipartite', action='store_true', help='Provide if the graph is bipartite')
    parser.add_argument('--cuda', action='store_true', help='Use cuda for GPU utilization')

    try:
        args = parser.parse_args()
    except SystemExit:
        parser.print_help()
        sys.exit(0)

    # Get dataset
    dataset_folder = args.dataset

    events = glob.glob(os.path.join(dataset_folder, '*_data.csv'))
    edge_features = glob.glob(os.path.join(dataset_folder, '*_edge_features.npy'))
    node_features = glob.glob(os.path.join(dataset_folder, '*_node_features.npy'))

    name = edge_features[0][:-18]
    assert len(events) == len(edge_features) == len(node_features) == 1
    assert name == edge_features[0][:-18] == events[0][:-9]

    name = name.split('/')[-1]
    print(name)
    all_event_data = pd.read_csv(events[0])
    edge_features = np.load(edge_features[0])
    node_features = np.load(node_features[0])

    dataset = ContinuousTimeDynamicGraphDataset(all_event_data, edge_features, node_features, name,
                                                directed=args.directed, bipartite=args.bipartite,
                                                parameters=TrainTestDatasetParameters(0.2, 0.4, 0.8, 1000, 1000, 500))

    device = 'cpu'
    if args.cuda:
        device = 'cuda'

    tgn = TGN(
        neighbor_finder=get_neighbor_finder(dataset.to_data_object(), uniform=False),
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

    tgn_wrapper = TGNWrapper(tgn, dataset, num_hops=2, model_name=name, device=device, n_neighbors=20,
                             batch_size=128, checkpoint_path=args.model)

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
