import glob
import os
import sys
from argparse import Namespace, ArgumentParser

import numpy as np
import pandas as pd
import torch

from CFTGNNExplainer.connector.tgnnwrapper import TGNWrapper
from CFTGNNExplainer.data.dataset import ContinuousTimeDynamicGraphDataset, TrainTestDatasetParameters
from TGN.model.tgn import TGN
from TGN.utils.utils import get_neighbor_finder


def parse_args(parser: ArgumentParser) -> Namespace:
    try:
        return parser.parse_args()
    except SystemExit:
        parser.print_help()
        sys.exit(0)


def add_dataset_arguments(parser: ArgumentParser):
    parser.add_argument('-d', '--dataset', required=True, type=str, help='Path to the dataset folder')
    parser.add_argument('--directed', action='store_true', help='Provide if the graph is directed')
    parser.add_argument('--bipartite', action='store_true', help='Provide if the graph is bipartite')


def add_wrapper_model_arguments(parser: ArgumentParser):
    parser.add_argument('-m', '--model', required=True, default=None, type=str,
                        help='Path to the model checkpoint to use')
    parser.add_argument('--cuda', action='store_true', help='Use cuda for GPU utilization')


def add_model_training_arguments(parser: ArgumentParser):
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the directory where the model checkpoints, final model and results are saved to.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train the model for.')


def create_dataset_from_args(args: Namespace, parameters: TrainTestDatasetParameters | None = None) -> (
        ContinuousTimeDynamicGraphDataset):
    if parameters is None:
        parameters = TrainTestDatasetParameters(0.2, 0.6, 0.8, 1000, 500, 500)

    # Get dataset
    dataset_folder = args.dataset

    events = glob.glob(os.path.join(dataset_folder, '*_data.csv'))
    edge_features = glob.glob(os.path.join(dataset_folder, '*_edge_features.npy'))
    node_features = glob.glob(os.path.join(dataset_folder, '*_node_features.npy'))

    name = edge_features[0][:-18]
    assert len(events) == len(edge_features) == len(node_features) == 1
    assert name == edge_features[0][:-18] == events[0][:-9]

    name = name.split('/')[-1]
    all_event_data = pd.read_csv(events[0])
    edge_features = np.load(edge_features[0])
    node_features = np.load(node_features[0])

    return ContinuousTimeDynamicGraphDataset(all_event_data, edge_features, node_features, name,
                                             directed=args.directed, bipartite=args.bipartite,
                                             parameters=parameters)


def create_tgn_wrapper_from_args(args: Namespace, dataset: ContinuousTimeDynamicGraphDataset | None = None):
    if dataset is None:
        dataset = create_dataset_from_args(args)

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

    tgn.to(device)

    return TGNWrapper(tgn, dataset, num_hops=2, model_name=dataset.name, device=device, n_neighbors=20,
                      batch_size=128, checkpoint_path=args.model)
