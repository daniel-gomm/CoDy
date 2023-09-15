import argparse
import sys
import os
import glob

import numpy as np
import pandas as pd
import torch

from TGN.model.tgn import TGN
from TGN.utils.utils import get_neighbor_finder
from CFTGNNExplainer.data.dataset import ContinuousTimeDynamicGraphDataset
from CFTGNNExplainer.connector.tgnnwrapper import TGNWrapper

if __name__ == '__main__':
    parser = argparse.ArgumentParser('T-GNN Training')

    parser.add_argument('-d', '--dataset', required=True, type=str, help='Path to the dataset folder')
    parser.add_argument('-m', '--model', required=False, default=None, type=str,
                        help='Path to the model checkpoint to use')
    parser.add_argument('--directed', action='store_true', help='Provide if the graph is directed')
    parser.add_argument('--bipartite', action='store_true', help='Provide if the graph is bipartite')
    parser.add_argument('--cuda', action='store_true', help='Use cuda for GPU utilization')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the directory where the model checkpoints, final model and results are saved to.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train the model for.')

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
                                                directed=args.directed, bipartite=args.bipartite)

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

    tgn_wrapper = TGNWrapper(tgn, dataset, num_hops=2, model_name=name, device=device, n_neighbors=20, batch_size=128)

    model_path = args.model_path
    checkpoints_path = os.path.join(model_path, 'checkpoints/')
    if not os.path.exists(checkpoints_path):
        os.mkdir(checkpoints_path)

    tgn_wrapper.train_model(args.epochs, checkpoint_path=checkpoints_path, model_path=model_path,
                            results_path=model_path + '/results.pkl')
