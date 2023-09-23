import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch

from CFTGNNExplainer.data.dataset import ContinuousTimeDynamicGraphDataset
from TTGN.model.tgn import TGN
from CFTGNNExplainer.sampling.embedding import StaticEmbedding
from CFTGNNExplainer.baseline.ttgnbridge import TTGNBridge
from CFTGNNExplainer.baseline.ttgnwrapper import TTGNWrapper
from CFTGNNExplainer.baseline.pgexplainer import TPGExplainer
from TTGN.utils.utils import get_neighbor_finder

if __name__ == '__main__':
    parser = argparse.ArgumentParser('PGExplainer Baseline Training')

    parser.add_argument('-d', '--dataset', required=True, type=str, help='Path to the dataset folder')
    parser.add_argument('-m', '--model', required=True, type=str,
                        help='Path to the model checkpoint to use')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the directory where the model checkpoints, final model and results are saved to.')
    parser.add_argument('--directed', action='store_true', help='Provide if the graph is directed')
    parser.add_argument('--bipartite', action='store_true', help='Provide if the graph is bipartite')
    parser.add_argument('--cuda', action='store_true', help='Use cuda for GPU utilization')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train the model for.')
    parser.add_argument('--candidates_size', type=int, default=30,
                        help='Number of candidates for the explanation.')

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

    tgn_wrapper = TTGNWrapper(tgn, dataset, num_hops=2, model_name=name, device=device, n_neighbors=20, batch_size=128,
                              checkpoint_path=args.model)
    candidates_size = args.candidates_size
    bridge = TTGNBridge(tgn_wrapper, explanation_candidates_size=candidates_size)

    explainer = TPGExplainer(bridge, StaticEmbedding(dataset, tgn_wrapper), device=device)

    explainer.train(epochs=args.epochs, learning_rate=0.0001, batch_size=16, model_name=name,
                    save_directory=args.model_path)
