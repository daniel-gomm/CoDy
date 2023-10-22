import argparse

import torch

from TTGN.utils.utils import get_neighbor_finder
from common import add_dataset_arguments, add_wrapper_model_arguments, create_dataset_from_args, parse_args

from CFTGNNExplainer.embedding import StaticEmbedding
from CFTGNNExplainer.implementations.ttgn import TTGNBridge, TTGNWrapper
from CFTGNNExplainer.implementations.tgn import to_data_object
from CFTGNNExplainer.explainer.baseline.pgexplainer import TPGExplainer
from TTGN.model.tgn import TGN

if __name__ == '__main__':
    parser = argparse.ArgumentParser('PGExplainer Baseline Training')
    add_dataset_arguments(parser)
    add_wrapper_model_arguments(parser)
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the directory where the model checkpoints, final model and results are saved to.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train the model for.')
    parser.add_argument('--candidates_size', type=int, default=30,
                        help='Number of candidates for the explanation.')

    args = parse_args(parser)

    dataset = create_dataset_from_args(args)

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
                              batch_size=32, checkpoint_path=args.model)

    candidates_size = args.candidates_size
    bridge = TTGNBridge(tgn_wrapper, explanation_candidates_size=candidates_size)

    explainer = TPGExplainer(bridge, StaticEmbedding(tgn_wrapper.dataset, tgn_wrapper), device=tgn_wrapper.device)

    explainer.train(epochs=args.epochs, learning_rate=0.0001, batch_size=16, model_name=tgn_wrapper.name,
                    save_directory=args.model_path)
