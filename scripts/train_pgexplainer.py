import argparse

from common import add_dataset_arguments, add_wrapper_model_arguments, create_tgn_wrapper_from_args, parse_args

from CFTGNNExplainer.sampling.embedding import StaticEmbedding
from CFTGNNExplainer.baseline.ttgnbridge import TTGNBridge
from CFTGNNExplainer.baseline.pgexplainer import TPGExplainer

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

    tgn_wrapper = create_tgn_wrapper_from_args(args)

    candidates_size = args.candidates_size
    bridge = TTGNBridge(tgn_wrapper, explanation_candidates_size=candidates_size)

    explainer = TPGExplainer(bridge, StaticEmbedding(tgn_wrapper.dataset, tgn_wrapper), device=tgn_wrapper.device)

    explainer.train(epochs=args.epochs, learning_rate=0.0001, batch_size=16, model_name=tgn_wrapper.name,
                    save_directory=args.model_path)
