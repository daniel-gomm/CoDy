import argparse
import logging
from typing import List

import numpy as np
import pandas as pd

from CFTGNNExplainer.data.dataset import TrainTestDatasetParameters
from CFTGNNExplainer.sampling.embedding import DynamicEmbedding, StaticEmbedding
from CFTGNNExplainer.sampling.sampler import create_embedding_model, PretrainedEdgeSamplerParameters
from common import (add_dataset_arguments, add_wrapper_model_arguments, create_dataset_from_args,
                    create_tgn_wrapper_from_args, parse_args, get_event_ids_from_file)

from CFTGNNExplainer.connector.bridge import DynamicTGNNBridge
from CFTGNNExplainer.explainer.evaluation import EvaluationExplainer, EvaluationCounterFactualExample, \
    EvaluationGreedyCFExplainer, EvaluationSearchingCFExplainer
from CFTGNNExplainer.utils import ProgressBar

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def evaluate(evaluated_explainer: EvaluationExplainer, explained_event_ids: np.ndarray):
    explanation_list = []

    progress_bar = ProgressBar(len(explained_event_ids), prefix='Evaluating explainer')
    last_event_id = np.min(explained_event_ids) - 1

    for event_id in explained_event_ids:
        progress_bar.update_postfix(f'Generating original score for event {event_id}')
        original_prediction = evaluated_explainer.get_evaluation_original_prediction(event_id, last_event_id)
        evaluated_explainer.tgnn_bridge.reset_model()
        progress_bar.update_postfix(f'Generating explanation for event {event_id}')
        explanation = evaluated_explainer.evaluate_explanation(event_id, original_prediction)
        explanation_list.append(explanation)
        last_event_id = event_id - 1
        progress_bar.next()

    return explanation_list


def export_explanations(explanation_list: List[EvaluationCounterFactualExample], filepath: str):
    explanations_dicts = [explanation.to_dict() for explanation in explanation_list]
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
                        choices=['greedy', 'searching'])
    parser.add_argument('--sampler', required=True, default='recent', type=str,
                        choices=['random', 'recent', 'closest', 'pretrained'])
    parser.add_argument('--sampler_model_path', default=None, type=str,
                        help='Path to the pretrained sampler model')
    parser.add_argument('--dynamic', action='store_true',
                        help='Provide to indicate that dynamic embeddings should be used')
    parser.add_argument('--predict_for_each_sample', action='store_true',
                        help='Provide if a the pretrained sampler should predict a delta for each sample separately')
    parser.add_argument('--sample_size', type=int, default=10,
                        help='Number of samples to draw in each sampling step')
    parser.add_argument('--candidates_size', type=int, default=50,
                        help='Number of candidates from which the samples are selected')
    parser.add_argument('--number_of_explained_events', type=int, default=1000,
                        help='Number of event ids to explain. Only has an effect if the explained_ids file has not '
                             'been initialized yet')

    args = parse_args(parser)

    dataset = create_dataset_from_args(args, TrainTestDatasetParameters(0.2, 0.6, 0.8, args.number_of_explained_events,
                                                                        500, 500))

    tgn_wrapper = create_tgn_wrapper_from_args(args, dataset)

    event_ids_to_explain = get_event_ids_from_file(args.explained_ids, dataset, logger)

    sampler_params = None

    if args.sampler == 'pretrained':
        if args.dynamic:
            embedding = DynamicEmbedding(dataset, tgn_wrapper, embed_static_node_features=False)
        else:
            embedding = StaticEmbedding(dataset, tgn_wrapper)

        pretrained_sampler_model = create_embedding_model(embedding, args.sampler_model_path, tgn_wrapper.device)
        sampler_params = PretrainedEdgeSamplerParameters(pretrained_sampler_model, embedding,
                                                         predict_for_each_sample=args.predict_for_each_sample)

    match args.explainer:
        case 'greedy':
            explainer = EvaluationGreedyCFExplainer(DynamicTGNNBridge(tgn_wrapper), sampling_strategy=args.sampler,
                                                    candidates_size=args.candidates_size, sample_size=args.sample_size,
                                                    pretrained_sampler_parameters=sampler_params)
        case 'searching':
            explainer = EvaluationSearchingCFExplainer(DynamicTGNNBridge(tgn_wrapper), sampling_strategy=args.sampler,
                                                       candidates_size=args.candidates_size,
                                                       sample_size=args.sample_size,
                                                       pretrained_sampler_parameters=sampler_params)
        case _:
            raise NotImplementedError

    explanations = evaluate(explainer, event_ids_to_explain)
    export_explanations(explanations, args.results)
