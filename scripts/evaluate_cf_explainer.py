import argparse
import logging
import time
import os
from typing import List

import numpy as np
import pandas as pd

from CFTGNNExplainer.data import TrainTestDatasetParameters
from CFTGNNExplainer.embedding import DynamicEmbedding, StaticEmbedding
from CFTGNNExplainer.sampler import create_embedding_model, PretrainedEdgeSamplerParameters
from common import (add_dataset_arguments, add_wrapper_model_arguments, create_dataset_from_args,
                    create_tgn_wrapper_from_args, parse_args, get_event_ids_from_file, SAMPLERS, column_to_int_array,
                    column_to_float_array)

from scripts.evaluation_explainers import EvaluationExplainer, EvaluationCounterFactualExample, \
    EvaluationGreedyCFExplainer, EvaluationSearchingCFExplainer, EvaluationCFTGNNExplainer
import scripts.evaluation_explainers
from CFTGNNExplainer.utils import ProgressBar

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def evaluate(evaluated_explainers: List[EvaluationExplainer], explained_event_ids: np.ndarray, optimize: bool = False,
             max_time_seconds: int = 72 * 60):
    assert len(evaluated_explainers) > 0
    progress_bar = ProgressBar(len(explained_event_ids), prefix='Evaluating explainer')
    last_event_id = np.min(explained_event_ids) - 1
    start_time = time.time()

    for event_id in explained_event_ids:
        progress_bar.update_postfix(f'Generating original score for event {event_id}')
        if time.time() - start_time > max_time_seconds:
            logger.info("Time limit reached. Finishing evaluation...")
            break
        if optimize:
            original_prediction = evaluated_explainers[0].get_evaluation_original_prediction(event_id, last_event_id)
            evaluated_explainers[0].tgnn.reset_model()
        else:
            original_prediction = None
        progress_bar.update_postfix(f'Generating explanation for event {event_id}')
        for selected_explainer in evaluated_explainers:
            explanation = selected_explainer.evaluate_explanation(event_id, original_prediction)
            selected_explainer.explanation_results_list.append(explanation)
            # Set the original prediction in the first iteration so that it does not have to be calculated again
            original_prediction = explanation.original_prediction
        scripts.evaluation_explainers.EVALUATION_STATE_CACHE = {}  # Reset the state cache
        last_event_id = event_id - 1
        progress_bar.next()
    progress_bar.close()


def export_explanations(explanation_list: List[EvaluationCounterFactualExample], filepath: str):
    explanations_dicts = [explanation.to_dict() for explanation in explanation_list]
    explanations_df = pd.DataFrame(explanations_dicts)
    parquet_file_path = filepath.rstrip('csv') + 'parquet'
    if os.path.exists(parquet_file_path):
        existing_results = pd.read_parquet(parquet_file_path)
        explanations_df = pd.concat([existing_results, explanations_df], axis='rows')
    elif os.path.exists(filepath):
        existing_results = pd.read_csv(filepath)
        existing_results = existing_results.iloc[:, 1:]
        column_to_int_array(existing_results, 'cf_example_event_ids')
        column_to_int_array(existing_results, 'candidates')
        column_to_float_array(existing_results, 'cf_example_absolute_importances')
        column_to_float_array(existing_results, 'cf_example_raw_importances')
        explanations_df = pd.concat([existing_results, explanations_df], axis='rows')
    try:
        explanations_df.to_parquet(parquet_file_path)
        logger.info(f'Saved evaluation results to {parquet_file_path}')
    except ImportError:
        logger.info('Failed to export to parquet format. Install pyarrow to export to parquet format '
                    '(pip install pyarrow)')
    explanations_df.to_csv(filepath)
    logger.info(f'Saved evaluation results to {filepath}')


def construct_results_save_path(arguments: argparse.Namespace, eval_explainer: EvaluationExplainer):
    if arguments.wrong_predictions_only:
        return (f'{arguments.results}/results_{eval_explainer.dataset.name}_{arguments.explainer}'
                f'_{eval_explainer.sampling_strategy}_wrong_only.csv')
    return (f'{arguments.results}/results_{eval_explainer.dataset.name}_{arguments.explainer}'
            f'_{eval_explainer.sampling_strategy}.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Explainer Evaluation')
    add_dataset_arguments(parser)
    add_wrapper_model_arguments(parser)
    parser.add_argument('--explained_ids', required=True, type=str,
                        help='Path to the file containing all the event ids that should be explained')
    parser.add_argument('--wrong_predictions_only', action='store_true',
                        help='Provide if evaluation should focus on wrong predictions only')
    parser.add_argument('--debug', action='store_true',
                        help='Add this flag for more detailed debug outputs')
    parser.add_argument('--optimize', action='store_true',
                        help='Add this flag to optimize evaluation performance at the cost of a bit of accuracy '
                             '(activate for debugging only)')
    parser.add_argument('-r', '--results', required=True, type=str,
                        help='Filepath for the evaluation results')
    parser.add_argument('--explainer', required=True, type=str, help='Which explainer to evaluate',
                        choices=['greedy', 'searching', 'cftgnnexplainer'])
    parser.add_argument('--sampler', required=True, default='recent', type=str,
                        choices=['random', 'recent', 'closest', 'pretrained', '1-best', 'all'])
    parser.add_argument('--sampler_model_path', default=None, type=str,
                        help='Path to the pretrained sampler model')
    parser.add_argument('--dynamic', action='store_true',
                        help='Provide to indicate that dynamic embeddings should be used')
    parser.add_argument('--predict_for_each_sample', action='store_true',
                        help='Provide if a the pretrained sampler should predict a delta for each sample separately')
    parser.add_argument('--sample_size', type=int, default=10,
                        help='Number of samples to draw in each sampling step')
    parser.add_argument('--candidates_size', type=int, default=64,
                        help='Number of candidates from which the samples are selected')
    parser.add_argument('--number_of_explained_events', type=int, default=1000,
                        help='Number of event ids to explain. Only has an effect if the explained_ids file has not '
                             'been initialized yet')
    parser.add_argument('--max_time', type=int, default=2400,
                        help='Maximal runtime (minutes)')
    parser.add_argument('--max_steps', type=int, default=100,
                        help='Maximum number of search steps to perform.')
    parser.add_argument('--no_approximation', action='store_true',
                        help='Provide if approximation should be disabled')

    args = parse_args(parser)

    dataset = create_dataset_from_args(args, TrainTestDatasetParameters(0.2, 0.6, 0.8, args.number_of_explained_events,
                                                                        500, 500))

    tgn_wrapper = create_tgn_wrapper_from_args(args, dataset)

    event_ids_to_explain = get_event_ids_from_file(args.explained_ids, logger, args.wrong_predictions_only,
                                                   tgn_wrapper)

    sampler_params = None

    if args.sampler == 'pretrained':
        if args.dynamic:
            embedding = DynamicEmbedding(dataset, tgn_wrapper, embed_static_node_features=False)
        else:
            embedding = StaticEmbedding(dataset, tgn_wrapper)

        pretrained_sampler_model = create_embedding_model(embedding, args.sampler_model_path, tgn_wrapper.device)
        sampler_params = PretrainedEdgeSamplerParameters(pretrained_sampler_model, embedding,
                                                         predict_for_each_sample=args.predict_for_each_sample)
    explainers = []
    match args.explainer:
        case 'greedy':
            if args.sampler == 'all':
                for sampler in SAMPLERS:
                    explainers.append(EvaluationGreedyCFExplainer(tgn_wrapper, sampling_strategy=sampler,
                                                                  candidates_size=args.candidates_size,
                                                                  sample_size=args.sample_size,
                                                                  pretrained_sampler_parameters=sampler_params,
                                                                  verbose=args.debug,
                                                                  approximate_predictions=not args.no_approximation))
            else:
                explainers.append(EvaluationGreedyCFExplainer(tgn_wrapper, sampling_strategy=args.sampler,
                                                              candidates_size=args.candidates_size,
                                                              sample_size=args.sample_size,
                                                              pretrained_sampler_parameters=sampler_params,
                                                              verbose=args.debug,
                                                              approximate_predictions=not args.no_approximation))
        case 'searching':
            if args.sampler == 'all':
                for sampler in SAMPLERS:
                    explainers.append(
                        EvaluationSearchingCFExplainer(tgn_wrapper, sampling_strategy=sampler,
                                                       candidates_size=args.candidates_size,
                                                       sample_size=args.sample_size, verbose=args.debug,
                                                       pretrained_sampler_parameters=sampler_params,
                                                       approximate_predictions=not args.no_approximation))
            else:
                explainers.append(EvaluationSearchingCFExplainer(tgn_wrapper, sampling_strategy=args.sampler,
                                                                 max_steps=args.max_steps,
                                                                 candidates_size=args.candidates_size,
                                                                 sample_size=args.sample_size, verbose=args.debug,
                                                                 pretrained_sampler_parameters=sampler_params,
                                                                 approximate_predictions=not args.no_approximation))
        case 'cftgnnexplainer':
            if args.sampler == 'all':
                for sampler in SAMPLERS:
                    explainers.append(EvaluationCFTGNNExplainer(tgn_wrapper, sampling_strategy=sampler,
                                                                candidates_size=args.candidates_size,
                                                                max_steps=args.max_steps, verbose=args.debug,
                                                                pretrained_sampler_parameters=sampler_params,
                                                                approximate_predictions=not args.no_approximation))
            else:
                explainers.append(EvaluationCFTGNNExplainer(tgn_wrapper, sampling_strategy=args.sampler,
                                                            candidates_size=args.candidates_size,
                                                            max_steps=args.max_steps, verbose=args.debug,
                                                            pretrained_sampler_parameters=sampler_params,
                                                            approximate_predictions=not args.no_approximation))
        case _:
            raise NotImplementedError

    if os.path.exists(construct_results_save_path(args, explainers[0])):
        previous_results = pd.read_csv(construct_results_save_path(args, explainers[0]))
        encountered_event_ids = previous_results['explained_event_id'].to_numpy()
        logger.info(f'Resuming evaluation. '
                    f'Already processed {len(encountered_event_ids)}/{len(event_ids_to_explain)} events.')
        event_ids_to_explain = event_ids_to_explain[~np.isin(event_ids_to_explain, encountered_event_ids)]

    evaluate(explainers, event_ids_to_explain, args.optimize, args.max_time * 60)
    for explainer in explainers:
        export_explanations(explainer.explanation_results_list, construct_results_save_path(args, explainer))
