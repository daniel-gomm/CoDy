import argparse
import os.path

import numpy as np
import pandas as pd

from cody.data import TrainTestDatasetParameters
from cody.utils import ProgressBar
from common import (add_dataset_arguments, add_wrapper_model_arguments, create_dataset_from_args,
                    create_tgn_wrapper_from_args, parse_args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Explainer Evaluation')
    add_dataset_arguments(parser)
    add_wrapper_model_arguments(parser)
    parser.add_argument('--results', required=True, type=str,
                        help='Path to the file containing the results of the factual explainer evaluation')

    args = parse_args(parser)

    dataset = create_dataset_from_args(args, TrainTestDatasetParameters(0.2, 0.6, 0.8, 500,
                                                                        500, 500))

    tgn_wrapper = create_tgn_wrapper_from_args(args, dataset)
    tgn_wrapper.set_evaluation_mode(True)

    if not os.path.exists(args.results):
        raise FileNotFoundError('Failed to locate the file containing the results')

    if args.results.endswith('parquet'):
        raw_filepath = args.results.rstrip('parquet')
        results_df = pd.read_parquet(args.results)
    else:
        raw_filepath = args.results.rstrip('csv')
        results_df = pd.read_csv(args.results)
        results_df = results_df.iloc[:, 1:]

    explanation_events = []
    explanation_sizes = []
    scores = []
    progress_bar = ProgressBar(len(results_df), 'Adding counterfacual information')
    for index, row in results_df.iterrows():
        progress_bar.next()
        # res_str = row['results'].replace("\'", '\"')
        # results = json.loads(res_str)
        results = row['results'].tolist()

        best_prediction = results[0]['prediction']
        best_result = results[0]
        for result in results:
            if result['prediction'] > best_prediction:
                best_prediction = result['prediction']
                best_result = result
        explanation_events.append(best_result['event_ids_in_explanation'])
        explanation_sizes.append(len(best_result['event_ids_in_explanation']))

        score, _ = tgn_wrapper.compute_edge_probabilities_for_subgraph(row['explained_event_id'],
                                                                       edges_to_drop=np.array(
                                                                           best_result['event_ids_in_explanation']),
                                                                       result_as_logit=True)
        tgn_wrapper.reset_model()
        scores.append(score.detach().cpu().item())

    progress_bar.close()
    results_df['explanation_event_ids'] = explanation_events
    results_df['explanation_size'] = explanation_sizes
    results_df['sparsity'] = results_df['explanation_size'] / results_df['candidate_size']
    results_df['counterfactual_prediction'] = scores

    results_df.to_csv(f'{raw_filepath}csv')
    print(f'Results saved to {raw_filepath}csv')
    try:
        results_df.to_parquet(f'{raw_filepath}parquet')
        print(f'Results saved to {raw_filepath}parquet')
    except ModuleNotFoundError:
        print('Could not save to parquet format. Install pyarrow if you want to export to parquet format')
