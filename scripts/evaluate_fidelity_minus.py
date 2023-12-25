import argparse
import os.path
from os import listdir

import pandas as pd
import numpy as np

from cody.implementations.connector import TGNNWrapper
from cody.data import TrainTestDatasetParameters
from cody.utils import ProgressBar
from common import (add_dataset_arguments, add_wrapper_model_arguments, create_dataset_from_args,
                    create_tgn_wrapper_from_args, parse_args)


def add_prediction_for_dataframe(results_df: pd.DataFrame, tgnn: TGNNWrapper):
    if 'explanation_event_ids' in results_df.columns:
        explanation_event_ids = 'explanation_event_ids'
    else:
        explanation_event_ids = 'cf_example_event_ids'

    progress_bar = ProgressBar(len(results_df), 'Predicting...')

    predictions = []
    for index, row in results_df.iterrows():
        progress_bar.next()
        event_ids = row[explanation_event_ids]
        event_ids = np.array(event_ids, dtype=int)
        explained_event_id = row['explained_event_id']
        tgnn.reset_model()
        edges_to_drop = dataset.edge_ids[~np.isin(dataset.edge_ids,
                                                  np.concatenate([event_ids, np.array([explained_event_id])]))]
        prediction, _ = tgnn.compute_edge_probabilities_for_subgraph(explained_event_id, edges_to_drop, True,
                                                                     event_ids)
        predictions.append(prediction.detach().cpu().item())

    progress_bar.close()
    results_df['prediction_explanation_events_only'] = predictions

    return results_df


def save_results_to_file(results_df: pd.DataFrame, raw_filepath: str):
    results_df.to_csv(f'{raw_filepath}csv')
    print(f'Results saved to {raw_filepath}csv')
    try:
        results_df.to_parquet(f'{raw_filepath}parquet')
        print(f'Results saved to {raw_filepath}parquet')
    except ModuleNotFoundError:
        print('Could not save to parquet format. Install pyarrow if you want to export to parquet format')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluate Fidelity+')
    add_dataset_arguments(parser)
    add_wrapper_model_arguments(parser)
    parser.add_argument('--results', required=True, type=str,
                        help='Path to the file containing the evaluation results')
    parser.add_argument('--all_files_in_dir', action='store_true',
                        help='If provided, iterate over all files in the results directory and infer prediction scores')

    args = parse_args(parser)

    dataset = create_dataset_from_args(args, TrainTestDatasetParameters(0.2, 0.6, 0.8, 500,
                                                                        500, 500))
    tgn_wrapper = create_tgn_wrapper_from_args(args, dataset)
    tgn_wrapper.set_evaluation_mode(True)

    if not os.path.exists(args.results):
        raise FileNotFoundError('Failed to locate the file containing the results')

    dataframes = []
    if args.all_files_in_dir:
        results_files = [f for f in listdir(args.results) if os.path.isfile(os.path.join(args.results, f)) and f.endswith('parquet')]
        for results_file in results_files:
            filepath = f'{args.results}/{results_file}'
            results = pd.read_parquet(filepath)
            dataframes.append((results, filepath.rstrip('parquet')))
    else:
        if args.results.endswith('parquet'):
            results = pd.read_parquet(args.results)
            dataframes.append((results, args.results.rstrip('parquet')))
        else:
            raise RuntimeError('Cannot read results. Only parquet files supported.')

    for results, filename in dataframes:
        results = add_prediction_for_dataframe(results, tgn_wrapper)
        save_results_to_file(results, filename)
