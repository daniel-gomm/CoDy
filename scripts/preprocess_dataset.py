import argparse
import sys
import pandas as pd
import numpy as np
from cody.constants import COL_ID, COL_STATE, COL_NODE_I, COL_NODE_U, COL_TIMESTAMP


def _check_required_columns(dataset: pd.DataFrame) -> None:
    """
    Check if all required columns are included in the dataset. If not raise an error.
    :param dataset: dataset to check for completeness
    """
    missing_columns = [column for column in [COL_NODE_I, COL_NODE_U, COL_TIMESTAMP, COL_STATE] if
                       column not in dataset.columns.to_list()]
    if missing_columns:
        raise KeyError(f'Cannot process the input because the column{"s" if len(missing_columns) > 1 else ""} '
                       f'{missing_columns} are missing in the input')


def _reindex_vertices(dataset: pd.DataFrame, bipartite: bool) -> pd.DataFrame:
    """
    Re-indexes the dataset such that all i nodes are different from all u nodes
    :param dataset: dataset to reindex
    :param bipartite: Set to true if sets of connected nodes are bipartite, false otherwise
    :rtype: DataFrame
    """
    dataset_copy = dataset.copy()

    if bipartite:
        assert (dataset[COL_NODE_I].max() - dataset[COL_NODE_I].min() + 1 == len(dataset[COL_NODE_I].unique()))
        assert (dataset[COL_NODE_U].max() - dataset[COL_NODE_U].min() + 1 == len(dataset[COL_NODE_U].unique()))
        dataset_copy[COL_NODE_I] = dataset_copy[COL_NODE_I] - dataset_copy[COL_NODE_I].min()  # Make zero indexed
        dataset_copy[COL_NODE_U] = dataset_copy[COL_NODE_U] - dataset_copy[COL_NODE_U].min()
        dataset_copy[COL_NODE_I] += dataset[COL_NODE_U].max() + 1

    return dataset_copy


def preprocess_data(dataset_path: str, output_directory: str, bipartite: bool = True,
                    node_features_path: str = None) -> None:
    """
    Preprocesses the provided dataset as csv file, extracting features and events from the input data
    :param bipartite: Set to true if sets of connected nodes are bipartite, false otherwise
    :param dataset_path: path of the dataset file. Dataset is expected to be in .csv format with the following columns:
            i (first node in interaction) | u (second node in interaction) | ts (timestamp) | label (0 for edge
            addition, 1 for deletion) | f_1 (feature 1) | f_2 (feature 2) | f_3 | ...
    :param output_directory: path to the directory for saving the preprocessed dataset to
    :param node_features_path: path of the file with the node features. Only provide if there are node features
    """
    dataset_name = dataset_path.split('/')[-1][:-4]

    u_list, i_list, ts_list, label_list, feat_list, idx_list = [], [], [], [], [], []

    with open(dataset_path) as f:
        next(f)  # Skip the first row, since it contains the headers
        for idx, line in enumerate(f):
            e = line.strip().split(',')
            features = np.array([float(x) for x in e[4:]])
            u_list.append(int(e[0]))
            i_list.append(int(e[1]))
            ts_list.append(float(e[2]))
            label_list.append(float(e[3]))
            idx_list.append(idx)
            feat_list.append(features)

    dataset = pd.DataFrame({COL_NODE_U: u_list,
                            COL_NODE_I: i_list,
                            COL_TIMESTAMP: ts_list,
                            COL_STATE: label_list,
                            COL_ID: idx_list})

    edge_features = np.array(feat_list)

    # Convert datatypes to best fitting (e.g., float -> integer if only integer values in column)
    dataset = dataset.convert_dtypes()

    _check_required_columns(dataset)

    dataset = _reindex_vertices(dataset, bipartite=bipartite)

    if node_features_path is not None:
        node_features = np.genfromtxt(node_features_path, delimiter=',', skip_header=True)
    else:
        max_idx = max(dataset[COL_NODE_I].max(), dataset[COL_NODE_U].max())
        node_features = np.zeros((max_idx + 1, 172))

    assert len(edge_features) == len(dataset)
    assert len(node_features) == max(dataset[COL_NODE_I].max(), dataset[COL_NODE_U].max()) + 1

    print(f'Dataset {dataset_name} has been processed and will be saved.\n\nDataset information:\nEdge feature shape: '
          f'{edge_features.shape}\nNode feature shape: {node_features.shape}')

    data_output_path = f'{output_directory.rstrip("/")}/{dataset_name}_data.csv'
    edge_features_output_path = f'{output_directory.rstrip("/")}/{dataset_name}_edge_features.npy'
    node_features_output_path = f'{output_directory.rstrip("/")}/{dataset_name}_node_features.npy'

    dataset[[COL_NODE_I, COL_NODE_U, COL_TIMESTAMP, COL_STATE, COL_ID]].to_csv(data_output_path, index=False)
    np.save(edge_features_output_path, edge_features)
    np.save(node_features_output_path, node_features)

    print(f'Successfully saved the preprocessed dataset to {output_directory}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Dataset preprocessor')

    parser.add_argument('-f', '--filepath', type=str, required=True, help='Path to the dataset csv file')
    parser.add_argument('-t', '--target', type=str, required=True,
                        help='Directory to which the processed dataset is to be stored to')
    parser.add_argument('-n', '--node_features', type=str, default=None,
                        help='Path to file with node features')
    parser.add_argument('-b', '--bipartite', action='store_true', help='Whether the graph is bipartite')

    try:
        args = parser.parse_args()
    except SystemExit:
        parser.print_help()
        sys.exit(0)

    filepath = args.filepath
    save_dir = args.target

    node_features_filepath = args.node_features
    is_bipartite = args.bipartite

    preprocess_data(filepath, save_dir, is_bipartite, node_features_filepath)
