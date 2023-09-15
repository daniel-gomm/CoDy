import pandas as pd
import numpy as np
import random
from CFTGNNExplainer.constants import COL_NODE_I, COL_NODE_U, COL_TIMESTAMP, COL_ID, COL_STATE
from dataclasses import dataclass
from TGN.utils.data_processing import Data


@dataclass
class TrainTestDatasetParameters:
    training_start: float
    training_end: float
    validation_end: float
    train_items: int
    validation_items: int
    test_items: int


@dataclass
class BatchData:
    source_node_ids: np.ndarray
    target_node_ids: np.ndarray
    timestamps: np.ndarray
    edge_ids: np.ndarray


class ContinuousTimeDynamicGraphDataset:

    def __init__(self, events: pd.DataFrame, edge_features: np.ndarray, node_features: np.ndarray, name: str,
                 directed: bool = False, bipartite: bool = False,
                 parameters: TrainTestDatasetParameters = TrainTestDatasetParameters(0.2, 0.6, 0.8, 1000, 500, 500)):
        self.events = events
        self.edge_features = edge_features
        self.node_features = node_features
        self.bipartite = bipartite
        self.directed = directed
        self.name = name
        self.parameters = parameters
        self.source_node_ids = self.events[COL_NODE_U].to_numpy(dtype=int)
        self.target_node_ids = self.events[COL_NODE_I].to_numpy(dtype=int)
        self.timestamps = self.events[COL_TIMESTAMP].to_numpy(dtype=int)
        self.edge_ids = self.events[COL_ID].to_numpy(dtype=int)
        assert self.edge_ids[0] == 1, 'Event ids should be one indexed'
        assert len(np.unique(self.edge_ids)) == len(self.edge_ids), 'All event ids should be unique'
        assert self.edge_ids[-1] == len(self.edge_ids), 'Some event ids might be missing or duplicates'
        self.labels = self.events[COL_STATE].to_numpy()

    def get_batch_data(self, start_index: int, end_index: int) -> BatchData:
        """
        Get batch data as numpy arrays.
        :param start_index: Index of the first event in the batch.
        :param end_index: Index of the last event in the batch.
        :return: (source node ids, target node ids, timestamps, edge ids)
        """
        return BatchData(self.source_node_ids[start_index:end_index], self.target_node_ids[start_index:end_index],
                         self.timestamps[start_index:end_index], self.edge_ids[start_index:end_index])

    def to_data_object(self, edges_to_drop: np.ndarray = None) -> Data:
        """
        Convert the dataset to a data object that can be used as input for a neighborhood finder
        :param edges_to_drop: Edges that should be excluded from the data
        :return: Data object of the dataset
        """
        if edges_to_drop is not None:
            edge_mask = ~np.isin(self.edge_ids, edges_to_drop)
            return Data(self.source_node_ids[edge_mask], self.target_node_ids[edge_mask], self.timestamps[edge_mask],
                        self.edge_ids[edge_mask], self.labels[edge_mask])
        return Data(self.source_node_ids, self.target_node_ids, self.timestamps, self.edge_ids, self.labels)

    def extract_random_event_ids(self, section: str = 'train') -> [int]:
        """
        Create a random set of event ids
        :param section: section from which ids should be extracted, options: 'train', 'validation', 'test'
        :return: Ordered random set of event ids in a specified range.
        """
        if section == 'train':
            start = self.parameters.training_start
            end = self.parameters.training_end
            size = self.parameters.train_items
        elif section == 'validation':
            start = self.parameters.training_end
            end = self.parameters.validation_end
            size = self.parameters.validation_items
        elif section == 'test':
            start = self.parameters.validation_end
            end = 1
            size = self.parameters.test_items
        else:
            raise AttributeError(f'"{section}" is an unrecognized value for the "section" parameter.')
        assert 0 <= start < end <= 1
        return sorted(np.random.randint(int(len(self.events) * start), int(len(self.events) * end), (size,)))

    def get_training_data(self, randomize_features: bool = False, validation_fraction: float = 0.15,
                          test_fraction: float = 0.15, new_test_nodes_fraction: float = 0.1,
                          different_new_nodes_between_val_and_test: bool = False):
        # Function adapted from data_processing.py in https://github.com/twitter-research/tgn
        node_features = self.node_features
        if randomize_features:
            node_features = np.random.rand(self.node_features.shape[0], node_features.shape[1])

        val_time, test_time = list(np.quantile(self.events[COL_TIMESTAMP],
                                               [1 - (validation_fraction + test_fraction), 1 - test_fraction]))

        full_data = Data(self.source_node_ids, self.target_node_ids, self.timestamps, self.edge_ids, self.labels)

        node_set = set(self.source_node_ids) | set(self.target_node_ids)
        unique_nodes = len(node_set)

        # Compute nodes which appear at test time
        test_node_set = set(self.source_node_ids[self.timestamps > val_time]) \
            .union(set(self.target_node_ids[self.timestamps > val_time]))
        # Sample nodes which we keep as new nodes (to test inductiveness), so than we have to remove all
        # their edges from training
        new_test_node_set = set(random.sample(sorted(test_node_set), int(new_test_nodes_fraction * unique_nodes)))

        # Mask saying for each source and destination whether they are new test nodes
        new_test_source_mask = self.events[COL_NODE_I].map(lambda x: x in new_test_node_set).values
        new_test_destination_mask = self.events[COL_NODE_U].map(lambda x: x in new_test_node_set).values

        # Mask which is true for edges with both destination and source not being new test nodes (because
        # we want to remove all edges involving any new test node)
        observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)

        # For train, we keep edges happening before the validation time which do not involve any new node
        # used for inductiveness
        train_mask = np.logical_and(self.timestamps <= val_time, observed_edges_mask)

        train_data = Data(self.source_node_ids[train_mask], self.target_node_ids[train_mask],
                          self.timestamps[train_mask],
                          self.edge_ids[train_mask], self.labels[train_mask])

        # define the new nodes sets for testing inductiveness of the model
        train_node_set = set(train_data.sources).union(train_data.destinations)
        assert len(train_node_set & new_test_node_set) == 0
        new_node_set = node_set - train_node_set

        val_mask = np.logical_and(self.timestamps <= test_time, self.timestamps > val_time)
        test_mask = self.timestamps > test_time

        if different_new_nodes_between_val_and_test:
            n_new_nodes = len(new_test_node_set) // 2
            val_new_node_set = set(list(new_test_node_set)[:n_new_nodes])
            test_new_node_set = set(list(new_test_node_set)[n_new_nodes:])

            edge_contains_new_val_node_mask = np.array(
                [(a in val_new_node_set or b in val_new_node_set) for a, b in
                 zip(self.source_node_ids, self.target_node_ids)])
            edge_contains_new_test_node_mask = np.array(
                [(a in test_new_node_set or b in test_new_node_set) for a, b in
                 zip(self.source_node_ids, self.target_node_ids)])
            new_node_val_mask = np.logical_and(val_mask, edge_contains_new_val_node_mask)
            new_node_test_mask = np.logical_and(test_mask, edge_contains_new_test_node_mask)
        else:
            edge_contains_new_node_mask = np.array(
                [(a in new_node_set or b in new_node_set) for a, b in zip(self.source_node_ids, self.target_node_ids)])
            new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
            new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

        # validation and test with all edges
        val_data = Data(self.source_node_ids[val_mask], self.target_node_ids[val_mask], self.timestamps[val_mask],
                        self.edge_ids[val_mask], self.labels[val_mask])

        test_data = Data(self.source_node_ids[test_mask], self.target_node_ids[test_mask], self.timestamps[test_mask],
                         self.edge_ids[test_mask], self.labels[test_mask])

        # validation and test with edges that at least has one new node (not in training set)
        new_node_val_data = Data(self.source_node_ids[new_node_val_mask], self.target_node_ids[new_node_val_mask],
                                 self.timestamps[new_node_val_mask],
                                 self.edge_ids[new_node_val_mask], self.labels[new_node_val_mask])

        new_node_test_data = Data(self.source_node_ids[new_node_test_mask], self.target_node_ids[new_node_test_mask],
                                  self.timestamps[new_node_test_mask], self.edge_ids[new_node_test_mask],
                                  self.labels[new_node_test_mask])

        print("The dataset has {} interactions, involving {} different nodes".format(full_data.n_interactions,
                                                                                     full_data.n_unique_nodes))
        print("The training dataset has {} interactions, involving {} different nodes".format(
            train_data.n_interactions, train_data.n_unique_nodes))
        print("The validation dataset has {} interactions, involving {} different nodes".format(
            val_data.n_interactions, val_data.n_unique_nodes))
        print("The test dataset has {} interactions, involving {} different nodes".format(
            test_data.n_interactions, test_data.n_unique_nodes))
        print("The new node validation dataset has {} interactions, involving {} different nodes".format(
            new_node_val_data.n_interactions, new_node_val_data.n_unique_nodes))
        print("The new node test dataset has {} interactions, involving {} different nodes".format(
            new_node_test_data.n_interactions, new_node_test_data.n_unique_nodes))
        print("{} nodes were used for the inductive testing, i.e. are never seen during training".format(
            len(new_test_node_set)))

        return (node_features, self.edge_features, full_data, train_data, val_data, test_data, new_node_val_data,
                new_node_test_data)
