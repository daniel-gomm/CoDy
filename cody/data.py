import pandas as pd
import numpy as np
from cody.constants import COL_NODE_I, COL_NODE_U, COL_TIMESTAMP, COL_ID, COL_STATE, COL_SUBGRAPH_DISTANCE
from dataclasses import dataclass


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
        assert self.edge_ids[0] == 0, 'Event ids should start with zero index'
        assert len(np.unique(self.edge_ids)) == len(self.edge_ids), 'All event ids should be unique'
        assert self.edge_ids[-1] == len(self.edge_ids) - 1, 'Some event ids might be missing or duplicates'
        self.labels = self.events[COL_STATE].to_numpy()

    def get_batch_data(self, start_index: int, end_index: int) -> BatchData:
        """
        Get batch data as numpy arrays.
        @param start_index: Index of the first event in the batch.
        @param end_index: Index of the last event in the batch.
        @return: (source node ids, target node ids, timestamps, edge ids)
        """
        return BatchData(self.source_node_ids[start_index:end_index], self.target_node_ids[start_index:end_index],
                         self.timestamps[start_index:end_index], self.edge_ids[start_index:end_index])

    def extract_random_event_ids(self, section: str = 'train') -> [int]:
        """
        Create a random set of event ids
        @param section: section from which ids should be extracted, options: 'train', 'validation', 'test'
        @return: Ordered random set of event ids in a specified range.
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


def _extract_center_node_ids(subgraph_events: pd.DataFrame, base_event_ids: [int], directed: bool = False) \
        -> np.ndarray:
    # Ids of the nodes that are involved in the base events
    center_node_ids = list(subgraph_events[subgraph_events[COL_ID].isin(base_event_ids)][COL_NODE_I].values)
    if not directed:
        # take both source and target side as center nodes in the undirected case
        center_node_ids.extend(
            list(subgraph_events[subgraph_events[COL_ID].isin(base_event_ids)][COL_NODE_U].values)
        )
    return np.array(center_node_ids)


class SubgraphGenerator:
    all_events: pd.DataFrame

    def __init__(self, dataset: ContinuousTimeDynamicGraphDataset):
        self.directed = dataset.directed
        self.all_events = dataset.events

    def _prepare_subgraph(self, base_event_id: int) -> (pd.DataFrame, int):
        subgraph_events = self.all_events.copy()

        # Make ids indexed to 0
        lowest_id = np.min((subgraph_events[COL_NODE_I].min(), subgraph_events[COL_NODE_U].min()))
        subgraph_events[COL_NODE_I] -= lowest_id
        subgraph_events[COL_NODE_U] -= lowest_id
        # Filter out events happening after the base event
        subgraph_events = subgraph_events[subgraph_events[COL_ID] <= base_event_id]

        return subgraph_events, lowest_id

    def get_k_hop_temporal_subgraph(self, num_hops: int, base_event_id: int = None,
                                    base_event_ids: list[int] = None) -> pd.DataFrame:
        # TODO: Test if it works with directed graph as well
        if base_event_ids is None:
            if base_event_id:
                base_event_ids = [base_event_id]
            else:
                raise Exception('Missing base event. Provide either a base_event_id or a list of base_event_ids.')
        subgraph_events, lowest_id = self._prepare_subgraph(max(base_event_ids))

        center_node_ids = _extract_center_node_ids(subgraph_events, base_event_ids, self.directed)

        unique_nodes = sorted(pd.concat((subgraph_events[COL_NODE_I], subgraph_events[COL_NODE_U])).unique())

        node_mask = np.zeros((np.max(unique_nodes) + 1,), dtype=bool)
        source_nodes = np.array(subgraph_events.loc[:, COL_NODE_I], dtype=int)
        target_nodes = np.array(subgraph_events.loc[:, COL_NODE_U], dtype=int)

        reached_nodes = [center_node_ids, ]

        for _ in range(num_hops):
            # Iteratively explore the neighborhood of the base nodes
            reached_nodes.append(self._get_next_hop_neighbors(reached_nodes[-1], source_nodes, target_nodes, node_mask))

        neighboring_nodes = np.unique(np.concatenate(reached_nodes))

        distance_from_base_event = np.repeat(num_hops + 2, len(subgraph_events))  # Set default distance

        for index, nodes in enumerate(reached_nodes):
            if index > 0:
                nodes = nodes[~np.isin(nodes, reached_nodes[index - 1])]
            distance_from_base_event[subgraph_events[COL_NODE_I].isin(nodes)] = index
            distance_from_base_event[subgraph_events[COL_NODE_U].isin(nodes)] = index

        subgraph_events[COL_SUBGRAPH_DISTANCE] = distance_from_base_event

        node_mask.fill(False)
        node_mask[neighboring_nodes] = True

        source_mask = node_mask[source_nodes]
        target_mask = node_mask[target_nodes]

        edge_mask = source_mask & target_mask

        subgraph_events = subgraph_events.iloc[edge_mask, :].copy()

        # Restore the original node ids
        subgraph_events[COL_NODE_I] += lowest_id
        subgraph_events[COL_NODE_U] += lowest_id

        return subgraph_events

    def get_fixed_size_k_hop_temporal_subgraph(self, num_hops: int, base_event_id: int, size: int):
        candidate_events = self.get_k_hop_temporal_subgraph(num_hops, base_event_id=base_event_id)

        if len(candidate_events) > size:
            return candidate_events[-size:]
        return candidate_events

    def get_fixed_size_k_hop_temporal_forward_subgraph(self, num_hops: int, base_event_id: int, size: int,
                                                       directed: bool = False):
        candidate_events = self.get_k_hop_temporal_subgraph(num_hops, base_event_id=base_event_id)

        reached_nodes = _extract_center_node_ids(candidate_events, [base_event_id], directed)

        candidate_events['selected'] = False

        selected_events = candidate_events[candidate_events['selected']]

        while len(selected_events) < size:
            unselected_events = candidate_events[~candidate_events['selected']]

            new_event = unselected_events[unselected_events[COL_NODE_I].isin(reached_nodes) |
                                          (not directed and unselected_events[COL_NODE_U].isin(reached_nodes))].tail(1)
            if len(new_event) == 0:
                return selected_events.drop('selected', axis=1)
            candidate_events.at[new_event.index.item(), 'selected'] = True
            selected_events = candidate_events[candidate_events['selected']]
            reached_nodes = np.unique(np.concatenate((selected_events[COL_NODE_I].unique(), reached_nodes))).tolist()
            if not directed:
                target_reached_nodes = selected_events[COL_NODE_U].unique()
                reached_nodes = np.unique(np.concatenate((reached_nodes, target_reached_nodes))).tolist()

        return selected_events.drop('selected', axis=1)

    def _get_next_hop_neighbors(self, reached_nodes: np.ndarray, source_nodes: np.ndarray, target_nodes: np.ndarray,
                                node_mask: np.ndarray) -> np.ndarray:
        node_mask.fill(False)
        node_mask[reached_nodes] = True
        source_target_edge_mask = node_mask[source_nodes]
        new_nodes_reached = target_nodes[source_target_edge_mask]
        if not self.directed:
            target_source_edge_mask = node_mask[target_nodes]
            new_source_nodes_reached = source_nodes[target_source_edge_mask]
            new_nodes_reached = np.concatenate((new_source_nodes_reached, new_nodes_reached))

        return np.unique(new_nodes_reached)
