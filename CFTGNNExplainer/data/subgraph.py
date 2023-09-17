import numpy as np
import pandas as pd
from CFTGNNExplainer.constants import COL_NODE_I, COL_NODE_U, COL_ID
from CFTGNNExplainer.data.dataset import ContinuousTimeDynamicGraphDataset


def _extract_center_node_ids(subgraph_events: pd.DataFrame, base_event_ids: [int], directed: bool = False):
    # Ids of the nodes that are involved in the base events
    center_node_ids = list(subgraph_events[subgraph_events[COL_ID].isin(base_event_ids)][COL_NODE_I].values)
    if not directed:
        # take both source and target side as center nodes in the undirected case
        center_node_ids.extend(
            list(subgraph_events[subgraph_events[COL_ID].isin(base_event_ids)][COL_NODE_U].values)
        )
    return center_node_ids


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

        neighboring_nodes = np.unique(np.concatenate([np.array(nodes) for nodes in reached_nodes]))

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

    def get_fixed_size_k_hop_temporal_subgraph(self, num_hops: int, base_event_id: int, size: int,
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

    def _get_next_hop_neighbors(self, reached_nodes: [int], source_nodes: np.ndarray, target_nodes: np.ndarray,
                                node_mask: np.ndarray) -> [int]:
        node_mask.fill(False)
        node_mask[np.array(reached_nodes)] = True
        source_target_edge_mask = node_mask[source_nodes]
        new_nodes_reached = target_nodes[source_target_edge_mask]
        if not self.directed:
            target_source_edge_mask = node_mask[target_nodes]
            new_source_nodes_reached = source_nodes[target_source_edge_mask]
            new_nodes_reached = np.concatenate((new_source_nodes_reached, new_nodes_reached))

        return np.unique(new_nodes_reached).tolist()
