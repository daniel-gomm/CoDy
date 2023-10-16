from __future__ import annotations

import sys
from typing import List

import numpy as np

from CFTGNNExplainer.connector.bridge import TGNNBridge
from CFTGNNExplainer.constants import EXPLAINED_EVENT_MEMORY_LABEL, COL_ID
from CFTGNNExplainer.explainer.base import Explainer, CounterFactualExample
from CFTGNNExplainer.explainer.searching import calculate_prediction_delta
from CFTGNNExplainer.sampling.sampler import PretrainedEdgeSamplerParameters, EdgeSampler


def find_best_non_counterfactual_example(root_node: MCTSTreeNode) -> MCTSTreeNode:
    """
        Breadth-first search for explanation that comes closest to counterfactual example
    """
    best_example = root_node
    nodes_to_visit = root_node.children.copy()
    while len(nodes_to_visit) != 0:
        explored_node = nodes_to_visit.pop()
        if explored_node.prediction is not None:
            if (calculate_prediction_delta(best_example.original_prediction, best_example.prediction) <
                    calculate_prediction_delta(explored_node.original_prediction, explored_node.prediction)):
                best_example = explored_node
        nodes_to_visit.extend(explored_node.children.copy())
    return best_example


class MCTSTreeNode:
    parent: MCTSTreeNode
    children: List[MCTSTreeNode]
    score: float
    number_of_selections: int
    is_counterfactual: bool
    edge_id: int
    sampling_rank: int
    prediction: float | None

    def __init__(self, edge_id: int, parent: MCTSTreeNode | None, original_prediction: float, sampling_rank: int):
        self.edge_id: int = edge_id
        self.parent = parent
        self.sampling_rank = sampling_rank
        self.original_prediction: float = original_prediction
        self.prediction = None
        self.is_counterfactual = False
        self.exploitation_score: float = 0
        self.number_of_selections: int = 1
        self.children = []
        if self.parent is None:
            self.depth = 0
        else:
            self.depth = self.parent.depth + 1
        self.expanded = False
        self.max_expansion_reached = False

    def expand(self, prediction: float):
        """
        Expand the node
        @param prediction: The prediction achieved when this tree node is included in the counterfactual example
        @return: None
        """
        self.prediction = prediction
        self.exploitation_score = max(0.0, (calculate_prediction_delta(self.original_prediction, self.prediction) /
                                            abs(self.original_prediction)))
        self.expanded = True
        if self.original_prediction * self.prediction < 0:
            self.is_counterfactual = True
            self.max_expansion_reached = True
        self.expansion_backpropagation()

    def _check_max_expanded(self):
        """
        Recursively check if the node is already maximally expanded, meaning that no further expansions of its child
        nodes are possible
        """
        if not self.max_expansion_reached:
            for child in self.children:
                if not child.max_expansion_reached or child.expanded:
                    return
            self.max_expansion_reached = True
            if self.parent is not None:
                self.parent._check_max_expanded()

    def _is_leaf(self) -> bool:
        """
        Check if the node is a leaf node, meaning that it has no children
        """
        return len(self.children) == 0

    def _calculate_score(self):
        """
        Calculate the search score which balances exploration with exploitation
        """
        exploration_score = np.sqrt(2) * np.sqrt(np.log(self.parent.number_of_selections) / self.number_of_selections)
        return self.exploitation_score + exploration_score

    def select_next_leaf(self, max_depth: int) -> MCTSTreeNode:
        """
        Select the next leaf node for expansion
        @param max_depth: Maximum depth at which to search for leaf nodes
        @return: Leaf node to expand
        """
        if self._is_leaf():
            return self
        if self.depth == max_depth:
            self.max_expansion_reached = True
            if self.parent is not None:
                self.parent._check_max_expanded()
                return self.parent.select_next_leaf(max_depth)
        selected_child = None
        best_score = 0
        candidate_children = [child for child in self.children if not (child.is_counterfactual or
                                                                       child.max_expansion_reached)]
        if max_depth == self.depth - 1:
            # If max depth is reached in the next level only consider children that can be directly expanded, otherwise
            #  the max depth requirement would be violated
            for child in self.children:
                child.max_expansion_reached = True
            candidate_children = [child for child in self.children if not child.expanded]
        for child in candidate_children:
            child_score = child._calculate_score()
            if child_score > best_score:
                best_score = child_score
                selected_child = child
        if selected_child is None:  # This means that there are no candidate children -> return oneself
            if self.expanded and self.parent is not None:
                self.max_expansion_reached = True
                self.parent._check_max_expanded()  # When no selection is possible the node is fully expanded
                return self.parent.select_next_leaf(max_depth)
            return self
        if not selected_child.expanded:
            # If a node that has not yet been expanded is selected then select the node from the unexpanded children
            # with the lowest sampling rank
            unexpanded_children = [child for child in self.children if not child.expanded]
            selected_child = min(unexpanded_children, key=lambda node: node.sampling_rank)
        return selected_child.select_next_leaf(max_depth)

    def expansion_backpropagation(self):
        """
        Propagate the information that a node is selected backwards and update scores
        """
        if not self._is_leaf():
            # Here the exploitation score could be updates. However, this seems to make the search performance worse

            # self.exploitation_score = max(0.0, np.average([child.exploitation_score for child in self.children
            #                                                if child.expanded]))
            pass

        self.number_of_selections += 1
        if self.parent is not None:
            self.parent.expansion_backpropagation()

    def to_cf_example(self) -> CounterFactualExample:
        """
        Returns an instance of CounterFactualExample for the current node by aggregating information from parents
        """
        cf_events = []
        cf_event_importances = []
        node = self
        while node.parent is not None:
            cf_events.append(node.edge_id)
            cf_event_importances.append(calculate_prediction_delta(self.original_prediction, node.prediction))
            node = node.parent
        cf_events.reverse()
        cf_event_importances.reverse()
        return CounterFactualExample(explained_event_id=node.edge_id,
                                     original_prediction=self.original_prediction,
                                     counterfactual_prediction=self.prediction,
                                     achieves_counterfactual_explanation=self.is_counterfactual,
                                     event_ids=np.array(cf_events),
                                     event_importances=np.array(cf_event_importances))

    def hash(self):
        edge_ids = []
        node = self
        while node.parent is not None:
            edge_ids.append(node.edge_id)
            node = node.parent
        sorted_edge_ids = sorted(edge_ids)
        return '-'.join(map(str, sorted_edge_ids))


class CFTGNNExplainer(Explainer):

    def __init__(self, tgnn_bridge: TGNNBridge, candidates_size: int = 75, sample_size: int = 10,
                 sampling_strategy: str = 'recent', max_steps: int = 50, verbose: bool = False,
                 pretrained_sampler_parameters: PretrainedEdgeSamplerParameters | None = None):
        super().__init__(tgnn_bridge, sampling_strategy, candidates_size=candidates_size, sample_size=sample_size,
                         verbose=verbose, pretrained_sampler_parameters=pretrained_sampler_parameters)
        self.max_steps = max_steps
        self.known_states = {}

    def _run_node_expansion(self, explained_edge_id: int, node_to_expand: MCTSTreeNode, sampler: EdgeSampler):
        edge_ids_to_exclude = []
        node = node_to_expand
        while node.parent is not None:
            edge_ids_to_exclude.append(node.edge_id)
            node = node.parent

        prediction = self.calculate_subgraph_prediction(candidate_events=sampler.subgraph[COL_ID],
                                                        cf_example_events=edge_ids_to_exclude,
                                                        explained_event_id=explained_edge_id,
                                                        candidate_event_id=node_to_expand.edge_id,
                                                        memory_label=EXPLAINED_EVENT_MEMORY_LABEL)

        self._expand_node(explained_edge_id, node_to_expand, prediction, sampler)

    def _expand_node(self, explained_edge_id: int, node_to_expand: MCTSTreeNode, prediction: float, sampler: EdgeSampler):
        node_to_expand.expand(prediction)

        self.known_states[node_to_expand.hash()] = prediction

        if node_to_expand.is_counterfactual:
            return

        edge_ids_to_exclude = []
        node = node_to_expand
        while node.parent is not None:
            edge_ids_to_exclude.append(node.edge_id)
            node = node.parent

        ranked_edge_ids = sampler.rank_subgraph(base_event_id=explained_edge_id,
                                                excluded_events=np.array(edge_ids_to_exclude))

        for rank, edge_id in enumerate(ranked_edge_ids):
            new_child = MCTSTreeNode(edge_id, node_to_expand, node_to_expand.original_prediction, rank)
            node_to_expand.children.append(new_child)
            if new_child.hash() in self.known_states.keys():
                self._expand_node(explained_edge_id, new_child, self.known_states[new_child.hash()], sampler)

    def explain(self, explained_event_id: int) -> CounterFactualExample:
        """
        Explain the prediction of the provided event id with a counterfactual example found by searching possible
        examples with an adapted 'Monte Carlo' Tree Search
        @param explained_event_id: The event id that is explained
        @return: The best CounterFactualExample found for explaining the event
        """
        original_prediction, sampler = self.initialize_explanation(explained_event_id)
        best_cf_example = None
        max_depth = sys.maxsize
        root_node = MCTSTreeNode(explained_event_id, parent=None, sampling_rank=0,
                                 original_prediction=original_prediction)
        root_node.prediction = original_prediction
        step = 0
        while step <= self.max_steps:
            node_to_expand = None
            while node_to_expand is None:
                node_to_expand = root_node.select_next_leaf(max_depth)
                if node_to_expand.hash() in self.known_states.keys():
                    # Already encountered this combination -> select new combination of events instead
                    self._expand_node(explained_event_id, node_to_expand, self.known_states[node_to_expand.hash()],
                                      sampler)
                    node_to_expand = None
            if node_to_expand.depth > max_depth:
                # Should not happen. TODO: Check if it is save to remove this if-condition
                node_to_expand.expansion_backpropagation()
                continue
            if node_to_expand == root_node and root_node.expanded:
                if self.verbose:
                    self.logger.info('Search Tree is fully expanded. Concluding search.')
                break  # No nodes are selectable, meaning that we can conclude the search
            if self.verbose:
                self.logger.info(f'Selected node {node_to_expand.edge_id} at depth {node_to_expand.depth}, hash: '
                                 f'{node_to_expand.hash()}')
            self._run_node_expansion(explained_event_id, node_to_expand, sampler)
            if node_to_expand.is_counterfactual:
                if best_cf_example is None or best_cf_example.depth > node_to_expand.depth:
                    best_cf_example = node_to_expand
                elif (best_cf_example.depth == node_to_expand.depth and
                      best_cf_example.exploitation_score < node_to_expand.exploitation_score):
                    best_cf_example = node_to_expand
                max_depth = best_cf_example.depth
                if self.verbose:
                    self.logger.info(f'Found counterfactual explanation: '
                                     + str(node_to_expand.to_cf_example()))
            step += 1
        if best_cf_example is None:
            best_cf_example = find_best_non_counterfactual_example(root_node)
        self.tgnn_bridge.remove_memory_backup(EXPLAINED_EVENT_MEMORY_LABEL)
        self.tgnn_bridge.reset_model()
        self.known_states = {}
        return best_cf_example.to_cf_example()
