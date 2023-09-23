from __future__ import annotations

import sys
from typing import List

import numpy as np

from CFTGNNExplainer.connector.bridge import TGNNBridge
from CFTGNNExplainer.explainer.base import Explainer, calculate_prediction_delta, CounterFactualExample
from CFTGNNExplainer.sampling.sampler import EdgeSampler
from CFTGNNExplainer.constants import CUR_IT_MIN_EVENT_MEM_LBL, EXPLAINED_EVENT_MEMORY_LABEL, COL_ID


def select_best_cf_example(current_best_example: TreeNode | None, candidate_examples: List[TreeNode]) -> TreeNode:
    sorted_candidates = sorted(candidate_examples, key=lambda node: node.depth)
    min_depth = sorted_candidates[0].depth
    if current_best_example is not None and min_depth > current_best_example.depth:
        return current_best_example
    min_depth_candidates = [candidate for candidate in candidate_examples if candidate.depth == min_depth]
    sorted_min_depth_candidates = sorted(min_depth_candidates, key=lambda node: node.exploitation_score, reverse=True)
    best_candidate = sorted_min_depth_candidates[0]
    if current_best_example is None:
        return best_candidate
    if (min_depth < current_best_example.depth or
            calculate_prediction_delta(current_best_example.original_prediction, current_best_example.prediction) >
            calculate_prediction_delta(best_candidate.original_prediction, best_candidate.prediction)):
        return best_candidate
    return current_best_example


def find_best_non_counterfactual_example(root_node: TreeNode) -> TreeNode:
    """
        Breadth-first search for explanation that comes closest to counterfactual example
    """
    best_example = root_node
    nodes_to_visit = root_node.children
    while len(nodes_to_visit) != 0:
        explored_node = nodes_to_visit.pop()
        if explored_node.is_leaf():
            if (calculate_prediction_delta(best_example.original_prediction, best_example.prediction) <
                    calculate_prediction_delta(explored_node.original_prediction, explored_node.prediction)):
                best_example = explored_node
        else:
            nodes_to_visit.extend(explored_node.children)
    return best_example


class TreeNode:
    parent: TreeNode
    children: List[TreeNode]
    score: float
    number_of_selections: int
    is_counterfactual: bool
    edge_id: int

    def __init__(self, edge_id: int, parent: TreeNode | None, prediction: float, original_prediction: float):
        self.edge_id: int = edge_id
        self.parent = parent
        self.prediction: float = prediction
        self.original_prediction: float = original_prediction
        if self.original_prediction * self.prediction < 0:
            self.is_counterfactual = True
        else:
            self.is_counterfactual = False
        self.exploitation_score: float = max(0.0,
                                             (calculate_prediction_delta(self.original_prediction, self.prediction) /
                                              abs(self.original_prediction)))
        self.number_of_selections: int = 1
        self.children = []
        depth = 0
        parent = self.parent
        while parent is not None:
            depth += 1
            parent = parent.parent
        self.depth = depth
        self.expanded = False
        self.fully_expanded = self.is_counterfactual  # Counterfactual nodes are always fully expanded

    def check_fully_expanded(self):
        """
        Recursively check if the node is already fully expanded, meaning that no further expansions of its child
        nodes are possible
        """
        for child in self.children:
            if not child.fully_expanded:
                return
        self.fully_expanded = True
        if self.parent is not None:
            self.parent.check_fully_expanded()

    def is_leaf(self) -> bool:
        """
        Check if the node is a leaf node, meaning that it has no children
        """
        return len(self.children) == 0

    def calculate_score(self):
        """
        Calculate the search score which balances exploration with exploitation
        """
        exploration_score = np.sqrt(2) * np.sqrt(np.log(self.parent.number_of_selections) / self.number_of_selections)
        return self.exploitation_score + exploration_score

    def select_next_leaf(self, max_depth: int) -> TreeNode:
        """
        Select the next leaf node for expansion
        @param max_depth: Maximum depth at which to search for leaf nodes
        @return: Leaf node to expand
        """
        if self.is_leaf() or self.depth == max_depth:
            return self
        selected_child = None
        best_score = 0
        candidate_children = [child for child in self.children if not (child.is_counterfactual or child.fully_expanded)]
        if max_depth == self.depth - 1:
            # If max depth is reached in the next level only consider children that can be directly expanded, otherwise
            #  the max depth requirement would be violated
            candidate_children = [child for child in self.children if child.is_leaf()]
        for child in candidate_children:
            child_score = child.calculate_score()
            if child_score > best_score:
                best_score = child_score
                selected_child = child
        if selected_child is None:  # This means that there are no candidate children -> return oneself
            if self.expanded:
                self.check_fully_expanded()  # When no selection is possible the node is fully expanded
            return self
        return selected_child.select_next_leaf(max_depth)

    def selection_backpropagation(self):
        """
        Propagate the information that a node is selected backwards and update scores
        """
        if not self.is_leaf():
            avg_exploitation_score = sum(child.exploitation_score for child in self.children) / len(self.children)
            # TODO: Assess if this should be max or just the avg or if the best score should be propagated backwards
            self.exploitation_score = max(self.exploitation_score, avg_exploitation_score)
        self.number_of_selections += 1
        if self.parent is not None:
            self.parent.selection_backpropagation()

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


class SearchingCFExplainer(Explainer):

    def __init__(self, tgnn_bridge: TGNNBridge, candidates_size: int = 75, sample_size: int = 10,
                 sampling_strategy: str = 'recent', max_steps: int = 50, verbose: bool = False):
        super().__init__(tgnn_bridge, sampling_strategy, candidates_size=candidates_size, sample_size=sample_size,
                         verbose=verbose)
        self.max_steps = max_steps

    def expand_node(self, explained_edge_id: int, node_to_expand: TreeNode, sampler: EdgeSampler,
                    known_cf_examples: List[np.ndarray] | None = None) -> List[TreeNode]:
        counterfactual_examples: List[TreeNode] = []
        original_prediction = node_to_expand.original_prediction
        if not node_to_expand.is_leaf():
            return counterfactual_examples

        edge_ids_to_exclude = []
        node = node_to_expand
        while node.parent is not None:
            edge_ids_to_exclude.append(node.edge_id)
            node = node.parent

        sampled_edge_ids = sampler.sample(explained_edge_id, excluded_events=np.array(edge_ids_to_exclude),
                                          size=self.sample_size, known_cf_examples=known_cf_examples)
        if self.verbose:
            self.logger.info(f'Selected node {str(node_to_expand.edge_id)} and excluded edge ids '
                             f'{str(edge_ids_to_exclude)}')
        if len(sampled_edge_ids) > 0:
            min_event_id = sampler.subgraph[COL_ID].min() - 1
            self.tgnn_bridge.initialize(min_event_id, show_progress=False,
                                        memory_label=EXPLAINED_EVENT_MEMORY_LABEL)
        for edge_id in sampled_edge_ids:
            prediction = self.calculate_subgraph_prediction(candidate_events=sampled_edge_ids,
                                                            cf_example_events=edge_ids_to_exclude,
                                                            explained_event_id=explained_edge_id,
                                                            candidate_event_id=edge_id,
                                                            memory_label=CUR_IT_MIN_EVENT_MEM_LBL)
            new_child = TreeNode(edge_id, node_to_expand, prediction, original_prediction)
            node_to_expand.children.append(new_child)
            if new_child.is_counterfactual:
                counterfactual_examples.append(new_child)
        self.tgnn_bridge.remove_memory_backup(CUR_IT_MIN_EVENT_MEM_LBL)
        return counterfactual_examples

    def explain(self, explained_event_id: int):
        original_prediction, sampler = self.initialize_explanation(explained_event_id)
        best_cf_example = None
        known_cf_examples = []
        max_depth = sys.maxsize
        root_node = TreeNode(explained_event_id, parent=None, prediction=original_prediction,
                             original_prediction=original_prediction)
        step = 0
        while step <= self.max_steps:
            step += 1
            node_to_expand = root_node.select_next_leaf(max_depth)
            node_to_expand.selection_backpropagation()
            if node_to_expand.depth == max_depth:
                continue
            if node_to_expand == root_node and root_node.expanded:
                break  # No nodes are selectable, meaning that we can conclude the search
            cf_examples = self.expand_node(explained_event_id, node_to_expand, sampler, known_cf_examples)
            node_to_expand.expanded = True
            if len(cf_examples) > 0:
                best_cf_example = select_best_cf_example(best_cf_example, cf_examples)
                max_depth = best_cf_example.depth
                known_cf_examples.extend(np.array(example.to_cf_example().event_ids) for example in cf_examples)
                if self.verbose:
                    self.logger.info(f'Found counterfactual explanation (could be old): '
                                     + str(best_cf_example.to_cf_example()))
        if best_cf_example is None:
            best_cf_example = find_best_non_counterfactual_example(root_node)
        self.tgnn_bridge.remove_memory_backup(EXPLAINED_EVENT_MEMORY_LABEL)
        self.tgnn_bridge.reset_model()
        return best_cf_example.to_cf_example()
