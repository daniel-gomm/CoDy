from __future__ import annotations

import sys
from typing import List

import numpy as np

from CFTGNNExplainer.connector import TGNNWrapper
from CFTGNNExplainer.explainer.base import Explainer, calculate_prediction_delta, TreeNode
from CFTGNNExplainer.sampler import EdgeSampler, PretrainedEdgeSamplerParameters
from CFTGNNExplainer.constants import CUR_IT_MIN_EVENT_MEM_LBL, EXPLAINED_EVENT_MEMORY_LABEL, COL_ID


def select_best_cf_example(current_best_example: BatchSearchTreeNode | None,
                           candidate_examples: List[BatchSearchTreeNode]) -> BatchSearchTreeNode:
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


def find_best_non_counterfactual_example(root_node: BatchSearchTreeNode) -> BatchSearchTreeNode:
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


class BatchSearchTreeNode(TreeNode):
    parent: BatchSearchTreeNode
    children: List[BatchSearchTreeNode]
    score: float
    number_of_selections: int
    is_counterfactual: bool
    edge_id: int

    def __init__(self, edge_id: int, parent: BatchSearchTreeNode | None, prediction: float, original_prediction: float):
        super().__init__(edge_id, parent, original_prediction)
        self.prediction: float = prediction
        self.exploitation_score: float = max(0.0,
                                             (calculate_prediction_delta(self.original_prediction, self.prediction) /
                                              abs(self.original_prediction)))
        self.number_of_selections: int = 1
        depth = 0
        parent = self.parent
        while parent is not None:
            depth += 1
            parent = parent.parent
        self.depth = depth
        self.max_expansion_reached = self.is_counterfactual  # Counterfactual nodes are always fully expanded

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

    def _calculate_score(self):
        """
        Calculate the search score which balances exploration with exploitation
        """
        exploration_score = np.sqrt(2) * np.sqrt(np.log(self.parent.number_of_selections) / self.number_of_selections)
        return self.exploitation_score + exploration_score

    def select_next_leaf(self, max_depth: int) -> BatchSearchTreeNode:
        """
        Select the next leaf node for expansion
        @param max_depth: Maximum depth at which to search for leaf nodes
        @return: Leaf node to expand
        """
        if self.is_leaf() or self.depth == max_depth:
            return self
        selected_child = None
        best_score = 0
        candidate_children = [child for child in self.children if not (child.is_counterfactual or
                                                                       child.max_expansion_reached)]
        if max_depth == self.depth - 1:
            # If max depth is reached in the next level only consider children that can be directly expanded, otherwise
            #  the max depth requirement would be violated
            candidate_children = [child for child in self.children if child.is_leaf()]
        for child in candidate_children:
            child_score = child._calculate_score()
            if child_score > best_score:
                best_score = child_score
                selected_child = child
        if selected_child is None:  # This means that there are no candidate children -> return oneself
            if self.expanded:
                self._check_max_expanded()  # When no selection is possible the node is fully expanded
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


class SearchingCFExplainer(Explainer):

    def __init__(self, tgnn_wrapper: TGNNWrapper, candidates_size: int = 75, sample_size: int = 10,
                 sampling_strategy: str = 'recent', max_steps: int = 50, verbose: bool = False,
                 approximate_predictions: bool = True,
                 pretrained_sampler_parameters: PretrainedEdgeSamplerParameters | None = None):
        super().__init__(tgnn_wrapper, sampling_strategy, candidates_size=candidates_size, sample_size=sample_size,
                         verbose=verbose, approximate_predictions=approximate_predictions,
                         pretrained_sampler_parameters=pretrained_sampler_parameters)
        self.max_steps = max_steps

    def expand_node(self, explained_edge_id: int, node_to_expand: BatchSearchTreeNode, sampler: EdgeSampler,
                    known_cf_examples: List[np.ndarray] | None = None) -> List[BatchSearchTreeNode]:
        counterfactual_examples: List[BatchSearchTreeNode] = []
        original_prediction = node_to_expand.original_prediction
        if not node_to_expand.is_leaf():
            return counterfactual_examples

        edge_ids_to_exclude = node_to_expand.get_parent_ids()
        sampled_edge_ids = sampler.sample(explained_edge_id, excluded_events=np.array(edge_ids_to_exclude),
                                          size=self.sample_size, known_cf_examples=known_cf_examples)
        if self.verbose:
            self.logger.info(f'Selected node {str(node_to_expand.edge_id)} and excluded edge ids '
                             f'{str(edge_ids_to_exclude)}')
        if len(sampled_edge_ids) > 0:
            min_event_id = sampler.subgraph[COL_ID].min() - 1
            self.tgnn.initialize(min_event_id, show_progress=False,
                                 memory_label=EXPLAINED_EVENT_MEMORY_LABEL)
        for edge_id in sampled_edge_ids:
            prediction = self.calculate_subgraph_prediction(candidate_events=sampled_edge_ids,
                                                            cf_example_events=edge_ids_to_exclude,
                                                            explained_event_id=explained_edge_id,
                                                            candidate_event_id=edge_id,
                                                            original_prediction=original_prediction,
                                                            memory_label=CUR_IT_MIN_EVENT_MEM_LBL)
            new_child = BatchSearchTreeNode(edge_id, node_to_expand, prediction, original_prediction)
            node_to_expand.children.append(new_child)
            if new_child.is_counterfactual:
                counterfactual_examples.append(new_child)
        self.tgnn.remove_memory_backup(CUR_IT_MIN_EVENT_MEM_LBL)
        return counterfactual_examples

    def explain(self, explained_event_id: int):
        original_prediction, sampler = self.initialize_explanation(explained_event_id)
        best_cf_example = None
        known_cf_examples = []
        max_depth = sys.maxsize
        root_node = BatchSearchTreeNode(explained_event_id, parent=None, prediction=original_prediction,
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
        self.tgnn.remove_memory_backup(EXPLAINED_EVENT_MEMORY_LABEL)
        self.tgnn.reset_model()
        return best_cf_example.to_cf_example()
