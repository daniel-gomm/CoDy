from __future__ import annotations

import sys
from typing import List

import numpy as np

from CFTGNNExplainer.connector import TGNNBridge
from CFTGNNExplainer.constants import EXPLAINED_EVENT_MEMORY_LABEL, COL_ID
from CFTGNNExplainer.explainer.base import Explainer, CounterFactualExample, calculate_prediction_delta, TreeNode
from CFTGNNExplainer.sampler import PretrainedEdgeSamplerParameters, EdgeSampler, OneBestEdgeSampler


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


class MCTSTreeNode(TreeNode):
    parent: MCTSTreeNode
    children: List[MCTSTreeNode]
    number_of_selections: int
    is_counterfactual: bool
    edge_id: int
    sampling_rank: int
    prediction: float | None

    def __init__(self, edge_id: int, parent: MCTSTreeNode | None, original_prediction: float, sampling_rank: int):
        super().__init__(edge_id, parent, original_prediction)
        self.sampling_rank = sampling_rank
        self.number_of_selections: int = 1
        if self.parent is None:
            self.depth = 0
        else:
            self.depth = self.parent.depth + 1

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
        if self.is_leaf():
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
        if not self.is_leaf():
            # Here the exploitation score could be updates. However, this seems to make the search performance worse

            # self.exploitation_score = max(0.0, np.average([child.exploitation_score for child in self.children
            #                                                if child.expanded]))
            if self.exploitation_score == 0:
                self.exploitation_score = max(0.0, np.average([child.exploitation_score for child in self.children
                                                               if child.expanded]))

        self.number_of_selections += 1
        if self.parent is not None:
            self.parent.expansion_backpropagation()


class CFTGNNExplainer(Explainer):

    def __init__(self, tgnn_bridge: TGNNBridge, candidates_size: int = 75, sampling_strategy: str = 'recent',
                 max_steps: int = 200, verbose: bool = False,
                 pretrained_sampler_parameters: PretrainedEdgeSamplerParameters | None = None):
        super().__init__(tgnn_bridge, sampling_strategy, candidates_size=candidates_size, sample_size=candidates_size,
                         verbose=verbose, pretrained_sampler_parameters=pretrained_sampler_parameters)
        self.max_steps = max_steps
        self.known_states = {}

    def _run_node_expansion(self, explained_edge_id: int, node_to_expand: MCTSTreeNode, sampler: EdgeSampler):
        edge_ids_to_exclude = node_to_expand.get_parent_ids()
        prediction = self.calculate_subgraph_prediction(candidate_events=sampler.subgraph[COL_ID],
                                                        cf_example_events=edge_ids_to_exclude,
                                                        explained_event_id=explained_edge_id,
                                                        candidate_event_id=node_to_expand.edge_id,
                                                        memory_label=EXPLAINED_EVENT_MEMORY_LABEL)

        self._expand_node(explained_edge_id, node_to_expand, prediction, sampler)

    def _expand_node(self, explained_edge_id: int, node_to_expand: MCTSTreeNode, prediction: float, sampler: EdgeSampler):
        self.known_states[node_to_expand.hash()] = prediction

        if node_to_expand.is_counterfactual:
            node_to_expand.expand(prediction, [])
            return

        edge_ids_to_exclude = node_to_expand.get_parent_ids()
        ranked_edge_ids = sampler.rank_subgraph(base_event_id=explained_edge_id,
                                                excluded_events=np.array(edge_ids_to_exclude))
        children = []
        for rank, edge_id in enumerate(ranked_edge_ids):
            new_child = MCTSTreeNode(edge_id, node_to_expand, node_to_expand.original_prediction, rank)
            children.append(new_child)
            if new_child.hash() in self.known_states.keys():
                self._expand_node(explained_edge_id, new_child, self.known_states[new_child.hash()], sampler)
        node_to_expand.expand(prediction, children)

    def explain(self, explained_event_id: int) -> CounterFactualExample:
        """
        Explain the prediction of the provided event id with a counterfactual example found by searching possible
        examples with an adapted 'Monte Carlo' Tree Search
        @param explained_event_id: The event id that is explained
        @return: The best CounterFactualExample found for explaining the event
        """
        original_prediction, sampler = self.initialize_explanation(explained_event_id)
        best_cf_example = None
        step = 0
        max_depth = sys.maxsize
        root_node = MCTSTreeNode(explained_event_id, parent=None, sampling_rank=0,
                                 original_prediction=original_prediction)
        self._expand_node(explained_event_id, root_node, original_prediction, sampler)

        if type(sampler) is OneBestEdgeSampler:
            for child in root_node.children:
                # Expand all children
                self._run_node_expansion(explained_event_id, child, sampler)
                if child.is_counterfactual:
                    if best_cf_example is None:
                        best_cf_example = child
                    elif best_cf_example.exploitation_score < child.exploitation_score:
                        best_cf_example = child
                    if self.verbose:
                        self.logger.info(f'Found counterfactual explanation: '
                                         + str(child.to_cf_example()))
                sampler.set_event_weight(child.edge_id, child.exploitation_score)
            if best_cf_example is not None:
                return best_cf_example.to_cf_example()
            step += 1

        while step <= self.max_steps:
            node_to_expand = None
            while node_to_expand is None:
                node_to_expand = root_node.select_next_leaf(max_depth)
                if node_to_expand.depth > max_depth:
                    node_to_expand.expansion_backpropagation()
                    continue
                if node_to_expand == root_node and root_node.expanded:
                    break  # No nodes are selectable, meaning that we can conclude the search
                if node_to_expand.hash() in self.known_states.keys():
                    # Already encountered this combination -> select new combination of events instead
                    self._expand_node(explained_event_id, node_to_expand, self.known_states[node_to_expand.hash()],
                                      sampler)
                    node_to_expand = None
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
                    self.logger.info(f'Found counterfactual explanation: ' + str(node_to_expand.to_cf_example()))
            step += 1
        if best_cf_example is None:
            best_cf_example = find_best_non_counterfactual_example(root_node)
        self.tgnn_bridge.remove_memory_backup(EXPLAINED_EVENT_MEMORY_LABEL)
        self.tgnn_bridge.reset_model()
        self.known_states = {}
        return best_cf_example.to_cf_example()
