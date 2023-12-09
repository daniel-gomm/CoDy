import sys

import numpy as np
from cody.constants import COL_ID, EXPLAINED_EVENT_MEMORY_LABEL, CUR_IT_MIN_EVENT_MEM_LBL
from cody.explainer.base import Explainer, CounterFactualExample, calculate_prediction_delta, TreeNode
from cody.selection import OneDeltaSelectionStrategy


class GreedyTreeNode(TreeNode):

    def __init__(self, edge_id: int, parent: TreeNode | None, original_prediction: float, prediction: float):
        super().__init__(edge_id, parent, original_prediction)
        self.prediction: float = prediction
        self.exploitation_score: float = max(0.0,
                                             (calculate_prediction_delta(self.original_prediction, self.prediction) /
                                              abs(self.original_prediction)))
        if self.original_prediction * self.prediction < 0:
            self.is_counterfactual = True
            self.max_expansion_reached = True

    def select_next_leaf(self, max_depth: int) -> TreeNode | None:
        if not self.expanded:
            return self
        best_child = None
        for child in self.children:
            if child.exploitation_score > self.exploitation_score:
                if best_child is None:
                    best_child = child
                elif child.exploitation_score > best_child.exploitation_score:
                    best_child = child
        if best_child is not None:
            return best_child.select_next_leaf(max_depth)
        return None

    def expansion_backpropagation(self):
        pass


class GreedyCFExplainer(Explainer):

    def explain(self, explained_event_id: int) -> CounterFactualExample:
        original_prediction, sampler = self.initialize_explanation(explained_event_id)
        min_event_id = sampler.subgraph[COL_ID].min() - 1
        root_node = GreedyTreeNode(explained_event_id, None, original_prediction=original_prediction,
                                   prediction=original_prediction)
        max_depth = sys.maxsize
        best_cf_example = None
        best_non_cf_example = root_node

        if type(sampler) is OneDeltaSelectionStrategy:
            for child_id in sampler.rank_subgraph(base_event_id=explained_event_id, excluded_events=np.array([])):
                prediction = self.calculate_subgraph_prediction(candidate_events=sampler.subgraph[COL_ID].to_numpy(),
                                                                cf_example_events=[],
                                                                explained_event_id=explained_event_id,
                                                                candidate_event_id=child_id,
                                                                original_prediction=original_prediction,
                                                                memory_label=EXPLAINED_EVENT_MEMORY_LABEL)
                child_node = GreedyTreeNode(child_id, parent=root_node, original_prediction=original_prediction,
                                            prediction=prediction)
                root_node.children.append(child_node)
                if child_node.is_counterfactual:
                    if best_cf_example is None:
                        best_cf_example = child_node
                    elif best_cf_example.exploitation_score < child_node.exploitation_score:
                        best_cf_example = child_node
                    if self.verbose:
                        self.logger.info(f'Found counterfactual explanation: ' + str(child_node.to_cf_example()))
                sampler.set_event_weight(child_node.edge_id, child_node.exploitation_score)
            if best_cf_example is not None:
                return best_cf_example.to_cf_example()
            root_node.expanded = True

        i = 0
        while True:
            node_to_expand = root_node.select_next_leaf(max_depth)
            if node_to_expand is None or (node_to_expand == root_node and root_node.expanded):
                break  # No more nodes can be selected -> conclude search without a cf-example
            best_non_cf_example = node_to_expand
            if self.verbose:
                self.logger.info(f'Iteration {i} selected event {node_to_expand.edge_id} with prediction '
                                 f'{node_to_expand.prediction}. '
                                 f'CF-example events: {node_to_expand.hash()}')
            sampled_edge_ids = sampler.sample(explained_event_id,
                                              excluded_events=np.array(node_to_expand.get_parent_ids()),
                                              size=self.sample_size)
            self.tgnn.initialize(min_event_id, show_progress=False,
                                 memory_label=EXPLAINED_EVENT_MEMORY_LABEL)
            for candidate_event_id in sampled_edge_ids:
                prediction = self.calculate_subgraph_prediction(candidate_events=sampled_edge_ids,
                                                                cf_example_events=node_to_expand.get_parent_ids() +
                                                                                  [node_to_expand.edge_id],
                                                                explained_event_id=explained_event_id,
                                                                candidate_event_id=candidate_event_id,
                                                                original_prediction=original_prediction,
                                                                memory_label=CUR_IT_MIN_EVENT_MEM_LBL)
                child_node = GreedyTreeNode(candidate_event_id, parent=node_to_expand,
                                            original_prediction=original_prediction, prediction=prediction)
                node_to_expand.children.append(child_node)
                if child_node.is_counterfactual:
                    if best_cf_example is None:
                        best_cf_example = child_node
                    elif best_cf_example.exploitation_score < child_node.exploitation_score:
                        best_cf_example = child_node
                    if self.verbose:
                        self.logger.info(f'Found counterfactual explanation: ' + str(child_node.to_cf_example()))
            self.tgnn.remove_memory_backup(CUR_IT_MIN_EVENT_MEM_LBL)
            node_to_expand.expanded = True
            if best_cf_example is not None:
                break
            i += 1

        best_example = best_cf_example
        if best_example is None:
            best_example = best_non_cf_example
        self.tgnn.remove_memory_backup(EXPLAINED_EVENT_MEMORY_LABEL)
        self.tgnn.reset_model()
        return best_example.to_cf_example()
