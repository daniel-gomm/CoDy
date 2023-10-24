import sys
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import time

from CFTGNNExplainer.connector import TGNNBridge
from CFTGNNExplainer.constants import CUR_IT_MIN_EVENT_MEM_LBL, EXPLAINED_EVENT_MEMORY_LABEL, COL_ID
from CFTGNNExplainer.explainer.base import Explainer, CounterFactualExample, TreeNode
from CFTGNNExplainer.explainer.greedy import GreedyCFExplainer, GreedyTreeNode
from CFTGNNExplainer.sampler import EdgeSampler, PretrainedEdgeSamplerParameters, OneBestEdgeSampler
from CFTGNNExplainer.explainer.searching import (BatchSearchTreeNode, select_best_cf_example,
                                                 find_best_non_counterfactual_example, SearchingCFExplainer)
from CFTGNNExplainer.explainer.mcts import CFTGNNExplainer, MCTSTreeNode
from CFTGNNExplainer.explainer.mcts import find_best_non_counterfactual_example as find_best_non_cf_example

LAST_PREDICTION_MEMORY_LABEL = 'last_original_score'
NEW_PREDICTION_MEMORY_LABEL = 'new_original_score'

EVALUATION_STATE_CACHE = {}


@dataclass
class PredictionResult:
    prediction_time_ns: int
    prediction: float


@dataclass
class EvaluationCounterFactualExample(CounterFactualExample):
    timings: Dict
    statistics: Dict

    def to_dict(self) -> Dict:
        results = {
            'explained_event_id': self.explained_event_id,
            'original_prediction': self.original_prediction,
            'counterfactual_prediction': self.counterfactual_prediction,
            'achieves_counterfactual_explanation': self.achieves_counterfactual_explanation,
            'cf_example_event_ids': self.event_ids,
            'cf_example_absolute_importances': self.get_absolute_importances(),
            'cf_example_raw_importances': self.event_importances
        }
        results.update(self.statistics)
        results.update(self.timings)
        return results


class EvaluationExplainer(Explainer):
    explanation_results_list: List[EvaluationCounterFactualExample] = []

    def get_evaluation_original_prediction(self, explained_event_id: int, last_event_id: int) -> float:
        """
        Optimize the calculation of the original prediction by resuming from last event id so that no complete rollout
        is necessary
        @param explained_event_id: Event ID for the event that is explained
        @param last_event_id: Event ID of the last event on which this function has been called
        @return: The original prediction for the explained event id
        """
        self.tgnn_bridge.set_evaluation_mode(True)
        self.tgnn_bridge.initialize(last_event_id, memory_label=LAST_PREDICTION_MEMORY_LABEL)
        self.tgnn_bridge.initialize(explained_event_id - 1, memory_label=NEW_PREDICTION_MEMORY_LABEL)
        original_prediction, _ = self.tgnn_bridge.predict(explained_event_id, result_as_logit=True)
        self.tgnn_bridge.memory_backups_map[LAST_PREDICTION_MEMORY_LABEL] = (
            self.tgnn_bridge.memory_backups_map)[NEW_PREDICTION_MEMORY_LABEL]
        del self.tgnn_bridge.memory_backups_map[NEW_PREDICTION_MEMORY_LABEL]
        return original_prediction.detach().cpu().item()

    def initialize_explanation_evaluation(self, explained_event_id: int, original_prediction: float) -> EdgeSampler:
        subgraph = self.subgraph_generator.get_fixed_size_k_hop_temporal_subgraph(num_hops=self.num_hops,
                                                                                  base_event_id=explained_event_id,
                                                                                  size=self.candidates_size)
        self.tgnn_bridge.set_evaluation_mode(True)
        return self._create_sampler(subgraph, explained_event_id, original_prediction=original_prediction)

    def evaluate_explanation(self, explained_event_id: int, original_prediction: float) -> (
            EvaluationCounterFactualExample):
        """
        Explain the provided event
        @param explained_event_id: Event id to explain
        @param original_prediction: Original prediction for the event
        @return: The counterfactual explanation
        """
        raise NotImplementedError


class EvaluationGreedyCFExplainer(GreedyCFExplainer, EvaluationExplainer):

    def __init__(self, tgnn_bridge: TGNNBridge, sampling_strategy: str = 'recent', sample_size: int = 10,
                 candidates_size: int = 75, verbose: bool = False,
                 pretrained_sampler_parameters: PretrainedEdgeSamplerParameters | None = None):
        super(GreedyCFExplainer, self).__init__(tgnn_bridge=tgnn_bridge, sampling_strategy=sampling_strategy,
                                                sample_size=sample_size, candidates_size=candidates_size,
                                                verbose=verbose,
                                                pretrained_sampler_parameters=pretrained_sampler_parameters)
        super(EvaluationExplainer, self).__init__(tgnn_bridge=tgnn_bridge, sampling_strategy=sampling_strategy,
                                                  candidates_size=candidates_size, sample_size=sample_size,
                                                  verbose=verbose,
                                                  pretrained_sampler_parameters=pretrained_sampler_parameters)
        self.last_min_id = 0

    def create_child_node(self, node_to_expand: TreeNode, memory_label: str, explained_event_id: int,
                          candidate_event_id: int, sampled_edge_ids):
        child_hash = f'{explained_event_id}-{node_to_expand.hash()}-{candidate_event_id}'
        exp_cache_save_time = 0
        if child_hash in EVALUATION_STATE_CACHE.keys():
            result = EVALUATION_STATE_CACHE[child_hash]
            oracle_call_duration = result.prediction_time_ns
            exp_cache_save_time = result.prediction_time_ns
            prediction = result.prediction
        else:
            oracle_call_start = time.time_ns()
            prediction = self.calculate_subgraph_prediction(candidate_events=sampled_edge_ids,
                                                            cf_example_events=node_to_expand.get_parent_ids() +
                                                                              [node_to_expand.edge_id],
                                                            explained_event_id=explained_event_id,
                                                            candidate_event_id=candidate_event_id,
                                                            memory_label=memory_label)
            oracle_call_duration = time.time_ns() - oracle_call_start
            EVALUATION_STATE_CACHE[child_hash] = PredictionResult(oracle_call_duration, prediction)
        child_node = GreedyTreeNode(candidate_event_id, parent=node_to_expand,
                                    original_prediction=node_to_expand.original_prediction, prediction=prediction)
        node_to_expand.children.append(child_node)
        return child_node, oracle_call_duration, exp_cache_save_time

    def evaluate_explanation(self, explained_event_id: int,
                             original_prediction: float) -> EvaluationCounterFactualExample:
        if original_prediction is None:
            original_prediction, sampler = self.initialize_explanation(explained_event_id)
        else:
            sampler = self.initialize_explanation_evaluation(explained_event_id, original_prediction)
        timings = {}
        statistics = {}
        oracle_calls = 0
        oracle_call_time = 0
        cache_saved_oracle_call_time = 0
        start_time = time.time_ns()
        min_event_id = sampler.subgraph[COL_ID].min() - 1
        root_node = GreedyTreeNode(explained_event_id, None, original_prediction=original_prediction,
                                   prediction=original_prediction)
        max_depth = sys.maxsize
        best_cf_example = None
        best_non_cf_example = root_node
        skip_search = False

        if type(sampler) is OneBestEdgeSampler:
            for child_id in sampler.rank_subgraph(base_event_id=explained_event_id, excluded_events=np.array([])):
                child_node, _, _ = self.create_child_node(node_to_expand=root_node,
                                                          memory_label=EXPLAINED_EVENT_MEMORY_LABEL,
                                                          explained_event_id=explained_event_id,
                                                          candidate_event_id=child_id,
                                                          sampled_edge_ids=sampler.subgraph[COL_ID])
                if child_node.is_counterfactual:
                    if best_cf_example is None:
                        best_cf_example = child_node
                    elif best_cf_example.exploitation_score < child_node.exploitation_score:
                        best_cf_example = child_node
                    if self.verbose:
                        self.logger.info(f'Found counterfactual explanation: ' + str(child_node.to_cf_example()))
                sampler.set_event_weight(child_node.edge_id, child_node.exploitation_score)
            if best_cf_example is not None:
                skip_search = True
            root_node.expanded = True

        i = 0
        init_end_time = time.time_ns()
        timings['init_duration'] = init_end_time - start_time
        while not skip_search:
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
            self.tgnn_bridge.initialize(min_event_id, show_progress=False,
                                        memory_label=EXPLAINED_EVENT_MEMORY_LABEL)
            for candidate_event_id in sampled_edge_ids:
                child_node, oracle_call_duration, exp_cache_save_time = (
                    self.create_child_node(node_to_expand, memory_label=CUR_IT_MIN_EVENT_MEM_LBL,
                                           explained_event_id=explained_event_id,
                                           candidate_event_id=candidate_event_id,
                                           sampled_edge_ids=sampled_edge_ids))
                oracle_call_time += oracle_call_duration
                cache_saved_oracle_call_time += exp_cache_save_time
                if child_node.is_counterfactual:
                    if best_cf_example is None:
                        best_cf_example = child_node
                    elif best_cf_example.exploitation_score < child_node.exploitation_score:
                        best_cf_example = child_node
                    if self.verbose:
                        self.logger.info(f'Found counterfactual explanation: ' + str(child_node.to_cf_example()))
            self.tgnn_bridge.remove_memory_backup(CUR_IT_MIN_EVENT_MEM_LBL)
            node_to_expand.expanded = True
            if best_cf_example is not None:
                break
            i += 1

        best_example = best_cf_example
        if best_example is None:
            best_example = best_non_cf_example
        self.tgnn_bridge.remove_memory_backup(EXPLAINED_EVENT_MEMORY_LABEL)
        self.tgnn_bridge.reset_model()
        end_time = time.time_ns()
        timings['oracle_call_duration'] = oracle_call_time
        timings['explanation_duration'] = end_time - start_time - oracle_call_time + cache_saved_oracle_call_time
        timings['total_duration'] = end_time - start_time + cache_saved_oracle_call_time
        statistics['oracle_calls'] = oracle_calls
        statistics['candidate_size'] = len(sampler.subgraph)
        statistics['candidates'] = sampler.subgraph[COL_ID].to_list()
        result_cf_example = best_example.to_cf_example()
        cf_example = EvaluationCounterFactualExample(explained_event_id=explained_event_id,
                                                     original_prediction=original_prediction,
                                                     counterfactual_prediction=result_cf_example.counterfactual_prediction,
                                                     achieves_counterfactual_explanation=
                                                     result_cf_example.achieves_counterfactual_explanation,
                                                     event_ids=result_cf_example.event_ids,
                                                     event_importances=result_cf_example.event_importances,
                                                     timings=timings,
                                                     statistics=statistics)
        if self.verbose:
            self.logger.info(f'Final explanation result: {str(cf_example)}\n')
        return cf_example


class EvaluationSearchingCFExplainer(SearchingCFExplainer, EvaluationExplainer):

    def __init__(self, tgnn_bridge: TGNNBridge, sampling_strategy: str = 'recent', max_steps: int = 50,
                 sample_size: int = 10, candidates_size: int = 75, verbose: bool = False,
                 pretrained_sampler_parameters: PretrainedEdgeSamplerParameters | None = None):
        SearchingCFExplainer.__init__(self, tgnn_bridge=tgnn_bridge, sampling_strategy=sampling_strategy,
                                      sample_size=sample_size, candidates_size=candidates_size, verbose=verbose,
                                      max_steps=max_steps, pretrained_sampler_parameters=pretrained_sampler_parameters)
        EvaluationExplainer.__init__(self, tgnn_bridge=tgnn_bridge, sampling_strategy=sampling_strategy,
                                     candidates_size=candidates_size, sample_size=sample_size, verbose=verbose,
                                     pretrained_sampler_parameters=pretrained_sampler_parameters)
        self.last_min_id = 0

    def expand_node(self, explained_edge_id: int, node_to_expand: BatchSearchTreeNode, sampler: EdgeSampler,
                    known_cf_examples: List[np.ndarray] | None = None) -> (List[BatchSearchTreeNode], int, int):
        oracle_calls = 0
        oracle_call_time = 0
        counterfactual_examples: List[BatchSearchTreeNode] = []
        original_prediction = node_to_expand.original_prediction
        if not node_to_expand.is_leaf():
            return counterfactual_examples, oracle_calls, oracle_call_time

        edge_ids_to_exclude = node_to_expand.get_parent_ids()
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
            oracle_call_start = time.time_ns()
            prediction = self.calculate_subgraph_prediction(candidate_events=sampled_edge_ids,
                                                            cf_example_events=edge_ids_to_exclude,
                                                            explained_event_id=explained_edge_id,
                                                            candidate_event_id=edge_id,
                                                            memory_label=CUR_IT_MIN_EVENT_MEM_LBL)
            oracle_call_time += time.time_ns() - oracle_call_start
            oracle_calls += 1
            new_child = BatchSearchTreeNode(edge_id, node_to_expand, prediction, original_prediction)
            node_to_expand.children.append(new_child)
            if new_child.is_counterfactual:
                counterfactual_examples.append(new_child)
        self.tgnn_bridge.remove_memory_backup(CUR_IT_MIN_EVENT_MEM_LBL)
        return counterfactual_examples, oracle_calls, oracle_call_time

    def evaluate_explanation(self, explained_event_id: int, original_prediction: float) -> (
            EvaluationCounterFactualExample):
        if original_prediction is None:
            original_prediction, sampler = self.initialize_explanation(explained_event_id)
        else:
            sampler = self.initialize_explanation_evaluation(explained_event_id, original_prediction)
        timings = {}
        statistics = {}
        start_time = time.time_ns()
        min_event_id = sampler.subgraph[COL_ID].min() - 1
        if 0 < self.last_min_id <= min_event_id:
            self.tgnn_bridge.initialize(self.last_min_id, show_progress=False,
                                        memory_label=EXPLAINED_EVENT_MEMORY_LABEL)
        self.tgnn_bridge.remove_memory_backup(EXPLAINED_EVENT_MEMORY_LABEL)
        oracle_calls = 0
        oracle_call_time = 0

        best_cf_example = None
        known_cf_examples = []
        max_depth = sys.maxsize
        root_node = BatchSearchTreeNode(explained_event_id, parent=None, prediction=original_prediction,
                                        original_prediction=original_prediction)
        step = 0
        init_end_time = time.time_ns()
        timings['init_duration'] = init_end_time - start_time
        while step <= self.max_steps:
            step += 1
            node_to_expand = root_node.select_next_leaf(max_depth)
            node_to_expand.selection_backpropagation()
            if node_to_expand.depth == max_depth:
                continue
            if node_to_expand == root_node and root_node.expanded:
                break  # No nodes are selectable, meaning that we can conclude the search
            cf_examples, ex_oracle_calls, ex_oracle_call_time = self.expand_node(explained_event_id, node_to_expand,
                                                                                 sampler, known_cf_examples)
            node_to_expand.expanded = True
            oracle_calls += ex_oracle_calls
            oracle_call_time += ex_oracle_call_time
            if len(cf_examples) > 0:
                best_cf_example = select_best_cf_example(best_cf_example, cf_examples)
                max_depth = best_cf_example.depth
                known_cf_examples.extend(np.array(example.to_cf_example().event_ids) for example in cf_examples)
                if self.verbose:
                    self.logger.info(f'Found counterfactual explanation (could be old): '
                                     + str(best_cf_example.to_cf_example()))
        if best_cf_example is None:
            best_cf_example = find_best_non_counterfactual_example(root_node)
        # self.tgnn_bridge.remove_memory_backup(EXPLAINED_EVENT_MEMORY_LABEL)
        self.last_min_id = sampler.subgraph[COL_ID].min() - 1
        self.tgnn_bridge.reset_model()
        end_time = time.time_ns()
        timings['oracle_call_duration'] = oracle_call_time
        timings['explanation_duration'] = end_time - start_time - oracle_call_time
        timings['total_duration'] = end_time - start_time
        statistics['oracle_calls'] = oracle_calls
        statistics['candidate_size'] = len(sampler.subgraph)
        statistics['candidates'] = sampler.subgraph[COL_ID].to_list()
        cf_ex = best_cf_example.to_cf_example()
        eval_cf_example = EvaluationCounterFactualExample(explained_event_id=explained_event_id,
                                                          original_prediction=original_prediction,
                                                          counterfactual_prediction=cf_ex.counterfactual_prediction,
                                                          achieves_counterfactual_explanation=
                                                          cf_ex.achieves_counterfactual_explanation,
                                                          event_ids=cf_ex.event_ids,
                                                          event_importances=cf_ex.event_importances,
                                                          timings=timings,
                                                          statistics=statistics)
        if self.verbose:
            self.logger.info(f'Final explanation result: {str(eval_cf_example)}\n')
        return eval_cf_example


class EvaluationCFTGNNExplainer(CFTGNNExplainer, EvaluationExplainer):

    def __init__(self, tgnn_bridge: TGNNBridge, sampling_strategy: str = 'recent', max_steps: int = 200,
                 candidates_size: int = 75, verbose: bool = False,
                 pretrained_sampler_parameters: PretrainedEdgeSamplerParameters | None = None):
        CFTGNNExplainer.__init__(self, tgnn_bridge=tgnn_bridge, sampling_strategy=sampling_strategy,
                                 candidates_size=candidates_size, verbose=verbose, max_steps=max_steps,
                                 pretrained_sampler_parameters=pretrained_sampler_parameters)
        EvaluationExplainer.__init__(self, tgnn_bridge=tgnn_bridge, sampling_strategy=sampling_strategy,
                                     candidates_size=candidates_size, sample_size=candidates_size, verbose=verbose,
                                     pretrained_sampler_parameters=pretrained_sampler_parameters)
        self.last_min_id = 0

    def _get_evaluation_subgraph_prediction(self, candidate_events: np.ndarray, node_to_expand: MCTSTreeNode,
                                            explained_event_id: int,
                                            memory_label: str = EXPLAINED_EVENT_MEMORY_LABEL) -> (float, int, int):
        full_hash = f'{explained_event_id}-{node_to_expand.hash()}'
        if full_hash in EVALUATION_STATE_CACHE.keys():
            result = EVALUATION_STATE_CACHE[full_hash]
            return result.prediction, result.prediction_time_ns, result.prediction_time_ns
        else:
            oracle_call_start_time = time.time_ns()
            prediction = self.calculate_subgraph_prediction(candidate_events=candidate_events,
                                                            cf_example_events=node_to_expand.get_parent_ids(),
                                                            explained_event_id=explained_event_id,
                                                            candidate_event_id=node_to_expand.edge_id,
                                                            memory_label=memory_label)
            oracle_call_time = time.time_ns() - oracle_call_start_time
            EVALUATION_STATE_CACHE[full_hash] = PredictionResult(oracle_call_time, prediction)
            return prediction, oracle_call_time, 0

    def _run_node_expansion(self, explained_edge_id: int, node_to_expand: MCTSTreeNode, sampler: EdgeSampler):
        prediction, oracle_call_time, cache_save_time = (
            self._get_evaluation_subgraph_prediction(candidate_events=sampler.subgraph[COL_ID],
                                                     node_to_expand=node_to_expand,
                                                     explained_event_id=explained_edge_id,
                                                     memory_label=EXPLAINED_EVENT_MEMORY_LABEL))
        self._expand_node(explained_edge_id, node_to_expand, prediction, sampler)
        return oracle_call_time, cache_save_time

    def evaluate_explanation(self, explained_event_id: int, original_prediction: float) -> (
            EvaluationCounterFactualExample):
        if original_prediction is None:
            original_prediction, sampler = self.initialize_explanation(explained_event_id)
        else:
            sampler = self.initialize_explanation_evaluation(explained_event_id, original_prediction)
        timings = {}
        statistics = {}
        oracle_calls = 0
        oracle_call_time = 0
        cache_saved_oracle_call_time = 0
        encountered_cf_examples = 0
        start_time = time.time_ns()

        best_cf_example = None
        best_cf_example_step = 0
        step = 0
        skip_search = False
        max_depth = sys.maxsize
        root_node = MCTSTreeNode(explained_event_id, parent=None, sampling_rank=0,
                                 original_prediction=original_prediction)
        self._expand_node(explained_event_id, root_node, original_prediction, sampler)

        if type(sampler) is OneBestEdgeSampler:
            for child in root_node.children:
                # Expand all children
                exp_oracle_call_time, exp_cache_save_time = self._run_node_expansion(explained_event_id, child, sampler)
                oracle_call_time += exp_oracle_call_time
                cache_saved_oracle_call_time += exp_cache_save_time
                oracle_calls += 1
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
                skip_search = True
            step += 1
        init_end_time = time.time_ns()
        timings['init_duration'] = init_end_time - start_time
        while step <= self.max_steps and not skip_search:
            node_to_expand = None
            while node_to_expand is None:
                node_to_expand = root_node.select_next_leaf(max_depth)
                if node_to_expand.depth > max_depth:
                    # Should not happen. TODO: Check if it is save to remove this if-condition
                    node_to_expand.expansion_backpropagation()
                    node_to_expand = None
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
            exp_oracle_call_time, exp_cache_save_time = self._run_node_expansion(explained_event_id, node_to_expand,
                                                                                 sampler)
            if self.verbose:
                self.logger.info(f'Selected node {node_to_expand.edge_id} at depth {node_to_expand.depth}, '
                                 f'prediction: {node_to_expand.prediction}, '
                                 f'exploitation score: {node_to_expand.exploitation_score}, hash: '
                                 f'{node_to_expand.hash()}')
            oracle_call_time += exp_oracle_call_time
            cache_saved_oracle_call_time += exp_cache_save_time
            oracle_calls += 1
            if node_to_expand.is_counterfactual:
                if best_cf_example is None or best_cf_example.depth > node_to_expand.depth:
                    best_cf_example = node_to_expand
                    best_cf_example_step = step
                    encountered_cf_examples += 1
                elif (best_cf_example.depth == node_to_expand.depth and
                      best_cf_example.exploitation_score < node_to_expand.exploitation_score):
                    best_cf_example = node_to_expand
                    best_cf_example_step = step
                    encountered_cf_examples += 1
                max_depth = best_cf_example.depth
                if self.verbose:
                    self.logger.info(f'Found counterfactual explanation: '
                                     + str(node_to_expand.to_cf_example()))
            step += 1
        if best_cf_example is None:
            best_cf_example = find_best_non_cf_example(root_node)
            best_cf_example_step = step
        self.tgnn_bridge.remove_memory_backup(EXPLAINED_EVENT_MEMORY_LABEL)
        self.tgnn_bridge.reset_model()
        self.known_states = {}
        end_time = time.time_ns()
        timings['oracle_call_duration'] = oracle_call_time
        timings['explanation_duration'] = end_time - start_time - oracle_call_time + cache_saved_oracle_call_time
        timings['total_duration'] = end_time - start_time + cache_saved_oracle_call_time
        statistics['oracle_calls'] = oracle_calls
        statistics['candidate_size'] = len(sampler.subgraph)
        statistics['candidates'] = sampler.subgraph[COL_ID].to_list()
        statistics['cf_example_step'] = best_cf_example_step
        statistics['encountered_cf_examples'] = encountered_cf_examples
        cf_ex = best_cf_example.to_cf_example()
        eval_cf_example = EvaluationCounterFactualExample(explained_event_id=explained_event_id,
                                                          original_prediction=original_prediction,
                                                          counterfactual_prediction=cf_ex.counterfactual_prediction,
                                                          achieves_counterfactual_explanation=
                                                          cf_ex.achieves_counterfactual_explanation,
                                                          event_ids=cf_ex.event_ids,
                                                          event_importances=cf_ex.event_importances,
                                                          timings=timings,
                                                          statistics=statistics)
        return eval_cf_example
