import copy
import math
import time
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass

import numpy as np
import torch
from pandas import DataFrame

from cody.embedding import Embedding
from cody.explainer.baseline.pgexplainer import TPGExplainer
from cody.explainer.baseline.common import greedy_highest_value_over_array, k_hop_temporal_subgraph
from cody.implementations.ttgn import TTGNWrapper
from cody.constants import COL_TIMESTAMP, COL_NODE_U, COL_NODE_I, COL_ID
from cody.explainer.base import Explainer
from cody.utils import ProgressBar
from TTGN.model.tgn import TGN


# Reimplementation of T-GNNExplainer by Xia et al. https://openreview.net/forum?id=BR_ZhvcYbGJ most of the code is
#  directly copied from the original implementation, alongside the TTGN (T-GNNExplainer TGN) version of the TGN model

def _agg_attention(model: TGN):
    attention_weights_list = model.embedding_module.atten_weights_list

    e_idx_weight_dict = {}
    for item in attention_weights_list:
        edge_idxs = item['src_ngh_eidx']
        weights = item['attn_weight']

        edge_idxs = edge_idxs.detach().cpu().numpy().flatten()
        weights = weights.detach().cpu().numpy().flatten()

        for e_idx, w in zip(edge_idxs, weights):
            if e_idx_weight_dict.get(e_idx, None) is None:
                e_idx_weight_dict[e_idx] = [w, ]
            else:
                e_idx_weight_dict[e_idx].append(w)

    for e_idx in e_idx_weight_dict.keys():
        e_idx_weight_dict[e_idx] = np.mean(e_idx_weight_dict[e_idx])

    return e_idx_weight_dict


def find_best_node_result(all_nodes, min_atoms=6):
    """ return the highest reward tree_node with its subgraph is smaller than max_nodes """
    all_nodes = filter(lambda x: len(x.coalition) <= min_atoms, all_nodes)  # filter using the min_atoms
    best_node = max(all_nodes, key=lambda x: x.P)
    return best_node


class MCTSNode(object):
    def __init__(self, coalition: list = None, created_by_remove: int = None,
                 c_puct: float = 10.0, w: float = 0, n: int = 0, p: float = 0, sparsity: float = 1,
                 ):
        self.coalition = coalition  # in our case, the coalition should be edge indices?
        self.c_puct = c_puct
        self.children = []
        self.created_by_remove = created_by_remove  # created by remove which edge from its parents
        self.W = w  # sum of node value
        self.N = n  # times of arrival
        self.P = p  # property score (reward)
        self.Sparsity = sparsity  # len(self.coalition)/len(candidates)

    def q(self):
        return self.W / self.N if self.N > 0 else 0

    def u(self, n):
        return self.c_puct * math.sqrt(n) / (1 + self.N)

    @property
    def info(self):
        info_dict = {
            'coalition': self.coalition,
            'created_by_remove': self.created_by_remove,
            'c_puct': self.c_puct,
            'W': self.W,
            'N': self.N,
            'P': self.P,
            'Sparsity': self.Sparsity,
        }
        return info_dict

    def load_info(self, info_dict):
        self.coalition = info_dict['coalition']
        self.created_by_remove = info_dict['created_by_remove']
        self.c_puct = info_dict['c_puct']
        self.W = info_dict['W']
        self.N = info_dict['N']
        self.P = info_dict['P']
        self.Sparsity = info_dict['Sparsity']

        self.children = []
        return self


def compute_scores(tgnn: TTGNWrapper, base_events, children, target_event_idx):
    results = []
    oracle_call_time = 0
    oracle_calls = 0
    original_prediction = tgnn.original_score
    for child in children:
        if child.P == 0:
            before_oracle_call = time.time_ns()
            with torch.no_grad():
                subgraph_prediction, _ = tgnn.predict(target_event_idx,
                                                      edge_id_preserve_list=base_events + child.coalition)
            subgraph_prediction = subgraph_prediction.detach().cpu().item()
            oracle_call_time += time.time_ns() - before_oracle_call
            oracle_calls += 1
            if original_prediction >= 0:
                reward = subgraph_prediction - original_prediction
            else:
                reward = original_prediction - subgraph_prediction
        else:
            reward = child.P
        results.append(reward)
    return results, oracle_call_time, oracle_calls


class MCTS(object):

    def __init__(self, events: DataFrame, tgnn: TTGNWrapper, candidate_events=None, base_events=None,
                 candidate_initial_weights=None, node_idx: int = None, event_idx: int = None, n_rollout: int = 10,
                 min_atoms: int = 5, c_puct: float = 10.0):

        self.run_time = 0
        self.events = events  # subgraph events or total events? subgraph events
        self.subgraph_num_nodes = self.events[COL_NODE_U].nunique() + self.events[COL_NODE_I].nunique()
        self.node_idx = node_idx  # node index to explain
        self.event_idx = event_idx  # event index to explain

        # improve the strategy later
        self.candidate_events = candidate_events
        self.base_events = base_events
        self.candidate_initial_weights = candidate_initial_weights

        # we only care these events, other events are preserved as is.
        # currently only take 10 temporal edges into consideration.

        self.num_nodes = self.events[COL_NODE_U].nunique() + self.events[COL_NODE_I].nunique()

        self.tgnn = tgnn

        self.n_rollout = n_rollout
        self.min_atoms = min_atoms
        self.c_puct = c_puct
        self.new_node_idx = None

        self.oracle_call_time = 0
        self.oracle_calls = 0

        self._initialize_tree()
        self._initialize_recorder()

    def _initialize_recorder(self):
        self.recorder = {
            'rollout': [],
            'runtime': [],
            'best_reward': [],
            'num_states': []
        }

    def mcts_rollout(self, tree_node):
        """
        The tree_node now is a set of events
        """
        if len(tree_node.coalition) < 1:
            return tree_node.P  # its score

        # Expand if this node has never been visited
        # Expand if this node has un-expanded children
        if len(tree_node.children) != len(tree_node.coalition):

            exist_children = set(map(lambda x: x.created_by_remove, tree_node.children))
            not_exist_children = list(filter(lambda e_idx: e_idx not in exist_children, tree_node.coalition))

            expand_events = self._select_expand_candidates(not_exist_children)

            new_tree_node = None

            for event in expand_events:
                important_events = [e_idx for e_idx in tree_node.coalition if e_idx != event]

                # check the state map and merge the same sub-tg-graph (node in the tree)
                find_same = False
                sub_node_coalition_key = self._node_key(important_events)
                for key in self.state_map.keys():
                    if key == sub_node_coalition_key:
                        new_tree_node = self.state_map[key]
                        find_same = True
                        break

                if not find_same:
                    new_tree_node = MCTSNode(
                        coalition=important_events,
                        created_by_remove=event,
                        c_puct=self.c_puct,
                        sparsity=len(important_events) / len(self.candidate_events)
                    )

                    self.state_map[sub_node_coalition_key] = new_tree_node

                # find same child ?
                find_same_child = False
                for child in tree_node.children:
                    if self._node_key(child.coalition) == self._node_key(new_tree_node.coalition):
                        find_same_child = True
                        break
                if not find_same_child:
                    tree_node.children.append(new_tree_node)

                # continue until one valid child is expanded, otherwise this rollout will be wasted
                if not find_same:
                    break
                else:
                    continue

            # compute scores of all children
            scores, compute_oracle_call_time, compute_oracle_calls = compute_scores(self.tgnn, self.base_events,
                                                                                    tree_node.children,
                                                                                    self.event_idx)
            self.oracle_call_time += compute_oracle_call_time
            self.oracle_calls += compute_oracle_calls
            for child, score in zip(tree_node.children, scores):
                child.P = score

        # If this node has children (it has been visited), then directly select one child
        sum_count = sum([c.N for c in tree_node.children])
        selected_node = max(tree_node.children, key=lambda x: self._compute_node_score(x, sum_count))

        v = self.mcts_rollout(selected_node)
        selected_node.W += v
        selected_node.N += 1
        return v

    def _select_expand_candidates(self, not_exist_children):
        assert self.candidate_initial_weights is not None
        return sorted(not_exist_children, key=self.candidate_initial_weights.get)

    def _compute_node_score(self, node, sum_count):
        """
        score for selecting a path
        """
        tscore_coefficient = 0
        beta = -3

        max_event_idx = max(self.root.coalition)
        curr_t = self.events[COL_TIMESTAMP][max_event_idx]
        ts = self.events[COL_TIMESTAMP][self.events[COL_ID].isin(node.coalition)].values
        delta_ts = curr_t - ts
        t_score_exp = np.exp(beta * delta_ts)
        t_score_exp = np.sum(t_score_exp)

        # uct score
        uct_score = node.q() + node.u(sum_count)

        # final score
        final_score = uct_score + tscore_coefficient * t_score_exp

        return final_score

    def mcts(self, verbose=True):
        if verbose:
            print(f"The nodes in graph is {self.subgraph_num_nodes}")

        start_time = time.time()
        progress_bar = ProgressBar(max_item=self.n_rollout, prefix='Simulating MCTS')
        for rollout_idx in range(self.n_rollout):
            progress_bar.next()
            self.mcts_rollout(self.root)
            elapsed_time = time.time() - start_time
            progress_bar.update_postfix(f'states: {len(self.state_map)}')
            # record
            self.recorder['rollout'].append(rollout_idx)
            self.recorder['runtime'].append(elapsed_time)
            curr_best_node = find_best_node_result(self.state_map.values(), self.min_atoms)
            self.recorder['best_reward'].append(curr_best_node.P)
            self.recorder['num_states'].append(len(self.state_map))

        end_time = time.time()
        self.run_time = end_time - start_time

        tree_nodes = list(self.state_map.values())
        progress_bar.close()
        return tree_nodes

    def _initialize_tree(self):
        # reset the search tree
        self.root_coalition = copy.copy(self.candidate_events)
        self.root = MCTSNode(self.root_coalition, created_by_remove=-1, c_puct=self.c_puct, sparsity=1.0)
        self.root_key = self._node_key(self.root_coalition)
        self.state_map = {self.root_key: self.root}

        max_event_idx = max(self.root.coalition)
        self.curr_t = self.events[COL_TIMESTAMP][max_event_idx]

    @staticmethod
    def _node_key(coalition):
        return "_".join(map(lambda x: str(x), sorted(coalition)))  # NOTE: have sorted


@dataclass
class TGNNExplainerExplanation:
    explained_event_id: int
    original_prediction: float
    best_prediction: float
    results: List[Dict]
    timings: Dict
    statistics: Dict
    tree_nodes: List[MCTSNode]

    def to_dict(self) -> Dict:
        results = {
            'explained_event_id': self.explained_event_id,
            'original_prediction': self.original_prediction,
            'best_prediction': self.best_prediction,
            'results': self.results
        }
        results.update(self.statistics)
        results.update(self.timings)
        return results


class TGNNExplainer(Explainer):

    def __init__(self, tgnn_wrapper: TTGNWrapper, embedding: Embedding, pg_explainer_model: TPGExplainer,
                 results_dir: str, device: str = 'cpu', rollout: int = 20, min_atoms: int = 1, c_puct: float = 10.0,
                 mcts_saved_dir: Optional[str] = None, save_results: bool = True):
        super().__init__(tgnn_wrapper)
        self.mcts_state_map = None
        self.tgnn = tgnn_wrapper
        self.embedding = embedding
        self.rollout = rollout
        self.min_atoms = min_atoms
        self.c_puct = c_puct

        self.results_dir = results_dir
        self.device = device
        self.save_results = save_results
        self.mcts_saved_dir = mcts_saved_dir
        self.pg_explainer = pg_explainer_model

    def write_from_mcts_node_list(self, mcts_node_list):
        if isinstance(mcts_node_list[0], MCTSNode):
            ret_list = [node.info for node in mcts_node_list]
        else:
            raise NotImplementedError
        return ret_list

    def _get_candidate_weights(self, event_idx):
        candidate_events = self.tgnn.candidate_events

        original_prediction, _ = self.tgnn.predict(event_idx,
                                                   edge_id_preserve_list=((candidate_events +
                                                                           self.tgnn.base_events)))
        original_prediction = original_prediction.detach().cpu().item()

        self.pg_explainer.explainer.eval()
        edge_weights = self.pg_explainer.get_event_scores(event_idx, candidate_events)

        _, _ = self.tgnn.predict(event_idx,
                                 candidate_event_ids=torch.tensor(candidate_events, dtype=torch.int64,
                                                                  device=self.device),
                                 edge_weights=edge_weights)

        e_idx_weight_dict = _agg_attention(self.tgnn.model)
        edge_weights = []
        for e_idx in candidate_events:
            if e_idx in e_idx_weight_dict.keys():
                edge_weights.append(e_idx_weight_dict[e_idx])
            else:
                edge_weights.append(0.0)
        edge_weights = np.array(edge_weights)

        candidate_initial_weights = {candidate_events[i]: edge_weights[i] for i in range(len(candidate_events))}
        return candidate_initial_weights, original_prediction

    def get_scores(self, event_idx: Optional[int] = None,
                   candidate_initial_weights=None, subgraph=None):
        #subgraph = k_hop_temporal_subgraph(self.tgnn.dataset.events, self.num_hops, event_idx)
        self.tgnn.initialize(event_idx, subgraph_event_ids=subgraph[COL_ID].to_numpy())
        assert event_idx is not None
        # search
        self.mcts_state_map = MCTS(events=subgraph,
                                   candidate_events=self.tgnn.candidate_events,
                                   base_events=self.tgnn.base_events,
                                   event_idx=event_idx,
                                   n_rollout=self.rollout,
                                   min_atoms=self.min_atoms,
                                   c_puct=self.c_puct,
                                   tgnn=self.tgnn,
                                   candidate_initial_weights=candidate_initial_weights)

        tree_nodes = self.mcts_state_map.mcts(verbose=self.verbose)  # search

        tree_node_x = find_best_node_result(tree_nodes, self.min_atoms)
        tree_nodes = sorted(tree_nodes, key=lambda x: x.P)

        return tree_nodes, tree_node_x

    def explain(self, explained_event_id: int) -> TGNNExplainerExplanation:
        timings = {}
        statistics = {}
        start_time = time.time_ns()
        self.tgnn.set_evaluation_mode(True)
        self.tgnn.reset_model()
        subgraph = k_hop_temporal_subgraph(self.tgnn.dataset.events, self.num_hops, explained_event_id)
        with torch.no_grad():
            self.tgnn.initialize(explained_event_id, subgraph_event_ids=subgraph[COL_ID].to_numpy())
            candidate_initial_weights, original_prediction = self._get_candidate_weights(event_idx=explained_event_id)
            init_end_time = time.time_ns()

        tree_nodes, tree_node_x = self.get_scores(event_idx=explained_event_id,
                                                  candidate_initial_weights=candidate_initial_weights,
                                                  subgraph=subgraph)
        self._save_mcts_recorder(explained_event_id)  # always store
        if self.save_results:  # sometimes store
            self._save_mcts_nodes_info(tree_nodes, explained_event_id)

        candidate_events = self.tgnn.candidate_events
        results = []
        for i in range(1, len(candidate_events) + 1):
            best_node_at_i = find_best_node_result(tree_nodes, i)
            results.append({
                'prediction': best_node_at_i.P,
                'event_ids_in_explanation': best_node_at_i.coalition
            })

        end_time = time.time_ns()
        oracle_call_time = self.mcts_state_map.oracle_call_time
        timings['oracle_call_duration'] = oracle_call_time
        timings['explanation_duration'] = end_time - init_end_time - oracle_call_time
        timings['init_duration'] = init_end_time - start_time
        timings['total_duration'] = end_time - start_time
        statistics['oracle_calls'] = self.mcts_state_map.oracle_calls
        statistics['candidate_size'] = len(candidate_events)
        statistics['candidates'] = np.array(candidate_events)

        tree_nodes.sort(key=lambda node: node.P, reverse=True)
        best_prediction = tree_nodes[0].P

        return TGNNExplainerExplanation(explained_event_id=explained_event_id, original_prediction=original_prediction,
                                        best_prediction=best_prediction, results=results, timings=timings,
                                        statistics=statistics, tree_nodes=tree_nodes)

    def evaluate_fidelity(self, explanation: TGNNExplainerExplanation) -> (List, List, List):
        tree_nodes = explanation.tree_nodes
        sparsity_list = []
        fidelity_list = []

        candidate_events = explanation.statistics['candidates']
        candidate_num = len(candidate_events)
        for node in tree_nodes:
            sparsity = len(node.coalition) / candidate_num
            assert np.isclose(sparsity, node.Sparsity)

            fidelity = node.P
            fidelity_list.append(fidelity)
            sparsity_list.append(sparsity)

        sparsity_list = np.array(sparsity_list)
        fidelity_list = np.array(fidelity_list)

        # sort according to sparsity
        sort_idx = np.argsort(sparsity_list)  # ascending of sparsity
        sparsity_list = sparsity_list[sort_idx]
        fidelity_list = fidelity_list[sort_idx]

        best_fidelity_at_depth = []
        for sparsity_val in np.unique(sparsity_list):
            best_fidelity_at_depth.append(fidelity_list[sparsity_list == sparsity_val].max())

        best_fidelity_at_depth = np.array(best_fidelity_at_depth)

        best_fidelity_list = greedy_highest_value_over_array(best_fidelity_at_depth)

        sparsity_list = np.unique(sparsity_list)

        sparsity_thresholds = np.arange(0, 1.05, 0.05)
        indices = []
        for sparsity in sparsity_thresholds:
            indices.append(np.where(sparsity_list <= sparsity)[0].max())

        fidelity_list = best_fidelity_at_depth[indices]
        best_fidelity_list = best_fidelity_list[indices]

        return sparsity_thresholds, fidelity_list, best_fidelity_list

    def _save_mcts_recorder(self, event_idx):
        # save records
        recorder_df = DataFrame(self.mcts_state_map.recorder)
        # ROOT_DIR.parent/'benchmarks'/'results'
        record_filename = self._mcts_recorder_path(self.results_dir, self.tgnn.name, self.dataset.name,
                                                   event_idx, suffix='')
        recorder_df.to_csv(record_filename, index=False)

        print(f'mcts recorder saved at {str(record_filename)}')

    def _save_mcts_nodes_info(self, tree_nodes, event_idx):
        if self.mcts_saved_dir is not None:
            saved_contents = {
                'saved_MCTSInfo_list': self.write_from_mcts_node_list(tree_nodes),
            }
            path = self._mcts_node_info_path(self.mcts_saved_dir, self.tgnn.name, self.dataset.name,
                                             event_idx, suffix='')
            torch.save(saved_contents, path)
            print(f'results saved at {path}')

    @staticmethod
    def _mcts_recorder_path(result_dir, model_name, dataset_name, event_idx, suffix):
        if suffix is not None:
            record_filename = result_dir + f'{model_name}_{dataset_name}_{event_idx}_mcts_recorder_{suffix}.csv'
        else:
            record_filename = result_dir + f'{model_name}_{dataset_name}_{event_idx}_mcts_recorder.csv'

        return record_filename

    @staticmethod
    def _mcts_node_info_path(node_info_dir, model_name, dataset_name, event_idx, suffix):
        if suffix is not None:
            node_info_filename = Path(
                node_info_dir) / f"{model_name}_{dataset_name}_{event_idx}_mcts_node_info_{suffix}.pt"
        else:
            node_info_filename = Path(node_info_dir) / f"{model_name}_{dataset_name}_{event_idx}_mcts_node_info.pt"

        return node_info_filename
