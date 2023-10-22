import itertools

import numpy as np
import torch

from CFTGNNExplainer.utils import ProgressBar
from TTGN.model.tgn import TGN

from CFTGNNExplainer.connector import TGNNWrapper, TGNNBridge
from CFTGNNExplainer.data import ContinuousTimeDynamicGraphDataset, BatchData
from TTGN.utils.data_processing import compute_time_statistics
from TTGN.utils.utils import NeighborFinder


# Implementation of the adjusted TGN model used by Xia et al. 2023 (https://openreview.net/forum?id=BR_ZhvcYbGJ)

def find_candidate_events(dataset: ContinuousTimeDynamicGraphDataset, neighborhood_finder: NeighborFinder,
                          target_event_idx: int, num_hops: int, candidates_size: int):
    target_mask = np.isin(dataset.edge_ids, target_event_idx)
    target_nodes = dataset.target_node_ids[target_mask][0]
    source_nodes = dataset.source_node_ids[target_mask][0]
    timestamps = dataset.timestamps[target_mask][0]

    accu_edge_idx = []
    accu_node = [[target_nodes, source_nodes, ]]
    accu_ts = [[timestamps, timestamps, ]]

    for i in range(num_hops):
        last_nodes = accu_node[-1]
        last_ts = accu_ts[-1]

        out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch = neighborhood_finder.get_temporal_neighbor(
            last_nodes,
            last_ts,
            n_neighbors=candidates_size,
            # edge_idx_preserve_list=edge_idx_preserve_list,  # NOTE: not needed?
        )

        out_ngh_node_batch = out_ngh_node_batch.flatten()
        out_ngh_eidx_batch = out_ngh_eidx_batch.flatten()
        out_ngh_t_batch = out_ngh_t_batch.flatten()

        mask = out_ngh_node_batch != 0
        out_ngh_node_batch = out_ngh_node_batch[mask]
        out_ngh_eidx_batch = out_ngh_eidx_batch[mask]
        out_ngh_t_batch = out_ngh_t_batch[mask]

        out_ngh_node_batch = out_ngh_node_batch.tolist()
        out_ngh_t_batch = out_ngh_t_batch.tolist()
        out_ngh_eidx_batch = (out_ngh_eidx_batch).tolist()

        accu_node.append(out_ngh_node_batch)
        accu_ts.append(out_ngh_t_batch)
        accu_edge_idx.append(out_ngh_eidx_batch)

    unique_e_idx = np.array(list(itertools.chain.from_iterable(accu_edge_idx)))
    unique_e_idx = unique_e_idx[unique_e_idx != 0]  # NOTE: 0 are padded e_idxs
    # unique_e_idx = unique_e_idx - 1 # NOTE: -1, because ngh_finder stored +1 e_idxs
    unique_e_idx = np.unique(unique_e_idx).tolist()

    candidate_events = unique_e_idx
    if len(candidate_events) > candidates_size:
        candidate_events = candidate_events[-candidates_size:]
        candidate_events = sorted(candidate_events)

    return candidate_events, unique_e_idx


class TTGNWrapper(TGNNWrapper):

    def __init__(self, model: TGN, dataset: ContinuousTimeDynamicGraphDataset, num_hops: int, model_name: str,
                 device: str = 'cpu', n_neighbors: int = 20, batch_size: int = 128, checkpoint_path: str = None):
        super().__init__(model, dataset, num_hops, model_name)

        model.mean_time_shift_src, model.std_time_shift_src, model.mean_time_shift_dst, model.std_time_shift_dst = \
            compute_time_statistics(self.dataset.source_node_ids, self.dataset.target_node_ids, self.dataset.timestamps)

        self.model = model
        self.n_neighbors = n_neighbors
        self.batch_size = batch_size
        self.device = device
        if checkpoint_path is not None:
            self.model.load_state_dict(torch.load(checkpoint_path))
            self.model.to(self.device)
        self.node_embedding_dimension = self.model.embedding_module.embedding_dimension
        self.time_embedding_dimension = self.model.time_encoder.dimension

    def compute_embeddings(self, source_nodes, target_nodes, edge_times, edge_ids,
                           negative_nodes=None,
                           candidate_event_ids: np.ndarray = None,
                           candidate_event_weights: np.ndarray = None):
        candidate_weights_dict = None
        if candidate_event_ids is not None and candidate_event_weights is not None:
            candidate_weights_dict = {
                'candidate_events': candidate_event_ids,
                'edge_weights': candidate_event_weights
            }
        return self.model.compute_temporal_embeddings(source_nodes, target_nodes, negative_nodes, edge_times, edge_ids,
                                                      n_neighbors=self.n_neighbors,
                                                      candidate_weights_dict=candidate_weights_dict)

    def rollout_until_event(self, event_id: int = None, batch_data: BatchData = None,
                            progress_bar: ProgressBar = None) -> None:
        assert event_id is not None or batch_data is not None
        if batch_data is None:
            batch_data = self.dataset.get_batch_data(self.latest_event_id, event_id)
        batch_id = 0
        number_of_batches = int(np.ceil(len(batch_data.source_node_ids) / self.batch_size))
        if progress_bar is not None:
            progress_bar.reset(number_of_batches)
        with torch.no_grad():
            for _ in range(number_of_batches):
                if progress_bar is not None:
                    progress_bar.next()
                batch_start = batch_id * self.batch_size
                batch_end = min((batch_id + 1) * self.batch_size, len(batch_data.source_node_ids))
                self.model.compute_temporal_embeddings(source_nodes=batch_data.source_node_ids[batch_start:batch_end],
                                                       destination_nodes=batch_data.target_node_ids[batch_start:
                                                                                                    batch_end],
                                                       edge_times=batch_data.timestamps[batch_start:batch_end],
                                                       edge_idxs=batch_data.edge_ids[batch_start:batch_end],
                                                       negative_nodes=None)
                self.model.memory.detach_memory()
                batch_id += 1

        self.latest_event_id = event_id

    def initialize(self, event_id, num_neighbors: int):
        candidate_events, unique_edge_ids = find_candidate_events(self.dataset, self.model.neighbor_finder, event_id,
                                                                  self.num_hops, num_neighbors)
        base_events = list(filter(lambda x: x not in set(candidate_events), unique_edge_ids))
        source_nodes, target_nodes, timestamps, event_ids = self.extract_event_information(event_id)

        if event_id in candidate_events:
            candidate_events.remove(event_id)

        original_score, _ = self.model.compute_edge_probabilities(source_nodes=source_nodes,
                                                                  destination_nodes=target_nodes,
                                                                  negative_nodes=None,
                                                                  edge_times=timestamps,
                                                                  edge_idxs=event_ids,
                                                                  perform_memory_update=False,
                                                                  edge_idx_preserve_list=base_events + candidate_events,
                                                                  result_as_logit=True)
        return candidate_events, unique_edge_ids, base_events, original_score

    def compute_edge_probabilities(self, source_nodes: np.ndarray, target_nodes: np.ndarray,
                                   edge_timestamps: np.ndarray, edge_ids: np.ndarray,
                                   negative_nodes: np.ndarray | None = None,
                                   candidate_event_ids: np.ndarray = None,
                                   candidate_event_weights: np.ndarray = None,
                                   edge_idx_preserve_list=None,
                                   result_as_logit: bool = False,
                                   perform_memory_update: bool = True):
        candidate_weights_dict = None
        if candidate_event_ids is not None and candidate_event_weights is not None:
            candidate_weights_dict = {
                'candidate_events': candidate_event_ids,
                'edge_weights': candidate_event_weights
            }
        return self.model.compute_edge_probabilities(source_nodes, target_nodes, negative_nodes, edge_timestamps,
                                                     edge_ids, self.n_neighbors, result_as_logit, perform_memory_update,
                                                     candidate_weights_dict=candidate_weights_dict,
                                                     edge_idx_preserve_list=edge_idx_preserve_list)

    def encode_timestamps(self, timestamps: np.ndarray):
        timestamps = torch.tensor(timestamps, dtype=torch.float32, device=self.device).reshape((1, -1))
        return self.model.time_encoder(timestamps)

    def get_current_node_embeddings(self, node_ids: np.ndarray, current_timestamp: int,
                                    candidate_event_ids: np.ndarray = None,
                                    candidate_event_weights: np.ndarray = None):
        candidate_weights_dict = None
        if candidate_event_ids is not None and candidate_event_weights is not None:
            candidate_weights_dict = {
                'candidate_events': candidate_event_ids,
                'edge_weights': candidate_event_weights
            }
        memory = None
        if self.model.use_memory:
            if self.model.memory_update_at_start:
                # Update memory for all nodes with messages stored in previous batches
                memory, last_update = self.model.get_updated_memory(list(range(self.model.n_nodes)),
                                                                    self.model.memory.messages)
            else:
                memory = self.model.memory.get_memory(list(range(self.model.n_nodes)))
        timestamps = np.repeat(current_timestamp, len(node_ids))

        return self.model.embedding_module.compute_embedding(memory=memory, source_nodes=node_ids, timestamps=timestamps
                                                             , n_layers=self.model.num_layers,
                                                             n_neighbors=self.n_neighbors,
                                                             candidate_weights_dict=candidate_weights_dict)

    def get_memory(self):
        return self.model.memory.backup_memory()

    def detach_memory(self):
        self.model.memory.detach_memory()

    def restore_memory(self, memory_backup, event_id):
        self.reset_model()
        self.model.memory.restore_memory(memory_backup)
        self.reset_latest_event_id(event_id)

    def reset_model(self):
        self.reset_latest_event_id()
        self.detach_memory()
        self.model.memory.__init_memory__()


class TTGNBridge(TGNNBridge):

    def __init__(self, model: TTGNWrapper, explanation_candidates_size: int):
        super().__init__(model)
        self.last_predicted_event_id = None
        self.model = model
        self.memory_backups_map = {}
        self.model.reset_model()
        self.model.reset_latest_event_id()
        self.explanation_candidates_size = explanation_candidates_size

    def initialize(self, event_id: int, show_progress: bool = False, memory_label: str = None):
        if show_progress:
            print(f'Initializing model for event {event_id}')

        (self.candidate_events,
         self.unique_edge_ids,
         self.base_events,
         original_score) = self.model.initialize(event_id, self.explanation_candidates_size)
        self.original_score = original_score.detach().cpu().item()
        self.last_predicted_event_id = event_id

    def initialize_static(self, event_id: int):
        self.model.reset_model()
        return self.model.initialize(event_id, self.explanation_candidates_size)

    def predict(self, event_id: int, candidate_event_ids=None, edge_weights=None, edge_id_preserve_list=None):
        source_node, target_node, timestamp, edge_id = self.model.extract_event_information(event_id)
        return self.model.compute_edge_probabilities(source_nodes=source_node,
                                                     target_nodes=target_node,
                                                     edge_timestamps=timestamp,
                                                     edge_ids=edge_id,
                                                     perform_memory_update=False,
                                                     candidate_event_ids=candidate_event_ids,
                                                     candidate_event_weights=edge_weights,
                                                     result_as_logit=True,
                                                     edge_idx_preserve_list=edge_id_preserve_list)

    def get_candidate_events(self, event_id: int):
        assert event_id == self.last_predicted_event_id, (f'Last event predicted {self.last_predicted_event_id} does '
                                                          f'not match with the provided event id {event_id}')
        return self.candidate_events

    def reset_model(self):
        self.model.reset_model()

    def remove_memory_backup(self, label: str):
        pass
