import logging

import numpy as np
import torch

from CFTGNNExplainer.data import BatchData, ContinuousTimeDynamicGraphDataset
from CFTGNNExplainer.utils import ProgressBar


class TGNNWrapper:
    node_embedding_dimension: int
    time_embedding_dimension: int

    def __init__(self, model: torch.nn.Module, dataset: ContinuousTimeDynamicGraphDataset, num_hops: int,
                 model_name: str, device: str = 'cpu'):
        self.num_hops = num_hops
        self.model = model
        self.dataset = dataset
        self.name = model_name
        self.device = device
        self.latest_event_id = 0
        self.evaluation_mode = False
        self.memory_backups_map = {}
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('TGNNWrapper')

    def initialize(self, event_id: int, show_progress: bool = False, memory_label: str = None):
        raise NotImplementedError

    def predict(self, event_id: int, result_as_logit: bool = False):
        raise NotImplementedError

    def rollout_until_event(self, event_id: int = None, batch_data: BatchData = None,
                            progress_bar: ProgressBar = None, event_ids_to_rollout: np.ndarray = None) -> None:
        raise NotImplementedError

    def compute_embeddings(self, source_nodes, target_nodes, edge_times, edge_ids, negative_nodes=None):
        raise NotImplementedError

    def encode_timestamps(self, timestamps: np.ndarray):
        raise NotImplementedError

    def compute_edge_probabilities(self, source_nodes: np.ndarray, target_nodes: np.ndarray,
                                   edge_timestamps: np.ndarray, edge_ids: np.ndarray,
                                   negative_nodes: np.ndarray | None = None, result_as_logit: bool = False,
                                   perform_memory_update: bool = True):
        raise NotImplementedError

    def compute_edge_probabilities_for_subgraph(self, event_id, edges_to_drop: np.ndarray,
                                                result_as_logit: bool = False,
                                                event_ids_to_rollout: np.ndarray = None) -> (torch.Tensor, torch.Tensor):
        raise NotImplementedError

    def get_memory(self):
        raise NotImplementedError

    def detach_memory(self):
        raise NotImplementedError

    def restore_memory(self, memory_backup, event_id):
        raise NotImplementedError

    def reset_model(self):
        raise NotImplementedError

    def remove_memory_backup(self, label: str):
        if label in self.memory_backups_map:
            del self.memory_backups_map[label]

    def set_evaluation_mode(self, activate_evaluation: bool):
        if activate_evaluation:
            self.model.eval()
            self.evaluation_mode = True
        else:
            self.model.train()
            self.evaluation_mode = False

    def post_batch_cleanup(self):
        pass

    def reset_latest_event_id(self, value: int = None):
        if value is not None:
            self.latest_event_id = value
        else:
            self.latest_event_id = 0

    def extract_event_information(self, event_ids: int | np.ndarray):
        edge_mask = np.isin(self.dataset.edge_ids, event_ids)
        source_nodes, target_nodes, timestamps = self.dataset.source_node_ids[edge_mask], \
            self.dataset.target_node_ids[edge_mask], self.dataset.timestamps[edge_mask]
        return source_nodes, target_nodes, timestamps, event_ids
