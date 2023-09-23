import numpy as np
import torch

from CFTGNNExplainer.baseline.ttgnwrapper import TTGNWrapper
from CFTGNNExplainer.data.dataset import ContinuousTimeDynamicGraphDataset


class Embedding:
    dimension: int

    def get_embedding(self, event_ids: np.ndarray, explained_event_id: int, explanation_mode: bool = True):
        raise NotImplementedError


class StaticEmbedding(Embedding):

    def __init__(self, dataset: ContinuousTimeDynamicGraphDataset, model: TTGNWrapper):
        self.dataset = dataset
        self.model = model
        time_embedding_dimension = model.time_embedding_dimension
        node_features = self.dataset.node_features.shape[1]
        edge_features = self.dataset.edge_features.shape[1]
        self.dimension = (2 * node_features + edge_features + time_embedding_dimension) * 2

    def get_embedding(self, event_ids: np.ndarray, explained_event_id: int, explanation_mode: bool = True):
        all_event_ids = np.concatenate([event_ids, np.array([explained_event_id])])
        edge_mask = np.isin(self.dataset.edge_ids, all_event_ids)
        involved_source_nodes = self.dataset.source_node_ids[edge_mask]
        involved_target_nodes = self.dataset.target_node_ids[edge_mask]

        source_node_features = self.dataset.node_features[involved_source_nodes]
        target_node_features = self.dataset.node_features[involved_target_nodes]
        edge_features = self.dataset.edge_features[edge_mask]
        timestamp_embeddings = self.model.encode_timestamps(self.dataset.timestamps[edge_mask])

        edge_embeddings = torch.cat((torch.tensor(source_node_features, dtype=torch.float32, device=self.model.device),
                                     torch.tensor(target_node_features, dtype=torch.float32, device=self.model.device),
                                     torch.tensor(edge_features, dtype=torch.float32, device=self.model.device),
                                     timestamp_embeddings.squeeze()), dim=1)

        explained_edge_embedding = edge_embeddings[-1]
        edge_embeddings = edge_embeddings[:-1]
        explained_edge_embeddings = torch.tile(explained_edge_embedding, (len(edge_embeddings), 1))

        return torch.concatenate((edge_embeddings, explained_edge_embeddings), dim=1)
