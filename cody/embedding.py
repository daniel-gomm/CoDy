import numpy as np
import torch

from cody.implementations.connector import TGNNWrapper
from cody.data import ContinuousTimeDynamicGraphDataset


class Embedding:
    double_dimension: int
    single_dimension: int

    def __init__(self, dataset: ContinuousTimeDynamicGraphDataset, tgnn: TGNNWrapper):
        self.dataset = dataset
        self.tgnn = tgnn

    def get_double_embedding(self, event_ids: np.ndarray, explained_event_id: int):
        edge_embeddings, explained_edge_embedding = self.get_embeddings(event_ids, explained_event_id)
        explained_edge_embeddings = torch.tile(explained_edge_embedding, (len(edge_embeddings), 1))
        return torch.concatenate((edge_embeddings, explained_edge_embeddings), dim=1)

    def get_embeddings(self, event_ids: np.ndarray, explained_event_id: int):
        raise NotImplementedError

    def extract_static_features(self, event_ids: np.ndarray, explained_event_id: int):
        all_event_ids = np.concatenate([event_ids, np.array([explained_event_id])])
        edge_mask = np.isin(self.dataset.edge_ids, all_event_ids)
        involved_source_nodes = self.dataset.source_node_ids[edge_mask]
        involved_target_nodes = self.dataset.target_node_ids[edge_mask]

        source_node_features = self.dataset.node_features[involved_source_nodes]
        target_node_features = self.dataset.node_features[involved_target_nodes]
        edge_features = self.dataset.edge_features[edge_mask]
        timestamp_embeddings = self.tgnn.encode_timestamps(self.dataset.timestamps[edge_mask])

        return (source_node_features, target_node_features, edge_features, timestamp_embeddings, involved_source_nodes,
                involved_target_nodes)


class StaticEmbedding(Embedding):

    def __init__(self, dataset: ContinuousTimeDynamicGraphDataset, tgnn: TGNNWrapper):
        super().__init__(dataset, tgnn)
        time_embedding_dimension = tgnn.time_embedding_dimension
        node_features = self.dataset.node_features.shape[1]
        edge_features = self.dataset.edge_features.shape[1]
        self.single_dimension = (2 * node_features + edge_features + time_embedding_dimension)
        self.double_dimension = self.single_dimension * 2

    def get_embeddings(self, event_ids: np.ndarray, explained_event_id: int):
        (source_node_features, target_node_features, edge_features,
         timestamp_embeddings, _, _) = self.extract_static_features(event_ids, explained_event_id)

        edge_embeddings = torch.cat((torch.tensor(source_node_features, dtype=torch.float32, device=self.tgnn.device),
                                     torch.tensor(target_node_features, dtype=torch.float32, device=self.tgnn.device),
                                     torch.tensor(edge_features, dtype=torch.float32, device=self.tgnn.device),
                                     timestamp_embeddings.squeeze()), dim=1)

        explained_edge_embedding = edge_embeddings[-1]
        edge_embeddings = edge_embeddings[:-1]
        return edge_embeddings, explained_edge_embedding


class DynamicEmbedding(Embedding):

    def __init__(self, dataset: ContinuousTimeDynamicGraphDataset, tgnn: TGNNWrapper,
                 embed_static_node_features: bool = False):
        super().__init__(dataset, tgnn)
        self.embed_static_node_features = embed_static_node_features
        node_embedding_dimension = tgnn.node_embedding_dimension
        time_embedding_dimension = tgnn.time_embedding_dimension
        node_features = self.dataset.node_features.shape[1]
        edge_features = self.dataset.edge_features.shape[1]
        if embed_static_node_features:
            self.single_dimension = (2 * node_embedding_dimension + 2 * node_features + edge_features +
                                     time_embedding_dimension)
        else:
            self.single_dimension = (2 * node_embedding_dimension + edge_features + time_embedding_dimension)
        self.double_dimension = self.single_dimension * 2

    def get_embeddings(self, event_ids: np.ndarray, explained_event_id: int):
        self.tgnn.set_evaluation_mode(True)
        (source_node_features, target_node_features, edge_features, timestamp_embeddings, involved_source_nodes,
         involved_target_nodes) = self.extract_static_features(event_ids, explained_event_id)

        _, _, explained_timestamp, _ = self.tgnn.extract_event_information(explained_event_id)
        current_timestamp_repeated = np.repeat(explained_timestamp, len(involved_source_nodes))
        all_event_ids = np.concatenate([event_ids, np.array([explained_event_id])])
        source_embeddings, target_embeddings = self.tgnn.compute_embeddings(involved_source_nodes,
                                                                            involved_target_nodes,
                                                                            current_timestamp_repeated,
                                                                            all_event_ids, negative_nodes=None)

        if self.embed_static_node_features:
            edge_embeddings = torch.cat((source_embeddings, target_embeddings,
                                         torch.tensor(source_node_features, dtype=torch.float32,
                                                      device=self.tgnn.device),
                                         torch.tensor(target_node_features, dtype=torch.float32,
                                                      device=self.tgnn.device),
                                         torch.tensor(edge_features, dtype=torch.float32, device=self.tgnn.device),
                                         timestamp_embeddings.squeeze()), dim=1)
        else:
            edge_embeddings = torch.cat((source_embeddings, target_embeddings,
                                         torch.tensor(edge_features, dtype=torch.float32, device=self.tgnn.device),
                                         timestamp_embeddings.squeeze()), dim=1)

        explained_edge_embedding = edge_embeddings[-1]
        edge_embeddings = edge_embeddings[:-1]

        return edge_embeddings, explained_edge_embedding
