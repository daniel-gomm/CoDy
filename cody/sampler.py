from dataclasses import dataclass
from typing import List

import torch
import numpy as np
import pandas as pd
from cody.constants import COL_ID, COL_SUBGRAPH_DISTANCE, COL_TIMESTAMP
from cody.embedding import Embedding


def create_embedding_model(emb: Embedding, model_path: str = None, device: str = 'cpu'):
    embedding_model = torch.nn.Sequential(
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(emb.single_dimension, 32),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(32, 32)
    )
    if model_path is not None:
        embedding_model.load_state_dict(torch.load(model_path))
    embedding_model.to(torch.device(device))
    return embedding_model


def filter_subgraph(base_event_id: int, excluded_events: np.ndarray, subgraph: pd.DataFrame,
                    known_cf_examples: List[np.ndarray] | None = None) -> pd.DataFrame:
    excluded_events = np.concatenate((excluded_events, np.array([base_event_id])))
    filtered_subgraph = subgraph[~subgraph[COL_ID].isin(excluded_events)]
    # Make sure that events that would lead to an already known cf example are not sampled as candidates
    further_events_to_exclude = []
    if known_cf_examples is not None:
        for cf_example in known_cf_examples:
            total_occurrences = np.sum(np.isin(excluded_events, cf_example))
            if total_occurrences >= len(cf_example) - 1:
                already_excluded_events = excluded_events[np.isin(excluded_events, cf_example)]
                event_to_exclude = cf_example[~np.isin(cf_example, already_excluded_events)][0]
                further_events_to_exclude.append(event_to_exclude)
    return filtered_subgraph[~filtered_subgraph[COL_ID].isin(further_events_to_exclude)]


class EdgeSampler:

    def __init__(self, subgraph: pd.DataFrame):
        assert len(subgraph) > 0
        self.subgraph = subgraph

    def sample(self, base_event_id: int, excluded_events: np.ndarray, size: int,
               known_cf_examples: List[np.ndarray] | None = None) -> np.ndarray:
        ranked_subgraph = self.rank_subgraph(base_event_id, excluded_events, known_cf_examples)
        if len(ranked_subgraph) < size:
            return ranked_subgraph
        return ranked_subgraph[:size]

    def rank_subgraph(self, base_event_id: int, excluded_events: np.ndarray,
                      known_cf_examples: List[np.ndarray] | None = None):
        raise NotImplementedError


class RandomEdgeSampler(EdgeSampler):

    def rank_subgraph(self, base_event_id: int, excluded_events: np.ndarray,
                      known_cf_examples: List[np.ndarray] | None = None):
        filtered_subgraph = filter_subgraph(base_event_id, excluded_events, self.subgraph, known_cf_examples)
        return filtered_subgraph.sample(frac=1)[COL_ID].to_numpy()


class RecentEdgeSampler(EdgeSampler):

    def rank_subgraph(self, base_event_id: int, excluded_events: np.ndarray,
                      known_cf_examples: List[np.ndarray] | None = None):
        filtered_subgraph = filter_subgraph(base_event_id, excluded_events, self.subgraph, known_cf_examples)
        return filtered_subgraph[COL_ID].to_numpy()[::-1]


class ClosestEdgeSampler(EdgeSampler):

    def rank_subgraph(self, base_event_id: int, excluded_events: np.ndarray,
                      known_cf_examples: List[np.ndarray] | None = None):
        filtered_subgraph = filter_subgraph(base_event_id, excluded_events, self.subgraph, known_cf_examples)
        sorted_subgraph = filtered_subgraph.sort_values(by=[COL_TIMESTAMP, COL_SUBGRAPH_DISTANCE],
                                                        ascending=[True, False])
        return sorted_subgraph[COL_ID].to_numpy()


@dataclass
class PretrainedEdgeSamplerParameters:
    embedding_model: torch.nn.Module
    embedding: Embedding
    predict_for_each_sample: bool


class PretrainedEdgeSampler(EdgeSampler):

    def __init__(self, subgraph: pd.DataFrame, parameters: PretrainedEdgeSamplerParameters, explained_event_id: int,
                 original_prediction: float):
        super().__init__(subgraph)
        self.embedding_model = parameters.embedding_model
        self.embedding = parameters.embedding
        self.positive_original_prediction = (original_prediction > 0)
        self.initial_weights = None
        self.embedding_model.eval()
        if not parameters.predict_for_each_sample:
            subgraph_ids = self.subgraph[COL_ID].to_numpy()
            weights = self._embeddings_to_weights(subgraph_ids, explained_event_id)
            self.initial_weights = weights.detach().cpu().flatten().numpy()

    def _embeddings_to_weights(self, event_ids, base_event_id):
        excluded_edges_embeddings, explained_edge_embedding = self.embedding.get_embeddings(event_ids, base_event_id)
        explained_edge_embeddings = torch.tile(explained_edge_embedding, (len(excluded_edges_embeddings), 1))
        predictions = torch.nn.functional.cosine_similarity(explained_edge_embeddings, excluded_edges_embeddings)
        return predictions.detach().cpu().flatten().numpy()

    def rank_subgraph(self, base_event_id: int, excluded_events: np.ndarray,
                      known_cf_examples: List[np.ndarray] | None = None):
        filtered_subgraph = filter_subgraph(base_event_id, excluded_events, self.subgraph, known_cf_examples)
        event_ids = filtered_subgraph[COL_ID].to_numpy()
        if self.initial_weights is None:
            weights = self._embeddings_to_weights(event_ids, base_event_id)
        else:
            subgraph_without_base_event = self.subgraph[self.subgraph[COL_ID] != base_event_id]
            edge_mask = subgraph_without_base_event[COL_ID].isin(event_ids).to_numpy()
            weights = self.initial_weights[edge_mask]
        filtered_subgraph['weight'] = weights
        sorted_subgraph = filtered_subgraph.sort_values(by='weight', ascending=self.positive_original_prediction)
        return sorted_subgraph[COL_ID].to_numpy()


class OneBestEdgeSampler(EdgeSampler):

    def __init__(self, subgraph: pd.DataFrame):
        super().__init__(subgraph)
        self.subgraph['weight'] = 0

    def set_event_weight(self, event_id: int, weight: float):
        self.subgraph.loc[self.subgraph[COL_ID] == event_id, 'weight'] = weight

    def rank_subgraph(self, base_event_id: int, excluded_events: np.ndarray,
                      known_cf_examples: List[np.ndarray] | None = None):
        filtered_subgraph = filter_subgraph(base_event_id, excluded_events, self.subgraph, known_cf_examples)
        sorted_subgraph = filtered_subgraph.sort_values(by='weight', ascending=False)
        return sorted_subgraph[COL_ID].to_numpy()
