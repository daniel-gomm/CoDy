from dataclasses import dataclass
from typing import List

import torch
import numpy as np
import pandas as pd
from CFTGNNExplainer.constants import COL_ID, COL_SUBGRAPH_DISTANCE, COL_TIMESTAMP
from CFTGNNExplainer.sampling.embedding import Embedding


def load_prediction_model(embedding_dim: int, model_path: str = None, device: str = 'cpu') -> torch.nn.Module:
    prediction_model = torch.nn.Sequential(
        torch.nn.Linear(embedding_dim, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 1)
    )
    if model_path is not None:
        prediction_model.load_state_dict(torch.load(model_path))
    prediction_model.to(torch.device(device))
    return prediction_model


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
        raise NotImplementedError


class RandomEdgeSampler(EdgeSampler):

    def sample(self, base_event_id: int, excluded_events: np.ndarray, size: int,
               known_cf_examples: List[np.ndarray] | None = None) -> np.ndarray:
        filtered_subgraph = filter_subgraph(base_event_id, excluded_events, self.subgraph, known_cf_examples)
        if len(filtered_subgraph) < size:
            return filtered_subgraph[COL_ID].to_numpy()
        return filtered_subgraph.sample(n=size, replace=False)[COL_ID].to_numpy()


class RecentEdgeSampler(EdgeSampler):

    def sample(self, base_event_id: int, excluded_events: np.ndarray, size: int,
               known_cf_examples: List[np.ndarray] | None = None) -> np.ndarray:
        filtered_subgraph = filter_subgraph(base_event_id, excluded_events, self.subgraph, known_cf_examples)
        if len(filtered_subgraph) < size:
            return filtered_subgraph[COL_ID].to_numpy()
        return filtered_subgraph[COL_ID].to_numpy()[-size:]


class ClosestEdgeSampler(EdgeSampler):

    def sample(self, base_event_id: int, excluded_events: np.ndarray, size: int,
               known_cf_examples: List[np.ndarray] | None = None) -> np.ndarray:
        filtered_subgraph = filter_subgraph(base_event_id, excluded_events, self.subgraph, known_cf_examples)
        if len(filtered_subgraph) < size:
            return filtered_subgraph[COL_ID].to_numpy()
        sorted_subgraph = filtered_subgraph.sort_values(by=[COL_TIMESTAMP, COL_SUBGRAPH_DISTANCE],
                                                        ascending=[True, False])
        return sorted_subgraph[COL_ID].to_numpy()[:size]


@dataclass
class PretrainedEdgeSamplerParameters:
    prediction_model: torch.nn.Module
    embedding: Embedding
    predict_for_each_sample: bool


class PretrainedEdgeSampler(EdgeSampler):

    def __init__(self, subgraph: pd.DataFrame, parameters: PretrainedEdgeSamplerParameters, explained_event_id: int,
                 original_prediction: float):
        super().__init__(subgraph)
        self.prediction_model = parameters.prediction_model
        self.embedding = parameters.embedding
        self.sort_weights_ascending = (original_prediction > 0)
        self.initial_weights = None
        self.prediction_model.eval()
        if not parameters.predict_for_each_sample:
            subgraph_ids = self.subgraph[COL_ID].to_numpy()
            embeddings = self.embedding.get_embedding(subgraph_ids, explained_event_id)
            weights = self.prediction_model(embeddings)
            self.initial_weights = weights.detach().cpu().flatten().numpy()

    def sample(self, base_event_id: int, excluded_events: np.ndarray, size: int,
               known_cf_examples: List[np.ndarray] | None = None) -> np.ndarray:
        filtered_subgraph = filter_subgraph(base_event_id, excluded_events, self.subgraph, known_cf_examples)
        event_ids = filtered_subgraph[COL_ID].to_numpy()
        if len(event_ids) < size:
            return event_ids
        if self.initial_weights is None:
            embeddings = self.embedding.get_embedding(event_ids, base_event_id)
            weights = self.prediction_model(embeddings)
            weights = weights.detach().cpu().flatten().numpy()
        else:
            subgraph_without_base_event = self.subgraph[self.subgraph[COL_ID] != base_event_id]
            edge_mask = subgraph_without_base_event[COL_ID].isin(event_ids).to_numpy()
            weights = self.initial_weights[edge_mask]
        filtered_subgraph['weights'] = weights
        sorted_subgraph = filtered_subgraph.sort_values(by='weights', ascending=self.sort_weights_ascending)
        return sorted_subgraph[COL_ID].to_numpy()[:size]
