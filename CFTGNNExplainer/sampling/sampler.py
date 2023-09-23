from typing import List

import torch.nn as nn
import numpy as np
import pandas as pd
from CFTGNNExplainer.constants import COL_ID, COL_SUBGRAPH_DISTANCE, COL_TIMESTAMP
from CFTGNNExplainer.sampling.embedding import Embedding

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


class PretrainedEdgeSampler(EdgeSampler):

    def __init__(self, subgraph: pd.DataFrame, prediction_model: nn.Module, embedding: Embedding):
        pass
    def sample(self, base_event_id: int, excluded_events: np.ndarray, size: int,
               known_cf_examples: List[np.ndarray] | None = None) -> np.ndarray:
        # TODO: Sample events based on scoring by pretrained model
        raise NotImplementedError