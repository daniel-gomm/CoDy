import numpy as np
import pandas as pd
from CFTGNNExplainer.constants import COL_ID


def filter_subgraph(base_event_id: int, excluded_events: np.ndarray, subgraph: pd.DataFrame) -> pd.DataFrame:
    excluded_events = np.concatenate((excluded_events, np.array([base_event_id])))
    return subgraph[~subgraph[COL_ID].isin(excluded_events)]


class EdgeSampler:

    def __init__(self, subgraph: pd.DataFrame):
        self.subgraph = subgraph

    def sample(self, base_event_id: int, excluded_events: np.ndarray, size: int) -> np.ndarray:
        raise NotImplementedError


class RandomEdgeSampler(EdgeSampler):

    def sample(self, base_event_id: int, excluded_events: np.ndarray, size: int) -> np.ndarray:
        filtered_subgraph = filter_subgraph(base_event_id, excluded_events, self.subgraph)
        if len(filtered_subgraph) < size:
            return filtered_subgraph[COL_ID].to_numpy()
        return filtered_subgraph.sample(n=size, replace=False)[COL_ID].to_numpy()


class RecentEdgeSampler(EdgeSampler):

    def sample(self, base_event_id: int, excluded_events: np.ndarray, size: int) -> np.ndarray:
        filtered_subgraph = filter_subgraph(base_event_id, excluded_events, self.subgraph)
        if len(filtered_subgraph) < size:
            return filtered_subgraph[COL_ID].to_numpy()
        return filtered_subgraph[COL_ID].to_numpy()[-size:]


class ClosestEdgeSampler(EdgeSampler):

    def sample(self, base_event_id: int, excluded_events: np.ndarray, size: int) -> np.ndarray:
        # This should sort the possible events by their distance to the
        raise NotImplementedError
