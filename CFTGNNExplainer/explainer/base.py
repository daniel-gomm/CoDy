import logging
from typing import List

import numpy as np
import pandas as pd
from dataclasses import dataclass

from CFTGNNExplainer.connector.bridge import TGNNBridge
from CFTGNNExplainer.constants import EXPLAINED_EVENT_MEMORY_LABEL, COL_ID
from CFTGNNExplainer.data.subgraph import SubgraphGenerator
from CFTGNNExplainer.explainer.sampler import EdgeSampler, RandomEdgeSampler, RecentEdgeSampler, ClosestEdgeSampler


@dataclass
class CounterFactualExample:
    explained_event_id: int
    original_prediction: float
    counterfactual_prediction: float
    achieves_counterfactual_explanation: bool
    event_ids: np.ndarray
    event_importances: np.ndarray

    def __str__(self):
        return (f'Counterfactual example including events: {str(self.event_ids.tolist())}, original prediction '
                f'{str(self.original_prediction)}, counterfactual prediction {str(self.counterfactual_prediction)}, '
                f'event importances {str(self.get_relative_importances().tolist())}')

    def get_absolute_importances(self) -> np.ndarray:
        return np.array([importance - np.sum(self.event_importances[index - 1:index]) for index, importance in
                         enumerate(self.event_importances)])

    def get_relative_importances(self) -> np.ndarray:
        if len(self.event_importances) == 0:
            return np.ndarray([])
        return self.get_absolute_importances() / self.event_importances[-1]


def calculate_prediction_delta(original_prediction: float, prediction_to_assess: float) -> float:
    if prediction_to_assess * original_prediction < 0:
        return abs(prediction_to_assess) + abs(original_prediction)
    return abs(original_prediction) - abs(prediction_to_assess)


class Explainer:

    def __init__(self, tgnn_bridge: TGNNBridge, sampling_strategy: str = 'recent', candidates_size: int = 75,
                 sample_size: int = 10, verbose: bool = False):
        self.tgnn_bridge = tgnn_bridge
        self.dataset = self.tgnn_bridge.model.dataset
        self.subgraph_generator = SubgraphGenerator(self.dataset)
        self.num_hops = self.tgnn_bridge.model.num_hops
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger()
        self.sampling_strategy = sampling_strategy
        self.candidates_size = candidates_size
        self.sample_size = sample_size
        self.verbose = verbose

    def _create_sampler(self, subgraph: pd.DataFrame) -> EdgeSampler:
        """
        Create sampler according to 
        @type subgraph: DataFrame The subgraph on which to create the sampler
        """
        if self.sampling_strategy == 'random':
            return RandomEdgeSampler(subgraph)
        elif self.sampling_strategy == 'recent':
            return RecentEdgeSampler(subgraph)
        elif self.sampling_strategy == 'closest':
            return ClosestEdgeSampler(subgraph)
        else:
            raise NotImplementedError(f'No sampler implemented for sampling strategy {self.sampling_strategy}')

    def calculate_original_score(self, explained_event_id: int, min_event_id: int) -> float:
        """
        Calculate the original prediction for the full graph
        @param explained_event_id: Event that is explained
        @param min_event_id: Lowest event id in the candidate events
        @return: The prediction for the full graph without exclusions
        """
        self.tgnn_bridge.initialize(min_event_id, show_progress=self.verbose, memory_label=EXPLAINED_EVENT_MEMORY_LABEL)
        self.tgnn_bridge.initialize(explained_event_id - 1)
        original_prediction, _ = self.tgnn_bridge.predict(explained_event_id, result_as_logit=True)
        original_prediction = original_prediction.detach().cpu().item()
        if self.verbose:
            self.logger.info(f'Original prediction {original_prediction}')
        return original_prediction

    def initialize_explanation(self, explained_event_id: int) -> (float, EdgeSampler):
        """
        Initialize the explanation process
        @param explained_event_id: ID of the event that should be explained
        @return: Original prediction for the event, EdgeSampler for the fixed-size-k-hop-temporal-subgraph around the
        explained event
        """
        subgraph = self.subgraph_generator.get_fixed_size_k_hop_temporal_subgraph(num_hops=self.num_hops,
                                                                                  base_event_id=explained_event_id,
                                                                                  size=self.candidates_size)
        min_event_id = subgraph[COL_ID].min() - 1  # One less since we do not want to simulate the minimal event

        self.tgnn_bridge.set_evaluation_mode(True)
        self.tgnn_bridge.reset_model()
        original_prediction = self.calculate_original_score(explained_event_id, min_event_id)
        return original_prediction, self._create_sampler(subgraph)

    def calculate_subgraph_prediction(self, candidate_events: np.ndarray, cf_example_events: List[int],
                                      explained_event_id: int, candidate_event_id: int,
                                      memory_label: str = EXPLAINED_EVENT_MEMORY_LABEL) -> float:
        """
        Calculate the prediction score for the explained event, when excluding the candidate events
        @param candidate_events: Candidate events
        @param cf_example_events: Events to exclude
        @param explained_event_id: ID of the explained event
        @param candidate_event_id: ID of the currently investigated candidate event
        @param memory_label: Provide name of memory label if it should be different from the default
        @return: Prediction when excluding the candidate events
        """
        self.tgnn_bridge.initialize(np.min(candidate_events) - 1, show_progress=False,
                                    memory_label=memory_label)
        subgraph_prediction, _ = self.tgnn_bridge.predict_from_subgraph(explained_event_id,
                                                                        np.array(cf_example_events +
                                                                                 [candidate_event_id]),
                                                                        result_as_logit=True)
        subgraph_prediction = subgraph_prediction.detach().cpu().item()
        return subgraph_prediction

    def explain(self, explained_event_id: int) -> CounterFactualExample:
        """
        Explain the provided event
        @param explained_event_id: Event id to explain
        @return: The counterfactual explanation
        """
        raise NotImplementedError
