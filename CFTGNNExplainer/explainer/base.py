import logging
import numpy as np
from dataclasses import dataclass

from CFTGNNExplainer.connector.bridge import TGNNBridge


@dataclass
class CounterFactualExample:
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
        return np.array([importance - np.sum(self.event_importances[index-1:index]) for index, importance in
                         enumerate(self.event_importances)])

    def get_relative_importances(self) -> np.ndarray:
        return self.get_absolute_importances() / self.event_importances[-1]


class Explainer:

    def __init__(self, tgnn_bridge: TGNNBridge):
        self.tgnn_bridge = tgnn_bridge
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger()

    def explain(self, explained_event_id: int) -> CounterFactualExample:
        raise NotImplementedError
