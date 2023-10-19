import logging

import numpy as np

from CFTGNNExplainer.connector.tgnnwrapper import TGNNWrapper


class TGNNBridge:

    def __init__(self, model: TGNNWrapper):
        self.model = model
        self.memory_backups_map = {}
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('TGNNBridge')

    def initialize(self, event_id: int, show_progress: bool = False, memory_label: str = None):
        raise NotImplementedError

    def predict(self, event_id: int, result_as_logit: bool = False):
        raise NotImplementedError

    def predict_from_subgraph(self, event_id: int, edges_to_drop: np.ndarray,
                              result_as_logit: bool = False):
        return self.model.compute_edge_probabilities_for_subgraph(event_id, edges_to_drop,
                                                                  result_as_logit=result_as_logit)

    def reset_model(self):
        self.model.reset_model()

    def post_batch_cleanup(self):
        pass

    def remove_memory_backup(self, label: str):
        if label in self.memory_backups_map:
            del self.memory_backups_map[label]

    def set_evaluation_mode(self, activate_evaluation: bool):
        if activate_evaluation:
            self.model.activate_evaluation_mode()
        else:
            self.model.activate_train_mode()
