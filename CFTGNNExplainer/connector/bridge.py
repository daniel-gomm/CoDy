import numpy as np

from CFTGNNExplainer.connector.tgnnwrapper import TGNNWrapper
from CFTGNNExplainer.utils import ProgressBar


#  TODO: Assess if it actually makes sense to have a bridge between explainer and wrapper or if this just introduces
#  more complexity with no benefit in clarity

class TGNNBridge:

    def __init__(self, model: TGNNWrapper, explanation_candidates_size: int = 30):
        self.model = model
        self.explanation_candidates_size = explanation_candidates_size
        self.memory_backups_map = {}

    def initialize(self, event_id: int, show_progress: bool = False, memory_label: str = None):
        raise NotImplementedError

    def predict(self, event_id: int, result_as_logit: bool = False):
        raise NotImplementedError

    def predict_from_subgraph(self, event_id: int, edges_to_drop: np.ndarray,
                              result_as_logit: bool = False):
        raise NotImplementedError

    def reset_model(self):
        raise NotImplementedError

    def post_batch_cleanup(self):
        pass

    def remove_memory_backup(self, label: str):
        raise NotImplementedError

    def set_evaluation_mode(self, activate_evaluation: bool):
        raise NotImplementedError


class DynamicTGNNBridge(TGNNBridge):

    def __init__(self, model: TGNNWrapper, explanation_candidates_size: int = 30):
        super().__init__(model=model, explanation_candidates_size=explanation_candidates_size)
        self.model.reset_model()
        self.model.reset_latest_event_id()  # Reset to a clean state

    def initialize(self, event_id: int, show_progress: bool = False, memory_label: str = None):
        if memory_label is not None and memory_label in self.memory_backups_map.keys():
            if show_progress:
                print(f'Restoring memory with label "{memory_label}"')
            memory_backup, backup_event_id = self.memory_backups_map[memory_label]
            assert backup_event_id == event_id, 'The provided event id does not match the event id of the backup'
            self.model.restore_memory(memory_backup, event_id)
            return
        progress_bar = None
        if show_progress:
            progress_bar = ProgressBar(0, prefix='Rolling out events')
        self.model.rollout_until_event(event_id, progress_bar=progress_bar)
        if progress_bar is not None:
            progress_bar.close()
        if memory_label is not None:
            current_memory = self.model.get_memory()
            self.memory_backups_map[memory_label] = (current_memory, event_id)
            if show_progress:
                print(f'Backed up memory with label "{memory_label}"')

    def predict(self, event_id: int, result_as_logit: bool = False):
        source_node, target_node, timestamp, edge_id = self.model.extract_event_information(event_id)
        return self.model.compute_edge_probabilities(source_nodes=source_node,
                                                     target_nodes=target_node,
                                                     edge_timestamps=timestamp,
                                                     edge_ids=edge_id,
                                                     result_as_logit=result_as_logit,
                                                     perform_memory_update=False)

    def predict_from_subgraph(self, event_id: int, edges_to_drop: np.ndarray,
                              result_as_logit: bool = False):
        return self.model.compute_edge_probabilities_for_subgraph(event_id, edges_to_drop,
                                                                  result_as_logit=result_as_logit)

    def remove_memory_backup(self, label: str):
        del self.memory_backups_map[label]

    def reset_model(self):
        self.model.reset_model()

    def post_batch_cleanup(self):
        self.model.detach_memory()

    def set_evaluation_mode(self, activate_evaluation: bool):
        if activate_evaluation:
            self.model.activate_evaluation_mode()
        else:
            self.model.activate_train_mode()
