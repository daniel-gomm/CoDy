from CFTGNNExplainer.baseline.ttgnwrapper import TTGNWrapper
from CFTGNNExplainer.connector.bridge import TGNNBridge


class TTGNBridge(TGNNBridge):

    def __init__(self, model: TTGNWrapper, explanation_candidates_size: int):
        super().__init__(model)
        self.last_predicted_event_id = None
        self.model = model
        self.memory_backups_map = {}
        self.model.reset_model()
        self.model.reset_latest_event_id()
        self.explanation_candidates_size = explanation_candidates_size

    def initialize(self, event_id: int, show_progress: bool = False, memory_label: str = None):
        if show_progress:
            print(f'Initializing model for event {event_id}')

        (self.candidate_events,
         self.unique_edge_ids,
         self.base_events,
         original_score) = self.model.initialize(event_id, self.explanation_candidates_size)
        self.original_score = original_score.detach().cpu().item()
        self.last_predicted_event_id = event_id

    def initialize_static(self, event_id: int):
        self.model.reset_model()
        return self.model.initialize(event_id, self.explanation_candidates_size)

    def predict(self, event_id: int, candidate_event_ids=None, edge_weights=None, edge_id_preserve_list=None):
        source_node, target_node, timestamp, edge_id = self.model.extract_event_information(event_id)
        return self.model.compute_edge_probabilities(source_nodes=source_node,
                                                     target_nodes=target_node,
                                                     edge_timestamps=timestamp,
                                                     edge_ids=edge_id,
                                                     perform_memory_update=False,
                                                     candidate_event_ids=candidate_event_ids,
                                                     candidate_event_weights=edge_weights,
                                                     result_as_logit=True,
                                                     edge_idx_preserve_list=edge_id_preserve_list)

    def get_candidate_events(self, event_id: int):
        assert event_id == self.last_predicted_event_id, (f'Last event predicted {self.last_predicted_event_id} does '
                                                          f'not match with the provided event id {event_id}')
        return self.candidate_events

    def reset_model(self):
        self.model.reset_model()

    def remove_memory_backup(self, label: str):
        pass

    def set_evaluation_mode(self, activate_evaluation: bool):
        if activate_evaluation:
            self.model.activate_evaluation_mode()
        else:
            self.model.activate_train_mode()
