import numpy as np
from CFTGNNExplainer.connector.bridge import TGNNBridge
from CFTGNNExplainer.constants import COL_ID, EXPLAINED_EVENT_MEMORY_LABEL, CURRENT_ITERATION_MIN_EVENT_MEMORY_LABEL
from CFTGNNExplainer.explainer.base import Explainer, CounterFactualExample, calculate_prediction_delta


def is_prediction_most_shifted(original_prediction: float, prediction_to_assess: float, previous_delta: float):
    delta = calculate_prediction_delta(original_prediction, prediction_to_assess)
    if previous_delta < delta:
        return True, delta
    return False, delta


class GreedyCFExplainer(Explainer):

    def __init__(self, tgnn_bridge: TGNNBridge, candidates_size: int = 75, sample_size: int = 10,
                 sampling_strategy: str = 'recent'):
        super().__init__(tgnn_bridge, sampling_strategy)
        self.candidates_size = candidates_size
        self.sample_size = sample_size

    def explain(self, explained_event_id: int, verbose: bool = False) -> CounterFactualExample:
        subgraph = self.subgraph_generator.get_fixed_size_k_hop_temporal_subgraph(num_hops=self.num_hops,
                                                                                  base_event_id=explained_event_id,
                                                                                  size=self.candidates_size)

        min_event_id = subgraph[COL_ID].min() - 1  # One less since we do not want to simulate the minimal event

        self.tgnn_bridge.set_evaluation_mode(True)
        self.tgnn_bridge.reset_model()

        original_prediction = self.calculate_original_score(explained_event_id, min_event_id, verbose)

        remaining_subgraph = subgraph[COL_ID].to_numpy()
        remaining_subgraph = remaining_subgraph[remaining_subgraph != explained_event_id]
        cf_example_prediction = original_prediction  # Initialize to orig prediction as complete subgraph is considered
        cf_example_events = []
        cf_example_importances = []
        achieved_counterfactual_explanation = True

        sampler = self._create_sampler(subgraph)

        largest_prediction_delta = 0
        while cf_example_prediction * original_prediction > 0:
            candidate_events = sampler.sample(explained_event_id, np.array(cf_example_events), size=self.sample_size)
            most_shifted_prediction = cf_example_prediction
            most_shifting_event_id = None
            i = 1
            self.tgnn_bridge.initialize(min_event_id, show_progress=False,
                                        memory_label=EXPLAINED_EVENT_MEMORY_LABEL)
            for candidate_event_id in candidate_events:
                current_subgraph_prediction = self.calculate_subgraph_prediction(candidate_events, cf_example_events,
                                                                                 explained_event_id, candidate_event_id)
                is_most_shifted, delta = is_prediction_most_shifted(original_prediction, current_subgraph_prediction,
                                                                    largest_prediction_delta)
                if verbose:
                    self.logger.info(f'Event {candidate_event_id}, prediction {current_subgraph_prediction}, '
                                     f'delta {delta}')
                if is_most_shifted:
                    most_shifted_prediction = current_subgraph_prediction
                    largest_prediction_delta = delta
                    most_shifting_event_id = candidate_event_id

            self.tgnn_bridge.remove_memory_backup(CURRENT_ITERATION_MIN_EVENT_MEMORY_LABEL)

            if most_shifting_event_id is None:
                # Unable to find a better counterfactual example
                achieved_counterfactual_explanation = False
                break

            cf_example_events.append(most_shifting_event_id)
            cf_example_importances.append(largest_prediction_delta)
            cf_example_prediction = most_shifted_prediction
            remaining_subgraph = remaining_subgraph[remaining_subgraph != most_shifting_event_id]
            if verbose:
                self.logger.info(f'Iteration {i} selected event {most_shifting_event_id} with prediction '
                                 f'{most_shifted_prediction}. '
                                 f'CF-example events: {cf_example_events}')
            i += 1

        self.tgnn_bridge.remove_memory_backup(EXPLAINED_EVENT_MEMORY_LABEL)
        self.tgnn_bridge.reset_model()
        return CounterFactualExample(original_prediction=original_prediction,
                                     counterfactual_prediction=cf_example_prediction,
                                     achieves_counterfactual_explanation=achieved_counterfactual_explanation,
                                     event_ids=np.array(cf_example_events),
                                     event_importances=np.array(cf_example_importances))
