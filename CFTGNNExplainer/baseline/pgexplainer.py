from dataclasses import dataclass
from typing import Dict

import numpy as np
import time

from CFTGNNExplainer.sampling.embedding import Embedding
from CFTGNNExplainer.baseline.ttgnbridge import TTGNBridge
from CFTGNNExplainer.explainer.base import Explainer

import torch.nn as nn
import torch.optim

from CFTGNNExplainer.utils import ProgressBar


@dataclass
class FactualExplanation:
    explained_event_id: int
    event_ids: np.ndarray
    event_importances: np.ndarray
    timings: Dict
    statistics: Dict

    def get_absolute_importances(self) -> np.ndarray:
        return np.array([importance - np.sum(self.event_importances[index - 1:index]) for index, importance in
                         enumerate(self.event_importances)])

    def get_relative_importances(self) -> np.ndarray:
        if len(self.event_importances) == 0:
            return np.ndarray([])
        return self.get_absolute_importances() / self.event_importances[-1]

    def to_dict(self) -> Dict:
        results = {
            'explained_event_id': self.explained_event_id,
            'event_ids': self.event_ids.tolist(),
            'results': self.event_importances.tolist()
        }
        results.update(self.statistics)
        results.update(self.timings)
        return results


class TPGExplainer(Explainer):

    def __init__(self, tgnn_bridge: TTGNBridge, embedding: Embedding, device: str = 'cpu', hidden_dimension: int = 128):
        super().__init__(tgnn_bridge)
        self.tgnn_bridge = tgnn_bridge
        self.device = device
        self.embedding = embedding
        self.hidden_dimension = hidden_dimension
        self.explainer = self._create_explainer()
        self.tgnn_bridge.set_evaluation_mode(True)

    def _create_explainer(self) -> nn.Module:
        embedding_dimension = self.embedding.dimension
        explainer_model = nn.Sequential(
            nn.Linear(embedding_dimension, self.hidden_dimension),
            nn.ReLU(),
            nn.Linear(self.hidden_dimension, 1)
        )
        explainer_model = explainer_model.to(self.device)
        return explainer_model

    def explain(self, explained_event_id: int) -> FactualExplanation:
        start_time = time.time_ns()
        self.tgnn_bridge.reset_model()
        self.explainer.eval()
        init_end_time = time.time_ns()
        with torch.no_grad():
            candidate_events = self.tgnn_bridge.get_candidate_events(explained_event_id)
            if len(candidate_events) == 0:
                raise RuntimeError(f'No candidates found to explain event {explained_event_id}')
            edge_weights = self.get_event_scores(explained_event_id, candidate_events)
            edge_weights = edge_weights.cpu().detach().numpy().flatten()
            sorted_indices = np.argsort(edge_weights)[::-1]  # declining
            edge_weights = edge_weights[sorted_indices]
            candidate_events = np.array(candidate_events)[sorted_indices]
        end_time = time.time_ns()
        timings = {
            'oracle_call_duration': 0,
            'explanation_duration': end_time - init_end_time,
            'init_duration': init_end_time - start_time,
            'total_duration': end_time - start_time
        }
        statistics = {
            'oracle_calls': 0,
            'candidate_size': len(candidate_events),
            'candidates': candidate_events
        }
        return FactualExplanation(explained_event_id, candidate_events, edge_weights, timings, statistics)

    @staticmethod
    def _loss(masked_probability, original_probability):
        if original_probability > 0:
            error_loss = (masked_probability - original_probability) * -1
        else:
            error_loss = (original_probability - masked_probability) * -1

        return error_loss

    def _save_explainer(self, path: str):
        state_dict = self.explainer.state_dict()
        torch.save(state_dict, path)

    def get_event_scores(self, explained_event_id, candidate_event_ids):
        self.tgnn_bridge.initialize(explained_event_id)
        edge_embeddings = self.embedding.get_embedding(candidate_event_ids, explained_event_id)
        return self.explainer(edge_embeddings)

    def train(self, epochs: int, learning_rate: float, batch_size: int, model_name: str, save_directory: str,
              train_event_ids: [int] = None):
        self.explainer.train()
        optimizer = torch.optim.Adam(self.explainer.parameters(), lr=learning_rate)

        generate_event_ids = (train_event_ids is None)

        for epoch in range(epochs):
            if generate_event_ids:
                train_event_ids = self.tgnn_bridge.model.dataset.extract_random_event_ids('train')

            self.logger.info(f'Starting training epoch {epoch}')
            optimizer.zero_grad()
            loss = torch.tensor([0], dtype=torch.float32, device=self.device)
            loss_list = []
            counter = 0
            skipped_events = 0

            self.tgnn_bridge.reset_model()

            progress_bar = ProgressBar(max_item=len(train_event_ids), prefix=f'Epoch {epoch}: Explaining events')
            for index, event_id in enumerate(sorted(train_event_ids)):
                progress_bar.next()
                self.tgnn_bridge.initialize(event_id)
                candidate_events = self.tgnn_bridge.get_candidate_events(event_id)
                if len(candidate_events) == 0:
                    skipped_events += 1
                    continue
                edge_weights = self.get_event_scores(event_id, candidate_events)

                prob_original_pos, prob_original_neg = self.tgnn_bridge.predict(event_id)
                prob_masked_pos, prob_masked_neg = self.tgnn_bridge.predict(event_id, candidate_events, edge_weights)

                event_loss = self._loss(prob_masked_pos, prob_original_pos)
                loss += event_loss.flatten()
                loss_list.append(event_loss.flatten().clone().cpu().detach().item())

                counter += 1
                if counter % batch_size == 0 or index + 1 == len(train_event_ids):
                    if counter % batch_size == 0:
                        loss = loss / batch_size
                    else:
                        loss = loss / (counter % batch_size)
                    loss.backward()
                    optimizer.step()
                    progress_bar.update_postfix(f"Cur. loss: {loss.item()}")
                    loss = torch.tensor([0], dtype=torch.float32, device=self.device)
                    self.tgnn_bridge.post_batch_cleanup()
                    optimizer.zero_grad()
                    counter = 0

            progress_bar.close()

            self.logger.info(
                f'Finished epoch {epoch} with mean loss of {np.mean(loss_list)}, median loss of {np.median(loss_list)},'
                f' loss variance {np.var(loss_list)} and {skipped_events} skipped events')

            checkpoint_path = f'{save_directory}/{model_name}_checkpt_e{epoch}.pth'
            self._save_explainer(checkpoint_path)
            self.logger.info(f"Saved checkpoint to {checkpoint_path}")

        model_path = f'{save_directory}/{model_name}_final.pth'
        self._save_explainer(model_path)
        self.logger.info(f'Finished training, saved explainer checkpoint at {model_path}')
