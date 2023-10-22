import argparse
import json
import logging
import random

import torch.nn as nn
import torch.optim
import numpy as np

from typing import List, Dict

from sklearn.metrics import ndcg_score

from CFTGNNExplainer.sampler import create_embedding_model
from common import (create_dataset_from_args, create_tgn_wrapper_from_args, add_dataset_arguments,
                    add_wrapper_model_arguments, parse_args)

from CFTGNNExplainer.implementations.tgn import TGNBridge
from CFTGNNExplainer.constants import EXPLAINED_EVENT_MEMORY_LABEL, CUR_IT_MIN_EVENT_MEM_LBL, COL_ID
from CFTGNNExplainer.data import TrainTestDatasetParameters
from CFTGNNExplainer.explainer.evaluation import EvaluationExplainer
from CFTGNNExplainer.explainer.base import calculate_prediction_delta
from CFTGNNExplainer.embedding import Embedding, DynamicEmbedding, StaticEmbedding
from CFTGNNExplainer.utils import ProgressBar

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyArrayEncoder, self).default(obj)


def extract_training_data(explainer: EvaluationExplainer, emb: Embedding, epoch_event_ids: List[int], depth: int = 2):
    res = []
    last_min_event_id = 0
    progress_bar = ProgressBar(max_item=len(epoch_event_ids), prefix='Collecting data for training')
    for index, event_id in enumerate(sorted(epoch_event_ids)):
        sampler = explainer.initialize_explanation_evaluation(event_id, original_prediction=0.0)
        if len(sampler.subgraph) == 0:
            continue

        min_event_id = sampler.subgraph[COL_ID].min() - 1

        removed_events = []

        for d in range(depth):
            sample = sampler.sample(event_id, np.unique(np.array(removed_events)), explainer.sample_size)
            explainer.tgnn_bridge.reset_model()
            if len(removed_events) == 0:
                if 0 < last_min_event_id <= min_event_id:
                    explainer.tgnn_bridge.initialize(last_min_event_id, show_progress=False,
                                                     memory_label=EXPLAINED_EVENT_MEMORY_LABEL)
                explainer.tgnn_bridge.remove_memory_backup(EXPLAINED_EVENT_MEMORY_LABEL)
            explainer.tgnn_bridge.initialize(min_event_id, show_progress=False,
                                             memory_label=EXPLAINED_EVENT_MEMORY_LABEL)

            curr_d_prediction = explainer.calculate_subgraph_prediction(sample, np.unique(removed_events).tolist(),
                                                                        event_id, candidate_event_id=event_id + 1,
                                                                        memory_label=CUR_IT_MIN_EVENT_MEM_LBL)

            sample_embeddings, explained_edge_embedding = emb.get_embeddings(sample, event_id)

            explainer.tgnn_bridge.initialize(min_event_id, show_progress=False,
                                             memory_label=EXPLAINED_EVENT_MEMORY_LABEL)

            sample_true_prediction_deltas = []
            sample_true_predictions = []
            progress_bar.add_inner_progress(len(sample), f'Processing sample for event {event_id}')
            for sample_edge_id in sample:
                subgraph_prediction = explainer.calculate_subgraph_prediction(sample,
                                                                              np.unique(removed_events).tolist(),
                                                                              event_id,
                                                                              candidate_event_id=sample_edge_id,
                                                                              memory_label=CUR_IT_MIN_EVENT_MEM_LBL)
                sample_true_prediction_deltas.append(calculate_prediction_delta(curr_d_prediction,
                                                                                subgraph_prediction))
                sample_true_predictions.append(subgraph_prediction)
                progress_bar.inner_next()
            progress_bar.inner_close()

            explainer.tgnn_bridge.remove_memory_backup(CUR_IT_MIN_EVENT_MEM_LBL)

            saved_results = {'explained_event_id': event_id,
                             'removed_events': removed_events.copy(),
                             'original_prediction': curr_d_prediction,
                             'exclusion_prediction': sample_true_predictions,
                             'sample': sample,
                             'true_values': sample_true_prediction_deltas,
                             'embeddings': sample_embeddings.detach().cpu().numpy(),
                             'explained_edge_embedding': explained_edge_embedding.detach().cpu().numpy()}
            res.append(saved_results)

            removed_events.extend(np.random.choice(sample, int(len(sample) / 2), replace=False).tolist())
        last_min_event_id = min_event_id
        progress_bar.next()
    progress_bar.close()
    return res


def calculate_gain(true_values: torch.Tensor) -> np.ndarray:
    true_vals = true_values.detach().cpu().numpy()
    sorted_indices = np.argsort(true_vals[true_vals > 0])
    gains = np.zeros_like(true_vals)
    gains[true_vals > 0] = np.arange(len(sorted_indices))[sorted_indices] + 1
    return gains


def calculate_embeddings(embedding_model: nn.Module, result: Dict, device: str):
    explained_edge_embedding = result['explained_edge_embedding']
    excluded_edges_embeddings = result['embeddings']

    explained_edge_embedding = torch.tensor(explained_edge_embedding, dtype=torch.float, device=torch.device(device))
    excluded_edges_embeddings = torch.tensor(excluded_edges_embeddings, dtype=torch.float, device=torch.device(device))

    explained_edge_embeddings = torch.tile(explained_edge_embedding, (len(excluded_edges_embeddings), 1))

    return embedding_model(explained_edge_embeddings), embedding_model(excluded_edges_embeddings)


def calculate_ndcg_from_embeddings(explained_edge_embeddings, excluded_edges_embeddings, true_values,
                                   original_prediction):
    predictions = torch.nn.functional.cosine_similarity(explained_edge_embeddings, excluded_edges_embeddings)
    predictions = predictions.reshape(1, len(explained_edge_embeddings))
    if len(explained_edge_embeddings) <= 1:
        return None
    if original_prediction < 0:
        predictions = 1 - predictions
        true_values = -true_values
    gains = calculate_gain(true_values)
    return ndcg_score(gains.reshape(1, len(gains)), predictions.detach().cpu().numpy())


class EarlyStoppingMonitor:

    def __init__(self, wait_epochs: int = 5, greater_better: bool = False):
        self.best_model_state_dict = None
        self.best_model_epoch = None
        self.wait_epochs = wait_epochs
        self.waited_epochs = 0
        if greater_better:  # If greater is better we multiply by -1 to find best epoch
            self.factor = -1
        else:
            self.factor = 1
        self.best_validation_loss = 100000000 * self.factor

    def check_new_epoch(self, model: nn.Module, validation_loss: float, epoch: int) -> bool:

        if validation_loss * self.factor < self.best_validation_loss * self.factor:
            self.best_validation_loss = validation_loss
            self.best_model_epoch = epoch
            self.waited_epochs = 0
            self.best_model_state_dict = model.state_dict()
            return True
        elif self.waited_epochs < self.wait_epochs:
            self.waited_epochs += 1
            return True
        return False

    def load_best_model(self, model: nn.Module):
        model.load_state_dict(self.best_model_state_dict)


def train_embedding_model(embedding_model: nn.Module, previous_results: List[Dict], device: str = 'cpu',
                          epochs: int = 500, verbose: bool = False, early_stopping_threshold: int = 50,
                          early_stopping_greater_better: bool = False):
    embedding_model.train()
    early_stopper = EarlyStoppingMonitor(early_stopping_threshold, greater_better=early_stopping_greater_better)
    random.shuffle(previous_results)
    num_prev_results = len(previous_results)
    num_train = int(0.7 * num_prev_results)
    num_eval = int(0.85 * num_prev_results)
    train_examples = previous_results[:num_train]
    validation_examples = previous_results[num_train:num_eval]
    test_examples = previous_results[num_eval:]
    criterion = torch.nn.CosineEmbeddingLoss()
    optimizer = torch.optim.Adam(embedding_model.parameters(), lr=1e-4)
    losses_list = []
    validation_losses_list = []
    progress_bar = ProgressBar(epochs, 'Training on previous results: Epoch')
    for e in range(epochs):
        embedding_model.train()
        progress_bar.next()
        optimizer.zero_grad()
        losses = []
        train_ndcg_scores = []
        for result in train_examples:
            explained_edge_embeddings, excluded_edges_embeddings = calculate_embeddings(embedding_model, result, device)
            true_values = np.array(result['true_values'])
            true_values = torch.tensor(true_values, dtype=torch.float, device=torch.device(device), requires_grad=True)
            labels = true_values.sign()

            loss = criterion(explained_edge_embeddings, excluded_edges_embeddings, labels)
            loss.backward()
            losses.append(loss.detach().cpu().item())
            optimizer.step()
            optimizer.zero_grad()

            ndcg = calculate_ndcg_from_embeddings(explained_edge_embeddings, excluded_edges_embeddings, true_values,
                                                  result['original_prediction'])
            if ndcg is not None:
                train_ndcg_scores.append(ndcg)

        if verbose:
            print(
                f'Finished epoch {e + 1} with mean loss of {np.mean(losses)} and mean NDCG score of '
                f'{np.mean(train_ndcg_scores)}')
        progress_bar.update_postfix(f'Avg. Loss: {np.mean(losses)}')
        losses_list.append(losses)

        validation_losses = []
        validation_ndcg_scores = []
        for result in validation_examples:
            embedding_model.eval()
            explained_edge_embeddings, excluded_edges_embeddings = calculate_embeddings(embedding_model, result, device)
            true_values_array = np.array(result['true_values'])
            true_values = torch.tensor(true_values_array, dtype=torch.float, device=torch.device(device),
                                       requires_grad=True)
            labels = true_values.sign()

            validation_losses.append(criterion(explained_edge_embeddings, excluded_edges_embeddings, labels).detach()
                                     .cpu().item())

            ndcg = calculate_ndcg_from_embeddings(explained_edge_embeddings, excluded_edges_embeddings, true_values,
                                                  result['original_prediction'])
            if ndcg is not None:
                validation_ndcg_scores.append(ndcg)
        if verbose:
            print(
                f'Epoch {e} finished with validation NDCG score of {np.mean(validation_ndcg_scores)} and validation '
                f'loss of {np.mean(validation_losses)}')
        validation_losses_list.append(validation_losses)
        if not early_stopper.check_new_epoch(embedding_model, np.mean(validation_losses), epoch=e):
            break

    early_stopper.load_best_model(embedding_model)
    print(
        f'Loaded best model from epoch {early_stopper.best_model_epoch} with validation loss of '
        f'{early_stopper.best_validation_loss}')

    test_losses = []
    test_ndcg_scores = []
    for result in test_examples:
        embedding_model.eval()
        explained_edge_embeddings, excluded_edges_embeddings = calculate_embeddings(embedding_model, result, device)
        true_values_array = np.array(result['true_values'])
        true_values = torch.tensor(true_values_array, dtype=torch.float, device=torch.device(device),
                                   requires_grad=True)
        labels = true_values.sign()

        test_losses.append(criterion(explained_edge_embeddings, excluded_edges_embeddings, labels).detach().cpu()
                           .item())

        ndcg = calculate_ndcg_from_embeddings(explained_edge_embeddings, excluded_edges_embeddings, true_values,
                                              result['original_prediction'])
        if ndcg is not None:
            test_ndcg_scores.append(ndcg)
    print(f'Training finished with test loss of {np.mean(test_losses)} and NDCG of {np.mean(test_ndcg_scores)}')

    progress_bar.close()
    return embedding_model, losses_list, validation_losses_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train sampler')

    add_dataset_arguments(parser)
    add_wrapper_model_arguments(parser)

    parser.add_argument('--dynamic', action='store_true', help='Provide to indicate that dynamic embeddings '
                                                               'should be used')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train the model for.')
    parser.add_argument('--model_save_path', required=True, type=str,
                        help='Path at which to save the model and its checkpoints')
    parser.add_argument('--train_examples', type=int, default=500, required=True,
                        help='Number of explained events to use to train the sampler')
    parser.add_argument('--depth', type=int, default=2, required=True,
                        help='Number of sampling steps per explained event')
    parser.add_argument('--sample_size', type=int, default=20,
                        help='Number of samples to draw in each sampling step')
    parser.add_argument('--candidates_size', type=int, default=75,
                        help='Number of candidates from which the samples are selected')

    args = parse_args(parser)

    dataset = create_dataset_from_args(args, TrainTestDatasetParameters(0.2, 0.6, 0.8, args.train_examples, 1000, 500))

    tgn_wrapper = create_tgn_wrapper_from_args(args, dataset)

    eval_explainer = EvaluationExplainer(TGNBridge(tgn_wrapper), sampling_strategy='random',
                                         candidates_size=args.candidates_size, sample_size=args.sample_size,
                                         verbose=False)

    if args.dynamic:
        embedding = DynamicEmbedding(dataset, tgn_wrapper, embed_static_node_features=False)
        embedding_type = 'dynamic'
    else:
        embedding_type = 'static'
        embedding = StaticEmbedding(dataset, tgn_wrapper)

    results_data = []
    loss_lists = []
    results = extract_training_data(eval_explainer, embedding, dataset.extract_random_event_ids('train'),
                                    depth=args.depth)

    with open(f'{args.model_save_path}/{tgn_wrapper.name}_intermediate_results.json', 'x+') as write_file:
        json.dump(results, write_file, cls=NumpyArrayEncoder)
    logger.info(f'Saved {len(results)} training examples to '
                f'{args.model_save_path}/{tgn_wrapper.name}_{embedding_type}_intermediate_results.json')

    logger.info('Starting training of sampler model...')
    emb_model = create_embedding_model(emb=embedding, device=tgn_wrapper.device)
    emb_model, train_losses, val_losses = train_embedding_model(emb_model, results, device=tgn_wrapper.device,
                                                                epochs=args.epochs, early_stopping_threshold=50,
                                                                early_stopping_greater_better=False)
    state_dict = emb_model.state_dict()
    torch.save(state_dict, f'{args.model_save_path}/{tgn_wrapper.name}_{embedding_type}_sampler.pth')
