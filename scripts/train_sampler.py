import argparse
import json
import logging

import torch.nn as nn
import torch.optim
import numpy as np

from typing import List, Dict

from CFTGNNExplainer.sampling.sampler import load_prediction_model
from common import (create_dataset_from_args, create_tgn_wrapper_from_args, add_dataset_arguments,
                    add_wrapper_model_arguments, parse_args)

from CFTGNNExplainer.connector.bridge import DynamicTGNNBridge
from CFTGNNExplainer.constants import EXPLAINED_EVENT_MEMORY_LABEL, CUR_IT_MIN_EVENT_MEM_LBL, COL_ID
from CFTGNNExplainer.data.dataset import TrainTestDatasetParameters
from CFTGNNExplainer.explainer.evaluation import EvaluationExplainer
from CFTGNNExplainer.explainer.base import calculate_prediction_delta
from CFTGNNExplainer.sampling.embedding import Embedding, DynamicEmbedding, StaticEmbedding
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


def train_epoch(explainer: EvaluationExplainer, emb: Embedding, model: nn.Module,
                optimizer: torch.optim.Optimizer, epoch_event_ids: List[int], device: str = 'cpu', depth: int = 2):
    optimizer.zero_grad()
    prediction_model.train()
    criterion = torch.nn.MSELoss()
    explainer.tgnn_bridge.reset_model()

    results = []
    losses = []
    last_min_event_id = 0

    progress_bar = ProgressBar(max_item=len(epoch_event_ids), prefix='Training on events')
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

            sample_embeddings = emb.get_embedding(sample, event_id)
            sample_weights = model(sample_embeddings)

            explainer.tgnn_bridge.initialize(min_event_id, show_progress=False,
                                             memory_label=EXPLAINED_EVENT_MEMORY_LABEL)

            sample_true_prediction_deltas = []
            progress_bar.add_inner_progress(len(sample), f'Processing sample for event {event_id}')
            for sample_edge_id in sample:
                subgraph_prediction = explainer.calculate_subgraph_prediction(sample,
                                                                              np.unique(removed_events).tolist(),
                                                                              event_id,
                                                                              candidate_event_id=sample_edge_id,
                                                                              memory_label=CUR_IT_MIN_EVENT_MEM_LBL)
                sample_true_prediction_deltas.append(calculate_prediction_delta(curr_d_prediction,
                                                                                subgraph_prediction))
                progress_bar.inner_next()
            progress_bar.inner_close()

            explainer.tgnn_bridge.remove_memory_backup(CUR_IT_MIN_EVENT_MEM_LBL)

            true_values = np.array(sample_true_prediction_deltas)
            true_values = (torch.tensor(true_values, device=torch.device(device), dtype=torch.float)
                           .reshape(len(true_values), 1))

            loss = criterion(sample_weights, true_values)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.detach().cpu().item())

            saved_results = {'explained_event_id': event_id,
                             'removed_events': removed_events.copy(),
                             'original_prediction': curr_d_prediction,
                             'predictions': sample_weights.detach().cpu().numpy().tolist(),
                             'sample': sample,
                             'true_values': sample_true_prediction_deltas,
                             'embeddings': sample_embeddings.detach().cpu().numpy()}
            results.append(saved_results)

            removed_events.extend(np.random.choice(sample, int(len(sample) / 2), replace=False).tolist())

        progress_bar.update_postfix(f"Avg. loss: {np.mean(losses)}")
        last_min_event_id = min_event_id
        progress_bar.next()
    progress_bar.close()
    return model, losses, results


def train_on_previous_results(model: nn.Module, optimizer: torch.optim.Optimizer, previous_results: List[Dict],
                              device: str = 'cpu', epochs: int = 5):
    model.train()
    criterion = torch.nn.MSELoss()
    losses_list = []
    progress_bar = ProgressBar(epochs, 'Training on previous results: Epoch')
    for e in range(epochs):
        progress_bar.next()
        progress_bar.add_inner_progress(len(previous_results), 'Batch')
        optimizer.zero_grad()
        losses = []
        for result in previous_results:
            progress_bar.inner_next()
            embeddings = np.asarray(result['embeddings'])
            embeddings = torch.tensor(embeddings, dtype=torch.float, device=torch.device(device))
            true_values = torch.tensor(np.array(result['true_values']), dtype=torch.float, device=torch.device(device))

            predictions = model(embeddings)

            loss = criterion(predictions, true_values)
            loss.backward()
            losses.append(loss.detach().cpu().item())

            optimizer.step()
            optimizer.zero_grad()

        progress_bar.inner_close()
        progress_bar.write(f'Finished epoch {e + 1} with mean loss of {np.mean(losses)}')
        losses_list.append(losses)

    progress_bar.close()
    return model, losses_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train sampler')

    add_dataset_arguments(parser)
    add_wrapper_model_arguments(parser)

    parser.add_argument('--dynamic', action='store_true', help='Provide to indicate that dynamic embeddings '
                                                               'should be used')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train the model for.')
    parser.add_argument('--model_save_path', required=True, type=str,
                        help='Path at which to save the model and its checkpoints')
    parser.add_argument('--resume_path', required=False, default=None, type=str,
                        help='Path of a pretrained sampler model on which to resume training.')
    parser.add_argument('--train_examples', type=int, default=100, required=True,
                        help='Number of explained events to use to train the sampler')
    parser.add_argument('--depth', type=int, default=2, required=True,
                        help='Number of sampling steps per explained event')
    parser.add_argument('--sample_size', type=int, default=10,
                        help='Number of samples to draw in each sampling step')
    parser.add_argument('--candidates_size', type=int, default=25,
                        help='Number of candidates from which the samples are selected')
    parser.add_argument('--retraining_epochs', type=int, default=0,
                        help='Number of epochs for training the prediction model on the data from the initial training')

    args = parse_args(parser)

    dataset = create_dataset_from_args(args, TrainTestDatasetParameters(0.2, 0.6, 0.8, args.train_examples, 1000, 500))

    tgn_wrapper = create_tgn_wrapper_from_args(args, dataset)

    eval_explainer = EvaluationExplainer(DynamicTGNNBridge(tgn_wrapper), sampling_strategy='random',
                                         candidates_size=args.candidates_size, sample_size=args.sample_size,
                                         verbose=False)

    if args.dynamic:
        embedding = DynamicEmbedding(dataset, tgn_wrapper, embed_static_node_features=False)
        embedding_type = 'dynamic'
    else:
        embedding_type = 'static'
        embedding = StaticEmbedding(dataset, tgn_wrapper)
    prediction_model = load_prediction_model(embedding.dimension, args.resume_path, tgn_wrapper.device)

    adam_optimizer = torch.optim.Adam(prediction_model.parameters(), lr=1e-4)
    results_data = []
    loss_lists = []
    for epoch in range(args.epochs):
        logger.info(f'Starting epoch {epoch + 1}')
        batch_event_ids = dataset.extract_random_event_ids('train')
        prediction_model, loss_list, intermediate_results = train_epoch(eval_explainer, embedding, prediction_model,
                                                                        adam_optimizer, batch_event_ids,
                                                                        tgn_wrapper.device, depth=args.depth)
        results_data.extend(intermediate_results)
        loss_lists.append(loss_list)
        checkpoint_path = (f'{args.model_save_path}/{tgn_wrapper.name}_{dataset.name}_{embedding_type}'
                           f'_sampler_checkpoint_e{str(epoch + 1)}.pth')
        state_dict = prediction_model.state_dict()
        torch.save(state_dict, checkpoint_path)
        logger.info(f'Epoch {epoch + 1} finished with mean loss {np.mean(loss_list)}')

    model_path = f'{args.model_save_path}/{tgn_wrapper.name}_{dataset.name}_{embedding_type}_sampler.pth'
    state_dict = prediction_model.state_dict()
    torch.save(state_dict, model_path)
    logger.info(f'Training finished. Final model has been saved to {model_path}')
    with open(f'{args.model_save_path}/{tgn_wrapper.name}_intermediate_results.json', 'x+') as write_file:
        json.dump(results_data, write_file, cls=NumpyArrayEncoder)

    with open(f'{args.model_save_path}/{tgn_wrapper.name}_losses.json', 'x+') as write_file:
        json.dump(loss_lists, write_file)

    if args.retraining_epochs > 0:
        prediction_model, all_losses = train_on_previous_results(prediction_model, adam_optimizer, results_data,
                                                                 device=tgn_wrapper.device,
                                                                 epochs=args.retraining_epochs)
        model_path = f'{args.model_save_path}/{tgn_wrapper.name}_{dataset.name}_{embedding_type}_sampler_final.pth'
        torch.save(prediction_model.state_dict(), model_path)
        logger.info(f'Saved final model to {model_path}')
