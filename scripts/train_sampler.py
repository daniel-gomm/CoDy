import argparse
import logging

import torch.nn as nn
import torch.optim
import numpy as np

from typing import List

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


def train_epoch(explainer: EvaluationExplainer, emb: Embedding, model: nn.Module,
                optimizer: torch.optim.Optimizer, epoch_event_ids: List[int], device: str = 'cpu', depth: int = 2):
    optimizer.zero_grad()
    criterion = torch.nn.MSELoss()
    explainer.tgnn_bridge.reset_model()

    losses = []
    last_min_event_id = 0

    progress_bar = ProgressBar(max_item=len(epoch_event_ids))
    for index, event_id in enumerate(sorted(epoch_event_ids)):
        sampler = explainer.initialize_explanation_evaluation(event_id)

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
            for sample_edge_id in sample:
                subgraph_prediction = explainer.calculate_subgraph_prediction(sample,
                                                                              np.unique(removed_events).tolist(),
                                                                              event_id,
                                                                              candidate_event_id=sample_edge_id,
                                                                              memory_label=CUR_IT_MIN_EVENT_MEM_LBL)
                sample_true_prediction_deltas.append(calculate_prediction_delta(curr_d_prediction,
                                                                                subgraph_prediction))

            explainer.tgnn_bridge.remove_memory_backup(CUR_IT_MIN_EVENT_MEM_LBL)

            true_values = np.array(sample_true_prediction_deltas)
            true_values = (torch.tensor(true_values, device=torch.device(device), dtype=torch.float)
                           .reshape(len(true_values), 1))

            loss = criterion(sample_weights, true_values)
            loss.backward()
            losses.append(loss.detach().cpu().item())

            removed_events.extend(np.random.choice(sample, int(len(sample) / 2), replace=False).tolist())

        progress_bar.update_postfix(f"Avg. loss: {np.mean(losses)}")
        last_min_event_id = min_event_id
        progress_bar.next()
    return model, losses


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

    args = parse_args(parser)

    dataset = create_dataset_from_args(args, TrainTestDatasetParameters(0.2, 0.6, 0.8, 100, 1000, 500))

    tgn_wrapper = create_tgn_wrapper_from_args(args, dataset)

    eval_explainer = EvaluationExplainer(DynamicTGNNBridge(tgn_wrapper), sampling_strategy='random', candidates_size=50,
                                         sample_size=25,
                                         verbose=False)

    if args.dynamic:
        embedding = DynamicEmbedding(dataset, tgn_wrapper, embed_static_node_features=True)
        embedding_type = 'dynamic'
    else:
        embedding_type = 'static'
        embedding = StaticEmbedding(dataset, tgn_wrapper)

    prediction_model = nn.Sequential(
        nn.Linear(embedding.dimension, 128),
        nn.ReLU(),
        nn.Linear(128, 1)
    )
    if args.resume_path is not None:
        prediction_model.load_state_dict(torch.load(args.resume_path))

    prediction_model.to(torch.device(tgn_wrapper.device))

    adam_optimizer = torch.optim.Adam(prediction_model.parameters(), lr=1e-4)

    for epoch in range(args.epochs):
        logger.info(f'Starting epoch {epoch + 1}')
        batch_event_ids = dataset.extract_random_event_ids('train')
        prediction_model, loss_list = train_epoch(eval_explainer, embedding, prediction_model, adam_optimizer,
                                                  batch_event_ids, tgn_wrapper.device, depth=2)

        checkpoint_path = (f'{args.model_save_path}/{tgn_wrapper.name}_{dataset.name}_{embedding_type}'
                           f'_sampler_checkpoint_e{str(epoch + 1)}.pth')
        state_dict = prediction_model.state_dict()
        torch.save(state_dict, checkpoint_path)
        logger.info(f'Epoch {epoch + 1} finished with mean loss {np.mean(loss_list)}')

    model_path = f'{args.model_save_path}/{tgn_wrapper.name}_{dataset.name}_{embedding_type}_sampler.pth'
    state_dict = prediction_model.state_dict()
    torch.save(state_dict, model_path)
    logger.info(f'Training finished. Final model has been saved to {model_path}')
