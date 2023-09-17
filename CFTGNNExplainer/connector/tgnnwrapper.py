import torch
import pickle
import time
import math
import logging
import numpy as np
from pathlib import Path
from TGN.model.tgn import TGN
from TGN.utils.utils import get_neighbor_finder, RandEdgeSampler, EarlyStopMonitor
from TGN.utils.data_processing import compute_time_statistics
from TGN.evaluation.evaluation import eval_edge_prediction
from CFTGNNExplainer.data.dataset import ContinuousTimeDynamicGraphDataset, BatchData
from CFTGNNExplainer.utils import ProgressBar, construct_model_path


class TGNNWrapper:

    def __init__(self, model: torch.nn.Module, dataset: ContinuousTimeDynamicGraphDataset, num_hops: int,
                 model_name: str):
        self.num_hops = num_hops
        self.model = model
        self.dataset = dataset
        self.name = model_name
        self.latest_event_id = 0
        self.evaluation_mode = False

    def rollout_until_event(self, event_id: int = None, batch_data: BatchData = None,
                            progress_bar: ProgressBar = None) -> None:
        raise NotImplementedError

    def compute_edge_probabilities(self, source_nodes: np.ndarray, target_nodes: np.ndarray,
                                   edge_timestamps: np.ndarray, edge_ids: np.ndarray,
                                   negative_nodes: np.ndarray | None = None, result_as_logit: bool = False,
                                   perform_memory_update: bool = True):
        raise NotImplementedError

    def compute_edge_probabilities_for_subgraph(self, event_id, subgraph_events: np.ndarray,
                                                edges_to_drop: np.ndarray,
                                                result_as_logit: bool = False) -> (torch.Tensor, torch.Tensor):
        raise NotImplementedError

    def get_memory(self):
        raise NotImplementedError

    def detach_memory(self):
        raise NotImplementedError

    def restore_memory(self, memory_backup, event_id):
        raise NotImplementedError

    def reset_model(self):
        raise NotImplementedError

    def activate_evaluation_mode(self):
        self.model.eval()
        self.evaluation_mode = True

    def activate_train_mode(self):
        self.model.train()
        self.evaluation_mode = False

    def reset_latest_event_id(self, value: int = None):
        if value is not None:
            self.latest_event_id = value
        else:
            self.latest_event_id = 0

    def extract_event_information(self, event_ids: int | np.ndarray):
        edge_mask = np.isin(self.dataset.edge_ids, event_ids)
        source_nodes, target_nodes, timestamps = self.dataset.source_node_ids[edge_mask], \
            self.dataset.target_node_ids[edge_mask], self.dataset.timestamps[edge_mask]
        return source_nodes, target_nodes, timestamps, event_ids


class TGNWrapper(TGNNWrapper):
    #  Wrapper for 'Temporal Graph Networks' model from https://github.com/twitter-research/tgn

    def __init__(self, model: TGN, dataset: ContinuousTimeDynamicGraphDataset, num_hops: int, model_name: str,
                 device: str = 'cpu', n_neighbors: int = 20, batch_size: int = 128, checkpoint_path: str = None):
        super().__init__(model=model, dataset=dataset, num_hops=num_hops, model_name=model_name)
        # Set time statistics values
        model.mean_time_shift_src, model.std_time_shift_src, model.mean_time_shift_dst, model.std_time_shift_dst = \
            compute_time_statistics(self.dataset.source_node_ids, self.dataset.target_node_ids, self.dataset.timestamps)

        self.model = model
        self.n_neighbors = n_neighbors
        self.batch_size = batch_size
        self.device = device
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger()
        if checkpoint_path is not None:
            self.model.load_state_dict(torch.load(checkpoint_path))

    def rollout_until_event(self, event_id: int = None, batch_data: BatchData = None,
                            progress_bar: ProgressBar = None) -> None:
        assert event_id is not None or batch_data is not None
        if batch_data is None:
            batch_data = self.dataset.get_batch_data(self.latest_event_id, event_id)
        batch_id = 0
        number_of_batches = int(np.ceil(len(batch_data.source_node_ids) / self.batch_size))
        if progress_bar is not None:
            progress_bar.reset(number_of_batches)
        with torch.no_grad():
            for _ in range(number_of_batches):
                if progress_bar is not None:
                    progress_bar.next()
                batch_start = batch_id * self.batch_size
                batch_end = min((batch_id + 1) * self.batch_size, len(batch_data.source_node_ids))
                self.model.compute_temporal_embeddings(source_nodes=batch_data.source_node_ids[batch_start:batch_end],
                                                       destination_nodes=batch_data.target_node_ids[batch_start:
                                                                                                    batch_end],
                                                       edge_times=batch_data.timestamps[batch_start:batch_end],
                                                       edge_idxs=batch_data.edge_ids[batch_start:batch_end],
                                                       negative_nodes=None)
                self.model.memory.detach_memory()
                batch_id += 1

        self.latest_event_id = event_id

    def compute_edge_probabilities(self, source_nodes: np.ndarray, target_nodes: np.ndarray,
                                   edge_timestamps: np.ndarray, edge_ids: np.ndarray,
                                   negative_nodes: np.ndarray | None = None, result_as_logit: bool = False,
                                   perform_memory_update: bool = True):
        return self.model.compute_edge_probabilities(source_nodes, target_nodes, negative_nodes, edge_timestamps,
                                                     edge_ids, self.n_neighbors, result_as_logit, perform_memory_update)

    def compute_edge_probabilities_for_subgraph(self, event_id, subgraph_events: np.ndarray,
                                                edges_to_drop: np.ndarray,
                                                result_as_logit: bool = False) -> (torch.Tensor, torch.Tensor):
        if not self.evaluation_mode:
            self.logger.info('Model not in evaluation mode. Do not use predictions for evaluation purposes!')
        edge_ids = self.dataset.edge_ids
        edge_ids = edge_ids[edge_ids < event_id]
        edge_ids = edge_ids[edge_ids > self.latest_event_id]
        edge_ids = edge_ids[~np.isin(edge_ids, edges_to_drop)]
        source_nodes, target_nodes, timestamps, edge_ids = self.extract_event_information(edge_ids)
        # Insert a new neighborhood finder so that the model does not consider dropped edges
        original_ngh_finder = self.model.ngh_finder
        self.model.set_neighbor_finder(get_neighbor_finder(self.dataset.to_data_object(edges_to_drop=edges_to_drop),
                                                           uniform=False))
        # Rollout the events from the subgraph
        self.rollout_until_event(batch_data=BatchData(source_nodes, target_nodes, timestamps, edge_ids))

        source_node, target_node, timestamp, edge_id = self.extract_event_information(event_ids=event_id)
        probabilities = self.compute_edge_probabilities(source_node, target_node, timestamp, edge_id,
                                                        result_as_logit=result_as_logit, perform_memory_update=False)
        # Reinsert the original neighborhood finder so that the model can be used as usual
        self.model.set_neighbor_finder(original_ngh_finder)
        return probabilities

    def get_memory(self):
        return self.model.memory.backup_memory()

    def detach_memory(self):
        self.model.memory.detach_memory()

    def restore_memory(self, memory_backup, event_id):
        self.reset_model()
        self.model.memory.restore_memory(memory_backup)
        self.reset_latest_event_id(event_id)

    def reset_model(self):
        self.reset_latest_event_id()
        self.detach_memory()
        self.model.memory.__init_memory__()

    def train_model(self, epochs: int = 50, learning_rate: float = 0.0001, early_stop_patience: int = 5,
                    checkpoint_path: str = './saved_checkpoints/', model_path: str = './saved_models/',
                    results_path: str = './results/dump.pkl'):
        # Adapted from train_self_supervised from https://github.com/twitter-research/tgn
        Path(results_path.rsplit('/', 1)[0] + '/').mkdir(parents=True, exist_ok=True)
        node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, \
            new_node_test_data = self.dataset.get_training_data(randomize_features=False, validation_fraction=0.15,
                                                                test_fraction=0.15, new_test_nodes_fraction=0.1,
                                                                different_new_nodes_between_val_and_test=False)

        train_neighborhood_finder = get_neighbor_finder(train_data, uniform=False)

        full_neighborhood_finder = get_neighbor_finder(full_data, uniform=False)

        # Initialize negative samplers. Set seeds for validation and testing so negatives are the same
        # across different runs
        # NB: in the inductive setting, negatives are sampled only amongst other new nodes
        train_random_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
        val_random_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
        nn_val_random_sampler = RandEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations, seed=1)
        test_random_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)
        new_nodes_test_random_sampler = RandEdgeSampler(new_node_test_data.sources,
                                                        new_node_test_data.destinations,
                                                        seed=3)

        device = torch.device(self.device)

        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.model = self.model.to(device)

        number_of_instances = len(train_data.sources)
        num_batch = math.ceil(number_of_instances / self.batch_size)

        self.logger.info('num of training instances: {}'.format(number_of_instances))
        self.logger.info('num of batches per epoch: {}'.format(num_batch))

        new_nodes_val_aps = []
        val_aps = []
        epoch_times = []
        total_epoch_times = []
        train_losses = []

        early_stopper = EarlyStopMonitor(max_round=early_stop_patience)

        for epoch in range(epochs):
            start_epoch = time.time()
            # ---Training---

            # Reinitialize memory of the model at the start of each epoch
            self.model.memory.__init_memory__()

            # Train using only training graph
            self.model.set_neighbor_finder(train_neighborhood_finder)
            m_loss = []

            self.logger.info('start {} epoch'.format(epoch))
            epoch_progress = ProgressBar(num_batch, prefix=f'Epoch {epoch}')
            for batch_id in range(0, num_batch):
                epoch_progress.next()
                loss = torch.tensor([0], device=device, dtype=torch.float)
                optimizer.zero_grad()

                start_id = batch_id * self.batch_size
                end_id = min(number_of_instances, start_id + self.batch_size)

                sources_batch, destinations_batch = train_data.sources[start_id:end_id], \
                    train_data.destinations[start_id:end_id]
                edge_ids_batch = train_data.edge_idxs[start_id: end_id]
                timestamps_batch = train_data.timestamps[start_id:end_id]

                size = len(sources_batch)
                _, negatives_batch = train_random_sampler.sample(size)

                with torch.no_grad():
                    positive_label = torch.ones(size, dtype=torch.float, device=device)
                    negative_label = torch.zeros(size, dtype=torch.float, device=device)

                self.model = self.model.train()

                positive_prob, negative_prob = self.model.compute_edge_probabilities(sources_batch, destinations_batch,
                                                                                     negatives_batch, timestamps_batch,
                                                                                     edge_ids_batch, self.n_neighbors)
                loss += criterion(positive_prob.squeeze(), positive_label) + criterion(negative_prob.squeeze(),
                                                                                       negative_label)

                loss.backward()
                optimizer.step()
                m_loss.append(loss.item())

                self.model.memory.detach_memory()

                epoch_time = time.time() - start_epoch
                epoch_times.append(epoch_time)
                epoch_progress.update_postfix(
                    f'Current loss: {np.round(loss.item(), 4)} | Avg. loss: {np.round(np.mean(m_loss), 4)}')

            # ---Validation---
            # Validation uses the full graph
            self.model.set_neighbor_finder(full_neighborhood_finder)

            # Backup memory at the end of training, so later we can restore it and use it for the
            # validation on unseen nodes
            train_memory_backup = self.model.memory.backup_memory()

            val_ap, val_auc, val_acc = eval_edge_prediction(model=self.model,
                                                            negative_edge_sampler=val_random_sampler,
                                                            data=val_data,
                                                            n_neighbors=self.n_neighbors)
            val_memory_backup = self.model.memory.backup_memory()
            # Restore memory we had at the end of training to be used when validating on new nodes.
            # Also, backup memory after validation so it can be used for testing (since test edges are
            # strictly later in time than validation edges)
            self.model.memory.restore_memory(train_memory_backup)

            # Validate on unseen nodes
            new_nodes_val_ap, nn_val_auc, nn_val_acc = eval_edge_prediction(model=self.model,
                                                                            negative_edge_sampler=nn_val_random_sampler,
                                                                            data=new_node_val_data,
                                                                            n_neighbors=self.n_neighbors)

            # Restore memory we had at the end of validation
            self.model.memory.restore_memory(val_memory_backup)

            new_nodes_val_aps.append(new_nodes_val_ap)
            val_aps.append(val_ap)
            train_losses.append(np.mean(m_loss))

            # Save temporary results to disk
            pickle.dump({
                "val_aps": val_aps,
                "new_nodes_val_aps": new_nodes_val_aps,
                "train_losses": train_losses,
                "epoch_times": epoch_times,
                "total_epoch_times": total_epoch_times
            }, open(results_path, "wb"))

            total_epoch_time = time.time() - start_epoch
            total_epoch_times.append(total_epoch_time)

            self.logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
            self.logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
            self.logger.info(
                'val auc: {}, new node val auc: {}'.format(val_auc, nn_val_auc))
            self.logger.info(
                'val ap: {}, new node val ap: {}'.format(val_ap, new_nodes_val_ap))
            self.logger.info(
                'val acc: {}, new node val acc: {}'.format(val_acc, nn_val_acc))

            # Early stopping
            if early_stopper.early_stop_check(val_acc):
                self.logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
                self.logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
                best_model_path = construct_model_path(checkpoint_path, self.name, self.dataset.name,
                                                       early_stopper.best_epoch)
                self.model.load_state_dict(torch.load(best_model_path))
                self.logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
                self.model.eval()
                break
            else:
                torch.save(self.model.state_dict(),
                           construct_model_path(checkpoint_path, self.name, self.dataset.name, str(epoch)))

        # Training has finished, we have loaded the best model, and we want to back up its current
        # memory (which has seen validation edges) so that it can also be used when testing on unseen
        # nodes
        val_memory_backup = self.model.memory.backup_memory()

        # ---Test---
        self.model.embedding_module.neighbor_finder = full_neighborhood_finder
        test_ap, test_auc, test_acc = eval_edge_prediction(model=self.model,
                                                           negative_edge_sampler=test_random_sampler,
                                                           data=test_data,
                                                           n_neighbors=self.n_neighbors)

        self.model.memory.restore_memory(val_memory_backup)

        # Test on unseen nodes
        nn_test_ap, nn_test_auc, nn_test_acc = eval_edge_prediction(model=self.model,
                                                                    negative_edge_sampler=new_nodes_test_random_sampler,
                                                                    data=new_node_test_data,
                                                                    n_neighbors=self.n_neighbors)

        self.logger.info(
            'Test statistics: Old nodes -- auc: {}, ap: {}, acc: {}'.format(test_auc, test_ap, test_acc))
        self.logger.info(
            'Test statistics: New nodes -- auc: {}, ap: {}, acc: {}'.format(nn_test_auc, nn_test_ap, test_acc))
        # Save results for this run
        pickle.dump({
            "val_aps": val_aps,
            "new_nodes_val_aps": new_nodes_val_aps,
            "test_ap": test_ap,
            "new_node_test_ap": nn_test_ap,
            "epoch_times": epoch_times,
            "train_losses": train_losses,
            "total_epoch_times": total_epoch_times
        }, open(results_path, "wb"))

        self.logger.info('Saving TGN model')
        # Restore memory at the end of validation (save a model which is ready for testing)
        self.model.memory.restore_memory(val_memory_backup)
        torch.save(self.model.state_dict(), construct_model_path(model_path, self.name, self.dataset.name))
        self.logger.info('TGN model saved')
