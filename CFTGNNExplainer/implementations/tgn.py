import math
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch

from CFTGNNExplainer.connector import TGNNWrapper
from CFTGNNExplainer.constants import COL_TIMESTAMP, COL_NODE_I, COL_NODE_U
from CFTGNNExplainer.data import BatchData, ContinuousTimeDynamicGraphDataset
from CFTGNNExplainer.utils import ProgressBar, construct_model_path
from TGN.evaluation.evaluation import eval_edge_prediction
from TGN.model.tgn import TGN
from TGN.utils.data_processing import compute_time_statistics, Data
from TGN.utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder


def to_data_object(dataset: ContinuousTimeDynamicGraphDataset, edges_to_drop: np.ndarray = None) -> Data:
    """
    Convert the dataset to a data object that can be used as input for a neighborhood finder
    @param dataset: Dataset to convert to the data object
    @param edges_to_drop: Edges that should be excluded from the data
    @return: Data object of the dataset
    """
    if edges_to_drop is not None:
        edge_mask = ~np.isin(dataset.edge_ids, edges_to_drop)
        return Data(dataset.source_node_ids[edge_mask], dataset.target_node_ids[edge_mask],
                    dataset.timestamps[edge_mask], dataset.edge_ids[edge_mask], dataset.labels[edge_mask])
    return Data(dataset.source_node_ids, dataset.target_node_ids, dataset.timestamps, dataset.edge_ids, dataset.labels)


class TGNWrapper(TGNNWrapper):
    #  Wrapper for 'Temporal Graph Networks' model from https://github.com/twitter-research/tgn

    def __init__(self, model: TGN, dataset: ContinuousTimeDynamicGraphDataset, num_hops: int, model_name: str,
                 device: str = 'cpu', n_neighbors: int = 20, batch_size: int = 32, checkpoint_path: str = None):
        super().__init__(model=model, dataset=dataset, num_hops=num_hops, model_name=model_name, device=device)
        # Set time statistics values
        model.mean_time_shift_src, model.std_time_shift_src, model.mean_time_shift_dst, model.std_time_shift_dst = \
            compute_time_statistics(self.dataset.source_node_ids, self.dataset.target_node_ids, self.dataset.timestamps)

        self.model = model
        self.n_neighbors = n_neighbors
        self.batch_size = batch_size
        if checkpoint_path is not None:
            self.model.load_state_dict(torch.load(checkpoint_path))
        self.model.to(torch.device(device))
        self.node_embedding_dimension = self.model.embedding_module.embedding_dimension
        self.time_embedding_dimension = self.model.time_encoder.dimension
        self.reset_model()
        self.reset_latest_event_id()  # Reset to a clean state

    def initialize(self, event_id: int, show_progress: bool = False, memory_label: str = None):
        if memory_label is not None and memory_label in self.memory_backups_map.keys():
            if show_progress:
                print(f'Restoring memory with label "{memory_label}"')
            memory_backup, backup_event_id = self.memory_backups_map[memory_label]
            if backup_event_id == event_id:
                self.restore_memory(memory_backup, event_id)
                return
            else:  # This should not happen. If this happens causes the model to reprocess everything from the beginning
                self.logger.warning('The provided event ID does not match the event id of the backup. '
                                    'Recreating the state by processing from the beginning.')
                self.reset_model()

        progress_bar = None
        if show_progress:
            progress_bar = ProgressBar(0, prefix='Rolling out events')
        self.rollout_until_event(event_id, progress_bar=progress_bar)
        if progress_bar is not None:
            progress_bar.close()
        if memory_label is not None:
            current_memory = self.get_memory()
            self.memory_backups_map[memory_label] = (current_memory, event_id)
            if show_progress:
                print(f'Backed up memory with label "{memory_label}"')

    def predict(self, event_id: int, result_as_logit: bool = False):
        source_node, target_node, timestamp, edge_id = self.extract_event_information(event_id)
        return self.compute_edge_probabilities(source_nodes=source_node,
                                               target_nodes=target_node,
                                               edge_timestamps=timestamp,
                                               edge_ids=edge_id,
                                               result_as_logit=result_as_logit,
                                               perform_memory_update=False)

    def post_batch_cleanup(self):
        self.model.detach_memory()

    def rollout_until_event(self, event_id: int = None, batch_data: BatchData = None,
                            progress_bar: ProgressBar = None, event_ids_to_rollout: np.ndarray = None) -> None:
        assert event_id is not None or batch_data is not None
        event_id += 1  # One more than the event id, since the event id sets the end index
        if batch_data is None:
            batch_data = self.dataset.get_batch_data(self.latest_event_id, event_id)
        batch_id = 0
        number_of_batches = int(np.ceil(len(batch_data.source_node_ids) / self.batch_size))
        edge_ids_batches = None
        if event_ids_to_rollout is not None:
            if len(event_ids_to_rollout) == 0:
                return
            event_ids_to_rollout = np.sort(event_ids_to_rollout)
            batches_boundaries = np.arange(self.latest_event_id,
                                           event_ids_to_rollout[-1] + self.batch_size, self.batch_size)
            edge_ids_batches = np.split(event_ids_to_rollout, np.searchsorted(event_ids_to_rollout, batches_boundaries))
            edge_ids_batches = [array for array in edge_ids_batches if len(array) > 0]
            number_of_batches = len(edge_ids_batches)
        if progress_bar is not None:
            progress_bar.reset(number_of_batches)
        with torch.no_grad():
            for batch_index in range(number_of_batches):
                if progress_bar is not None:
                    progress_bar.next()
                if edge_ids_batches is not None:
                    # Only process the edge ids in these batches
                    edge_idxs = edge_ids_batches[batch_index]
                    source_nodes, destination_nodes, edge_times, _ = self.extract_event_information(edge_idxs)
                    self.model.compute_temporal_embeddings(source_nodes=source_nodes,
                                                           destination_nodes=destination_nodes,
                                                           edge_times=edge_times,
                                                           edge_idxs=edge_idxs,
                                                           negative_nodes=None)
                else:
                    batch_start = batch_id * self.batch_size
                    batch_end = min((batch_id + 1) * self.batch_size, len(batch_data.source_node_ids))
                    batch_id += 1

                    source_nodes = batch_data.source_node_ids[batch_start:batch_end]
                    destination_nodes = batch_data.target_node_ids[batch_start:batch_end]
                    edge_times = batch_data.timestamps[batch_start:batch_end]
                    edge_idxs = batch_data.edge_ids[batch_start:batch_end]
                    self.model.compute_temporal_embeddings(source_nodes=source_nodes,
                                                           destination_nodes=destination_nodes,
                                                           edge_times=edge_times,
                                                           edge_idxs=edge_idxs,
                                                           negative_nodes=None)
                self.model.memory.detach_memory()

        self.latest_event_id = event_id

    def compute_embeddings(self, source_nodes, target_nodes, edge_times, edge_ids, negative_nodes=None):
        src_node_embedding, target_node_embedding, _ = (self.model.
                                                        compute_temporal_embeddings(source_nodes, target_nodes,
                                                                                    negative_nodes, edge_times,
                                                                                    edge_ids,
                                                                                    n_neighbors=self.n_neighbors,
                                                                                    perform_memory_update=False))
        return src_node_embedding, target_node_embedding

    def encode_timestamps(self, timestamps: np.ndarray):
        timestamps = torch.tensor(timestamps, dtype=torch.float32, device=self.device).reshape((1, -1))
        return self.model.time_encoder(timestamps)

    def compute_edge_probabilities(self, source_nodes: np.ndarray, target_nodes: np.ndarray,
                                   edge_timestamps: np.ndarray, edge_ids: np.ndarray,
                                   negative_nodes: np.ndarray | None = None, result_as_logit: bool = False,
                                   perform_memory_update: bool = True):
        return self.model.compute_edge_probabilities(source_nodes, target_nodes, negative_nodes, edge_timestamps,
                                                     edge_ids, self.n_neighbors, result_as_logit, perform_memory_update)

    def compute_edge_probabilities_for_subgraph(self, event_id, edges_to_drop: np.ndarray,
                                                result_as_logit: bool = False,
                                                event_ids_to_rollout: np.ndarray = None) -> (
    torch.Tensor, torch.Tensor):
        if not self.evaluation_mode:
            self.logger.info('Model not in evaluation mode. Do not use predictions for evaluation purposes!')
        # Insert a new neighborhood finder so that the model does not consider dropped edges
        original_ngh_finder = self.model.neighbor_finder
        self.model.set_neighbor_finder(get_neighbor_finder(to_data_object(self.dataset, edges_to_drop=edges_to_drop),
                                                           uniform=False))
        if event_ids_to_rollout is None:
            event_ids_to_rollout = self.dataset.edge_ids[~np.isin(self.dataset.edge_ids, edges_to_drop)]
            event_ids_to_rollout = event_ids_to_rollout[event_ids_to_rollout > self.latest_event_id]
        event_ids_to_rollout = event_ids_to_rollout[event_ids_to_rollout < event_id]
        # Rollout the events from the subgraph
        self.rollout_until_event(event_id=event_id, event_ids_to_rollout=event_ids_to_rollout)

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
        self.reset_latest_event_id(event_id + 1)

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
            new_node_test_data = self.get_training_data(randomize_features=False, validation_fraction=0.15,
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
        self.model.set_neighbor_finder(full_neighborhood_finder)
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

    def get_training_data(self, randomize_features: bool = False, validation_fraction: float = 0.15,
                          test_fraction: float = 0.15, new_test_nodes_fraction: float = 0.1,
                          different_new_nodes_between_val_and_test: bool = False):
        dataset = self.dataset
        # Function adapted from data_processing.py in https://github.com/twitter-research/tgn
        node_features = dataset.node_features
        if randomize_features:
            node_features = np.random.rand(dataset.node_features.shape[0], node_features.shape[1])

        val_time, test_time = list(np.quantile(dataset.events[COL_TIMESTAMP],
                                               [1 - (validation_fraction + test_fraction), 1 - test_fraction]))

        full_data = Data(dataset.source_node_ids, dataset.target_node_ids, dataset.timestamps, dataset.edge_ids,
                         dataset.labels)

        node_set = set(dataset.source_node_ids) | set(dataset.target_node_ids)
        unique_nodes = len(node_set)

        # Compute nodes which appear at test time
        test_node_set = set(dataset.source_node_ids[dataset.timestamps > val_time]) \
            .union(set(dataset.target_node_ids[dataset.timestamps > val_time]))
        # Sample nodes which we keep as new nodes (to test inductiveness), so than we have to remove all
        # their edges from training
        new_test_node_set = set(random.sample(sorted(test_node_set), int(new_test_nodes_fraction * unique_nodes)))

        # Mask saying for each source and destination whether they are new test nodes
        new_test_source_mask = dataset.events[COL_NODE_I].map(lambda x: x in new_test_node_set).values
        new_test_destination_mask = dataset.events[COL_NODE_U].map(lambda x: x in new_test_node_set).values

        # Mask which is true for edges with both destination and source not being new test nodes (because
        # we want to remove all edges involving any new test node)
        observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)

        # For train, we keep edges happening before the validation time which do not involve any new node
        # used for inductiveness
        train_mask = np.logical_and(dataset.timestamps <= val_time, observed_edges_mask)

        train_data = Data(dataset.source_node_ids[train_mask], dataset.target_node_ids[train_mask],
                          dataset.timestamps[train_mask],
                          dataset.edge_ids[train_mask], dataset.labels[train_mask])

        # define the new nodes sets for testing inductiveness of the model
        train_node_set = set(train_data.sources).union(train_data.destinations)
        assert len(train_node_set & new_test_node_set) == 0
        new_node_set = node_set - train_node_set

        val_mask = np.logical_and(dataset.timestamps <= test_time, dataset.timestamps > val_time)
        test_mask = dataset.timestamps > test_time

        if different_new_nodes_between_val_and_test:
            n_new_nodes = len(new_test_node_set) // 2
            val_new_node_set = set(list(new_test_node_set)[:n_new_nodes])
            test_new_node_set = set(list(new_test_node_set)[n_new_nodes:])

            edge_contains_new_val_node_mask = np.array(
                [(a in val_new_node_set or b in val_new_node_set) for a, b in
                 zip(dataset.source_node_ids, dataset.target_node_ids)])
            edge_contains_new_test_node_mask = np.array(
                [(a in test_new_node_set or b in test_new_node_set) for a, b in
                 zip(dataset.source_node_ids, dataset.target_node_ids)])
            new_node_val_mask = np.logical_and(val_mask, edge_contains_new_val_node_mask)
            new_node_test_mask = np.logical_and(test_mask, edge_contains_new_test_node_mask)
        else:
            edge_contains_new_node_mask = np.array(
                [(a in new_node_set or b in new_node_set) for a, b in
                 zip(dataset.source_node_ids, dataset.target_node_ids)])
            new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
            new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

        # validation and test with all edges
        val_data = Data(dataset.source_node_ids[val_mask], dataset.target_node_ids[val_mask],
                        dataset.timestamps[val_mask],
                        dataset.edge_ids[val_mask], dataset.labels[val_mask])

        test_data = Data(dataset.source_node_ids[test_mask], dataset.target_node_ids[test_mask],
                         dataset.timestamps[test_mask],
                         dataset.edge_ids[test_mask], dataset.labels[test_mask])

        # validation and test with edges that at least has one new node (not in training set)
        new_node_val_data = Data(dataset.source_node_ids[new_node_val_mask], dataset.target_node_ids[new_node_val_mask],
                                 dataset.timestamps[new_node_val_mask],
                                 dataset.edge_ids[new_node_val_mask], dataset.labels[new_node_val_mask])

        new_node_test_data = Data(dataset.source_node_ids[new_node_test_mask],
                                  dataset.target_node_ids[new_node_test_mask],
                                  dataset.timestamps[new_node_test_mask], dataset.edge_ids[new_node_test_mask],
                                  dataset.labels[new_node_test_mask])

        print("The dataset has {} interactions, involving {} different nodes".format(full_data.n_interactions,
                                                                                     full_data.n_unique_nodes))
        print("The training dataset has {} interactions, involving {} different nodes".format(
            train_data.n_interactions, train_data.n_unique_nodes))
        print("The validation dataset has {} interactions, involving {} different nodes".format(
            val_data.n_interactions, val_data.n_unique_nodes))
        print("The test dataset has {} interactions, involving {} different nodes".format(
            test_data.n_interactions, test_data.n_unique_nodes))
        print("The new node validation dataset has {} interactions, involving {} different nodes".format(
            new_node_val_data.n_interactions, new_node_val_data.n_unique_nodes))
        print("The new node test dataset has {} interactions, involving {} different nodes".format(
            new_node_test_data.n_interactions, new_node_test_data.n_unique_nodes))
        print("{} nodes were used for the inductive testing, i.e. are never seen during training".format(
            len(new_test_node_set)))

        return (node_features, dataset.edge_features, full_data, train_data, val_data, test_data, new_node_val_data,
                new_node_test_data)
