import glob
import os
from enum import Enum
from typing import Optional

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import random_split
from torch_geometric.data import Data


class Activation(Enum):
    """Activation enum so we can experiment.
    """
    ReLU = 'relu'
    PReLU = 'prelu'
    ELU = 'elu'
    SELU = 'selu'
    Tanh = 'tanh'


class Experiments:
    """
    All trainer shared method.  Update metrics, save load checkpoint etc.
    """

    def __init__(
            self, epochs,
            batch_size,
            midi_dataset,
            model_type: Optional[str] = "",
            lr: Optional[float] = 0.01,
            activation: Optional[Activation] = Activation.ReLU,
            train_update_rate: Optional[int] = 1,
            test_update_freq: Optional[int] = 10,
            eval_update_freq: Optional[int] = 20,
            save_freq: Optional[int] = 20):
        """
        """
        if epochs is None:
            raise ValueError("epochs cannot be None")
        if batch_size is None:
            raise ValueError("batch_size cannot be None")
        if midi_dataset is None:
            raise ValueError("midi_dataset cannot be None.")
        if not hasattr(midi_dataset, '__getitem__') or not hasattr(midi_dataset, '__len__'):
            raise ValueError("midi_dataset must have __getitem__ and __len__ methods.")

        assert isinstance(batch_size, int) and batch_size > 0, "batch_size must be a positive integer"
        assert isinstance(epochs, int) and epochs > 0, "epochs must be a positive integer"

        self._dataset = midi_dataset
        self.num_classes = self._dataset.total_num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        assert isinstance(self.device, torch.device), "device must be a torch.device object"

        self.datasize = 0
        self.test_size = 0
        self._num_workers = 0
        self._batch_size = batch_size
        self._epochs = epochs
        self.current_epoch = 0
        self.start_epoch = 0
        self.model_type = model_type

        self.save_freq = save_freq
        self.train_update_rate = train_update_rate
        self.test_update_freq = test_update_freq
        self.eval_update_freq = eval_update_freq

        #
        self.data_loader = None
        self.train_ration = 0.8
        self.test_dataset = None
        self.train_dataset = None
        self.model = None
        self.optimizer = None

        self.val_precisions = []
        self.val_f1s = []
        self.val_accs = []
        self.val_recalls = []
        self.test_recalls = []
        self.test_precisions = []
        self.test_accs = []
        self.test_f1s = []
        self.train_precisions = []
        self.train_recalls = []
        self.train_f1s = []
        self.train_accs = []
        self.train_losses = []
        self.test_aps = []
        self.test_aucs = []

        self.metrics_dir = "metric"
        self.save_metrics = True
        self.metrics_rate = 100

    def plot_metrics(self, output_dir="metric", model_type=""):
        """Plot metrics
        :return:
        """
        if self.model_type and len(self.model_type) > 0:
            model_type = self.model_type

        if not (isinstance(self.train_losses, (list, np.ndarray))
                and isinstance(self.train_f1s, (list, np.ndarray))
                and isinstance(self.train_accs, (list, np.ndarray))
                and isinstance(self.train_precisions, (list, np.ndarray))
                and isinstance(self.train_recalls, (list, np.ndarray))):
            print("Method accept list or nd.ndarray")
            return

        epochs = range(1, len(self.train_losses) + 1)
        plt.figure(figsize=(15, 10))
        if self.train_losses:
            plt.plot(epochs, self.train_losses, label="train_loss")
        if self.train_f1s:
            plt.plot(epochs, self.train_f1s, label="train_f1")
        if self.train_accs:
            plt.plot(epochs, self.train_accs, label="train_acc")
        if self.train_precisions:
            plt.plot(epochs, self.train_precisions, label="train_precision")
        if self.train_recalls:
            plt.plot(epochs, self.train_recalls, label="train_recall")

        if self.val_accs and self.val_f1s and self.val_precisions and self.val_recalls:
            val_epochs = range(self.eval_update_freq,
                               len(self.val_accs) * self.eval_update_freq + 1,
                               self.eval_update_freq)
            if self.val_accs:
                plt.plot(val_epochs, self.val_accs, label="val_acc")
            if self.val_f1s:
                plt.plot(val_epochs, self.val_f1s, label="val_f1")
            if self.val_precisions:
                plt.plot(val_epochs, self.val_precisions, label="val_precision")
            if self.val_recalls:
                plt.plot(val_epochs, self.val_recalls, label="val_recall")
            if self.test_aucs:
                plt.plot(val_epochs, self.test_aucs, label="test_auc")
            if self.test_aps:
                plt.plot(val_epochs, self.test_aps, label="test_ap")

        if self.test_accs and self.test_f1s and self.test_precisions and self.test_recalls:
            test_epochs = range(self.test_update_freq,
                                len(self.test_accs) * self.test_update_freq + 1,
                                self.test_update_freq)
            if self.test_accs:
                plt.plot(test_epochs, self.test_accs, label="test_acc")
            if self.test_f1s:
                plt.plot(test_epochs, self.test_f1s, label="test_f1")
            if self.test_precisions:
                plt.plot(test_epochs, self.test_precisions, label="test_precision")
            if self.test_recalls:
                plt.plot(test_epochs, self.test_recalls, label="test_recall")
            if self.test_aucs:
                plt.plot(test_epochs, self.test_aucs, label="test_auc")
            if self.test_aps:
                plt.plot(test_epochs, self.test_aps, label="test_ap")

        plt.xlabel("Epoch")
        plt.ylabel("Metric")
        plt.title(f"{model_type} Training Metrics")
        plt.legend()

        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            print(f"Saving plot {model_type}_metrics.png")
            filename = os.path.join(output_dir, f"{model_type}_metrics.png")
            plt.savefig(filename)

        plt.show()

    def save_model(self, filename, output_dir='checkpoints'):
        """
        :param filename:
        :param output_dir:
        :return:
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        filename = os.path.join(output_dir, filename)
        torch.save(self.model.state_dict(), filename)

    def save_checkpoint(self, epoch, optimizer_state, output_dir='checkpoints', model_name='model'):
        """save checkpoint each model gets on dir.
        :param model_name:
        :param epoch:
        :param optimizer_state:
        :param output_dir:
        :return:
        """
        checkpoint_dir = os.path.join(output_dir, model_name)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{epoch}.pt")

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer_state
        }, checkpoint_path)

    def load_checkpoint(self, model_type: str, checkpoint_dir: str = 'checkpoints'):
        """Loads the last saved checkpoint from a given directory.
        :param model_type:
        :param checkpoint_dir: Path to the checkpoint dir
        """
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        checkpoint_dir = os.path.join(checkpoint_dir, model_type)
        if not os.path.exists(checkpoint_dir):
            print(f"Checkpoint directory {checkpoint_dir} does not exist.")
            return

        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
        metrics_files = sorted(glob.glob(os.path.join(self.metrics_dir, f"{model_type}*.pt")))
        if not checkpoint_files:
            print("No checkpoints found.")
            return

        checkpoint_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]))
        checkpoint_path = checkpoint_files[-1]
        checkpoint = torch.load(checkpoint_path)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.current_epoch = checkpoint['epoch'] + 1

        if self.save_metrics and f"{model_type}_train_metrics.pt" in metrics_files:
            train_metrics = torch.load(os.path.join(self.metrics_dir, f"{model_type}_train_metrics.pt"))
            print(f"Loaded test metric from {train_metrics}.")
            self.train_losses = train_metrics["train_losses"]
            self.train_accs = train_metrics["train_accs"]
            self.train_f1s = train_metrics["train_f1s"]
            self.train_precisions = train_metrics["train_precisions"]
            self.train_recalls = train_metrics["train_recalls"]
        if self.save_metrics and f"{model_type}_validation_metrics.pt" in metrics_files:
            val_metrics = torch.load(os.path.join(self.metrics_dir, f"{model_type}_validation_metrics.pt"))
            print(f"Loaded test metric from {val_metrics}.")
            self.val_accs = val_metrics["val_accs"]
            self.val_f1s = val_metrics["val_f1s"]
            self.val_precisions = val_metrics["val_precisions"]
            self.val_recalls = val_metrics["val_recalls"]
        if self.save_metrics and f"{model_type}_test_metrics.pt" in metrics_files:
            test_metrics = torch.load(os.path.join(self.metrics_dir, f"{model_type}_test_metrics.pt"))
            print(f"Loaded test metric from {test_metrics}.")
            self.test_accs = test_metrics["test_accs"]
            self.test_f1s = test_metrics["test_f1s"]
            self.test_precisions = test_metrics["test_precisions"]
            self.test_recalls = test_metrics["test_recalls"]

        print(f"Loaded checkpoint from {checkpoint_path}.")

    def update_test_metric(
            self,
            test_acc: Optional[float] = None,
            test_f1: Optional[float] = None,
            test_precision: Optional[float] = None,
            test_recall: Optional[float] = None,
            test_ap: Optional[float] = None,
            test_auc: Optional[float] = None):
        """Update test metrics.

        :param test_ap:
        :param test_auc:
        :param test_acc:
        :param test_f1:
        :param test_precision:
        :param test_recall:
        :return:
        """
        if test_acc is not None:
            self.test_accs.append(test_acc)
        if test_f1 is not None:
            self.test_f1s.append(test_f1)
        if test_precision is not None:
            self.test_precisions.append(test_precision)
        if test_recall is not None:
            self.test_recalls.append(test_recall)
        if test_auc is not None:
            self.test_aucs.append(test_auc)
        if test_ap is not None:
            self.test_aps.append(test_ap)

        if self.save_metrics and self.metrics_rate and self.current_epoch % self.metrics_rate == 0:
            metrics = {
                'test_accs': self.val_accs,
                'tet_f1s': self.val_f1s,
                'test_precisions': self.val_precisions,
                'test_recalls': self.val_recalls,
                'test_aucs': self.test_aucs,
                'test_aps': self.test_aps,
            }
            if self.metrics_dir:
                if not os.path.exists(self.metrics_dir):
                    os.makedirs(self.metrics_dir)
            filename = os.path.join(self.metrics_dir, f"{self.model_type}_test_metrics.pt")
            torch.save(metrics, filename)

    def update_metrics(self, loss_avg: float,
                       train_f1: Optional[float] = None,
                       train_acc: Optional[float] = None,
                       train_recall: Optional[float] = None,
                       train_precision: Optional[float] = None):
        """Update the training metrics for each epoch.

        :param loss_avg: The average training loss for the current epoch.
        :type loss_avg: float
        :param train_f1: The training F1 score for the current epoch (optional).
        :type train_f1: float or None
        :param train_acc: The training accuracy for the current epoch (optional).
        :type train_acc: float or None
        :param train_recall: The training recall score for the current epoch (optional).
        :type train_recall: float or None
        :param train_precision: The training precision score for the current epoch (optional).
        :type train_precision: float or None
        """
        if loss_avg is not None:
            self.train_losses.append(loss_avg)
        if train_f1 is not None:
            self.train_f1s.append(train_f1)
        if train_acc is not None:
            self.train_accs.append(train_acc)
        if train_recall is not None:
            self.train_recalls.append(train_recall)
        if train_precision is not None:
            self.train_precisions.append(train_precision)

        if self.save_metrics and self.metrics_rate and self.current_epoch % self.metrics_rate == 0:
            if self.metrics_dir:
                if not os.path.exists(self.metrics_dir):
                    os.makedirs(self.metrics_dir)
            metrics = {
                'test_accs': self.val_accs,
                'tet_f1s': self.val_f1s,
                'test_precisions': self.val_precisions,
                'test_recalls': self.val_recalls
            }
            filename = os.path.join(self.metrics_dir, f"{self.model_type}_train_metrics.pt")
            torch.save(metrics, filename)

    def update_val_metric(self, val_acc, val_f1, val_precision, val_recall):
        """ Update validation metrics.
        :param val_acc: validation accuracy.
        :param val_f1: validation f1 score.
        :param val_precision: validation precision.
        :param val_recall: validation recall.
        :return:
        """
        self.val_accs.append(val_acc)
        self.val_f1s.append(val_f1)
        self.val_precisions.append(val_precision)
        self.val_recalls.append(val_recall)

        if self.save_metrics and self.metrics_rate and self.current_epoch % self.metrics_rate == 0:
            metrics = {
                'val_accs': self.val_accs,
                'val_f1s': self.val_f1s,
                'val_precisions': self.val_precisions,
                'val_recalls': self.val_recalls
            }
            filename = os.path.join(self.metrics_dir, f"{self.model_type}_validation_metrics.pt")
            torch.save(metrics, filename)

    def split(self, ds):
        """
        :return:
        """
        if ds is None:
            raise ValueError("ds cannot be None")

        ds = ds.shuffle()
        # Define the train-test split ratio (e.g., 80% train, 20% test)
        train_size = int(len(ds) * self.train_ration)
        test_size = len(ds) - train_size
        self.train_dataset, self.test_dataset = torch.utils.data.random_split(ds, [train_size, test_size])

    @staticmethod
    def remove_nodes_and_edges(data, remove_ratio=0.2):
        """remove nodes and edges.
        :param data:
        :param remove_ratio:
        :return:
        """
        if data is None:
            raise ValueError("data cannot be None")

        num_nodes_to_remove = int(data.num_nodes * remove_ratio)
        nodes_to_remove = np.random.choice(data.num_nodes, num_nodes_to_remove, replace=False)
        mask = torch.ones(data.num_nodes, dtype=torch.bool)
        mask[nodes_to_remove] = False
        data.x = data.x[mask]
        data.edge_index = data.edge_index[:, mask[data.edge_index[0]] & mask[data.edge_index[1]]]
        return data

    def dataset_split_mask(self, split_mask_ration=0.8):
        """Compute split mask
        :return:
        """
        if self._dataset is None:
            raise ValueError("ds cannot be None")

        # Split the dataset into a train set and a test set
        train_size = int(split_mask_ration * len(self._dataset))
        test_size = len(self._dataset) - train_size
        train_dataset, test_dataset = random_split(self._dataset, [train_size, test_size])
        # create binary masks for the train set and test set
        train_mask = torch.zeros(len(self._dataset), dtype=torch.bool)
        test_mask = torch.zeros(len(self._dataset), dtype=torch.bool)
        train_indices = train_dataset.indices
        test_indices = test_dataset.indices
        train_mask[train_indices] = True
        test_mask[test_indices] = True

    def clear_metric(self):
        """

        :return:
        """
        del self.val_precisions
        self.val_precisions = []
        del self.val_f1s
        self.val_f1s = []
        del self.val_accs
        self.val_accs = []
        del self.val_recalls
        self.val_recalls = []
        del self.test_recalls
        self.test_recalls = []
        del self.test_precisions
        self.test_precisions = []
        del self.test_accs
        self.test_accs = []
        del self.test_f1s
        self.test_f1s = []
        del self.train_precisions
        self.train_precisions = []
        del self.train_recalls
        self.train_recalls = []
        del self.train_f1s
        self.train_f1s = []
        del self.train_accs
        self.train_accs = []
        del self.train_losses
        self.train_losses = []
        del self.test_aps
        self.test_aps = []
        del self.test_aucs
        self.test_aucs = []


def complex_synthetic():
    """geneerate graph useful to test LSTM and djakstra.
    :return:
    """
    subgraph_x = torch.tensor([
        [1, 0],
        [0, 1],
        [1, 1],
        [0, 0],
        [1, 0]
    ], dtype=torch.float)

    subgraph_edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 0],
        [1, 0, 2, 1, 3, 2, 4, 3, 0, 4]
    ], dtype=torch.long)

    subgraph_edge_attr = torch.tensor([0.5, 0.6, 0.3,
                                       0.2, 0.9, 0.4, 0.7,
                                       0.8, 0.1, 0.5], dtype=torch.float)
    subgraph_data = Data(
        x=subgraph_x,
        edge_index=subgraph_edge_index,
        edge_attr=subgraph_edge_attr)
    subgraph_data_list = [subgraph_data]
    return subgraph_data_list


def create_synthetic_data():
    """
    :return:
    """
    subgraph_x = torch.tensor([
        [1, 0],
        [0, 1],
        [1, 1]
    ], dtype=torch.float)

    subgraph_edge_index = torch.tensor([
        [0, 1, 1, 2],
        [1, 0, 2, 1]
    ], dtype=torch.long)

    # 3 edges
    subgraph_edge_attr = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float)
    subgraph_data = Data(
        x=subgraph_x,
        edge_index=subgraph_edge_index,
        edge_attr=subgraph_edge_attr,
        y=torch.tensor([1, 0, 0]))

    return subgraph_data

