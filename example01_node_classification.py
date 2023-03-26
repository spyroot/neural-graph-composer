"""
First problem a node classification.

We have a MIDI dataset and want a classifier to
classify each node in the input graph.

Example Node classification based on 3 models.In this example
we're trying to learn from a graph that represent.

The default model set to GCN with 3 layer and PRELU activation.
Second model GAT and we also include weighted edges.

Graph Neural Ntwork layer

Author Mus spyroot@gmail.com
           mbayramo@stanford.edu
"""
import argparse
import glob
import os
from enum import Enum
from typing import Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric
import torch_geometric.transforms as T
from torch.optim.lr_scheduler import StepLR
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GINConv, GATConv
import wandb

from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from example_shared import Experiments
from neural_graph_composer.midi_dataset import MidiDataset


class Activation(Enum):
    """Activation enum so we can experiment.
    """
    ReLU = 'relu'
    PReLU = 'prelu'
    ELU = 'elu'
    SELU = 'selu'
    Tanh = 'tanh'


class GCN2(torch.nn.Module):
    """ Two layer GCN with option to swap activation layers """
    def __init__(
            self, num_feature,
            hidden_channels,
            num_classes,
            activation: Optional[Activation] = Activation.ReLU,
            dropout_p=0.5):
        super(GCN2, self).__init__()

        if activation == Activation.ReLU:
            self.activation = nn.ReLU()
        elif activation == Activation.PReLU:
            self.activation = nn.PReLU(hidden_channels)
        elif activation == Activation.ELU:
            self.activation = nn.ELU(alpha=1.0)
        elif activation == Activation.SELU:
            self.activation = nn.SELU()
        elif activation == Activation.Tanh:
            self.activation = nn.Tanh()

        self.conv1 = GCNConv(num_feature, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)
        self.dropout_p = dropout_p

    def forward(self, data):
        """Standard GCN layer batch data.
        :param data:
        :return:
        """
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GCN3(torch.nn.Module):
    """Three layer GCN with option swap activation layer.
    """
    def __init__(
            self, num_feature: int,
            hidden_channels: int,
            num_classes: int,
            activation: Optional[Activation] = Activation.ReLU,
            dropout_p: float = 0.5):
        super(GCN3, self).__init__()

        if activation == Activation.ReLU:
            self.activation = nn.ReLU()
        elif activation == Activation.PReLU:
            self.activation = nn.PReLU(hidden_channels)
        elif activation == Activation.ELU:
            self.activation = nn.ELU(alpha=1.0)
        elif activation == Activation.SELU:
            self.activation = nn.SELU()
        elif activation == Activation.Tanh:
            self.activation = nn.Tanh()

        self.conv1 = GCNConv(num_feature, hidden_channels)
        self.prelu01 = nn.PReLU(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.prelu02 = nn.PReLU(hidden_channels)
        self.conv3 = GCNConv(hidden_channels, num_classes)
        self.dropout_p = dropout_p

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)


class GIN(torch.nn.Module):
    """GIN layer. Note the model output different shape that way we have
    re-structure output to get same re-presentation as GCN3 that way we have special
    case in train loop for GIN and GAT.
    """
    def __init__(self, num_feature, hidden_channels, num_classes: int):
        super(GIN, self).__init__()
        self.conv1 = GINConv(torch.nn.Sequential(
            torch.nn.Linear(num_feature, hidden_channels), torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels)))
        self.conv2 = GINConv(torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels), torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels)))

        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        x = torch_geometric.nn.global_mean_pool(x, batch)
        return F.log_softmax(x, dim=1)


class GAT(torch.nn.Module):
    """GAT that also  include edge_weight since we have directed weight graph.
     we want to use weights.  Note the model output different shape that way we have
    re-structure output to get same re-presentation as GCN3 that way we have special
    case in train loop for GIN and GAT.
    """
    def __init__(self, num_feature, hidden_channels, num_classes,
                 use_edge_weights=True, dropout=0.3):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.use_edge_weights = use_edge_weights
        self.conv1 = GATConv(num_feature, hidden_channels, add_self_loops=True)
        self.conv2 = GATConv(hidden_channels, num_classes, add_self_loops=True)

    def forward(self, data):
        """Forward pass note we also pass edge_weight
        :param data:
        :return:
        """
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_attr, data.batch
        if self.use_edge_weights:
            x = self.conv1(x, edge_index, edge_weight)
        else:
            x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.use_edge_weights:
            x = self.conv2(x, edge_index, edge_weight)
        else:
            x = self.conv2(x, edge_index)
        x = torch_geometric.nn.global_mean_pool(x, batch)
        return F.log_softmax(x, dim=1)


class ExampleNodeClassification(Experiments):
    """
    In first experiment node classification task.
    We have a MIDI dataset and want a classifier to classify each node in the input graph.
    Model allow to swap activation and experiment with GCN , GIN and GAT.
    """
    def __init__(
            self, epochs: int,
            batch_size: int,
            midi_dataset: MidiDataset, hidden_dim: int,
            model_type: Optional[str] = "GCN3",
            lr: Optional[float] = 0.01,
            activation: Optional[Activation] = Activation.ReLU,
            train_update_rate: Optional[int] = 1,
            test_update_freq: Optional[int] = 10,
            eval_update_freq: Optional[int] = 20,
            save_freq: Optional[int] = 20,
            is_data_split: Optional[bool] = False):
        """Example experiment for training a graph neural network on MIDI data.

        :param epochs: num epochs
        :param batch_size: default batch (for colab on GPU use 4)
        :param hidden_dim: hidden for all models.
        :param model_type:  (GCN3/GAT)
         :param lr: learning rate.
        """
        super().__init__(
            epochs, batch_size, midi_dataset,
            train_update_rate=train_update_rate,
            test_update_freq=test_update_freq,
            eval_update_freq=eval_update_freq,
            save_freq=save_freq)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        assert self.device is not None, "Device is not set."
        assert self.datasize is not None, "Datasize is not set."
        assert self.test_size is not None, "Test size is not set."
        assert self._num_workers is not None, "Number of workers is not set."
        assert self._batch_size is not None, "Batch size is not set."

        self.datasize = 0
        self.test_size = 0
        self._num_workers = 0
        self._batch_size = batch_size
        self._hidden_dim = hidden_dim
        self._feature_dim = midi_dataset.num_node_features
        self._num_classes = midi_dataset.total_num_classes
        self._lr = lr
        self._is_gin = False
        self._is_gat = False

        self.train_dataset = midi_dataset
        self.test_dataset = midi_dataset

        self.train_loader = DataLoader(
            midi_dataset, batch_size=self._batch_size, shuffle=True)

        # if we do random split it produce different dataset,
        # at least this our current understanding how it works. :)
        # if it wrong please let me know.
        self.val_loader = None
        self.test_loader = None

        if is_data_split:
            self.val_loader = DataLoader(
                midi_dataset, batch_size=batch_size, shuffle=False)

            self.test_loader = DataLoader(
                midi_dataset, batch_size=batch_size, shuffle=False)

        self.model_type = model_type
        if model_type == "GCN3":
            print(f"Creating GCN3 feature dim: {self._feature_dim} "
                  f"hidden size {self._hidden_dim} num classes "
                  f"{self._num_classes} batch size {self._batch_size} "
                  f"lr {self._lr} activate {activation.value}")
            self.model = GCN3(
                self._feature_dim, self._hidden_dim,
                self._num_classes, activation=activation).to(self.device)
        elif model_type == "GAT":
            print(f"Creating GAT feature dim: {self._feature_dim} "
                  f"hidden size {self._hidden_dim} num classes "
                  f"{self._num_classes} batch size {self._batch_size} lr {self._lr}").to(self.device)
            self.model = GAT(
                self._feature_dim, self._hidden_dim, self._num_classes
            ).to(self.device)
            self._is_gat = True
        else:
            self.model = GIN(
                self._feature_dim, self._hidden_dim, self._num_classes)
            print(f"Creating GIN feature dim: {self._feature_dim} "
                  f"hidden size {self._hidden_dim} num classes "
                  f"{self._num_classes} batch size {self._batch_size} lr {self._lr}")
            self._is_gin = True

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self._lr, weight_decay=5e-4)

        self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.1)

    def train_epoch(self):
        """
        :return:
        """
        self.model.train()
        epoch_loss = 0.0
        loss_all = 0.0
        tp = 0.0
        fp = 0.0
        fn = 0.0

        pred_train_all = []
        y_train_all = []
        total_graph = 0

        for i, b in enumerate(self.train_loader):
            train_batch = b
            #
            # if isinstance(b, list):
            #     train_batch, _, _ = b
            # else:
            #     train_batch = b

            train_batch.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(train_batch)

            train_mask = train_batch.train_mask
            y_train = train_batch.y[train_mask]

            if self._is_gin:
                node_idx = torch.arange(out.shape[0]).to(self.device)
                out_sum = torch.zeros((train_batch.num_nodes, out.shape[1]), dtype=torch.float).to(self.device)
                out_sum.scatter_(0, node_idx.unsqueeze(1).expand(-1, out.shape[1]), out)
                out_train = out_sum[train_mask]
            elif self._is_gat:
                node_idx = torch.arange(out.shape[0]).to(self.device)
                out_sum = torch.zeros((train_batch.num_nodes, out.shape[1]), dtype=torch.float).to(self.device)
                out_sum.scatter_(0, node_idx.unsqueeze(1).expand(-1, out.shape[1]), out)
                out_train = out_sum[train_mask, -self._num_classes:]
            else:
                out_train = out[train_mask]

            loss = F.nll_loss(out_train, y_train)
            loss.backward()
            self.optimizer.step()

            loss_all += train_batch.num_graphs * loss.item()
            epoch_loss += loss.item()
            total_graph += train_batch.num_graphs

            # calculate F1 and accuracy metrics
            pred_train = out_train.argmax(dim=1).cpu().numpy()
            y_train_np = y_train.cpu().numpy()
            pred_train_all.append(pred_train)
            y_train_all.append(y_train_np)

            pred_class_idx = torch.argmax(out_train, dim=1)
            tp += torch.sum((pred_class_idx == 1) & (y_train == 1)).item()
            fp += torch.sum((pred_class_idx == 1) & (y_train == 0)).item()
            fn += torch.sum((pred_class_idx == 0) & (y_train == 1)).item()

        # calculate F1 and accuracy metrics
        pred_train_all = np.concatenate(pred_train_all, axis=0)
        y_train_all = np.concatenate(y_train_all, axis=0)
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        train_f1 = f1_score(y_train_all, pred_train_all, average='macro')
        train_acc = (pred_train_all == y_train_all).sum() / len(y_train_all)

        loss_avg = loss_all / total_graph
        return loss_avg, train_f1, train_acc, recall, precision

    def train(self):
        """
        :return:
        """
        self.load_checkpoint(self.model_type)

        best_val_acc = 0.
        best_epoch = 0.
        best_test_acc = 0.

        current_epoch = len(self.train_losses)

        for e in range(current_epoch, self._epochs + 1):
            loss_avg, train_f1, train_acc, train_recall, train_precision = self.train_epoch()
            print(
                f"Epoch: {e}, "
                f"Loss: {loss_avg:.5f}, "
                f"Train Acc: {train_acc:.5f}, "
                f"Train F1: {train_f1:.5f}, "
                f"Train Precision: {train_precision:.5f}, "
                f"Train Recall: {train_recall:.5f}")

            self.scheduler.step()
            self.update_metrics(loss_avg, train_f1, train_acc, train_recall, train_precision)

            if e % self.test_update_freq == 0:
                test_acc, test_f1, test_precision, test_recall = self.evaluate(
                    is_eval=False)

                print(
                    f"Epoch: {e}, "
                    f"Test Acc: {test_acc:.5f}, "
                    f"Test F1: {test_f1:.5f}, "
                    f"Test Precision: {test_precision:.5f}, "
                    f"Test Recall: {test_recall:.5f}")
                self.update_test_metric(test_acc, test_f1, test_precision, test_recall)

            if e % self.eval_update_freq == 0:
                val_acc, val_f1, val_precision, val_recall = self.evaluate(is_eval=False)
                test_acc, test_f1, test_precision, test_recall = self.evaluate(is_eval=True)
                self.update_test_metric(test_acc, test_f1, test_precision, test_recall)
                self.update_val_metric(val_acc, val_f1, val_precision, val_recall)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = e
                    best_test_acc = test_acc

                print(
                    f"Epoch: {e}, "
                    f"Val Acc: {val_acc:.5f}, "
                    f"Val F1: {val_f1:.5f}, "
                    f"Val Precision: {val_precision:.5f}, "
                    f"Val Recall: {val_recall:.5f}, "
                    f"Test Acc: {test_acc:.5f}, "
                    f"Test F1: {test_f1:.5f}, "
                    f"Test Precision: {test_precision:.5f}, "
                    f"Test Recall: {test_recall:.5f}")

            if e % self.save_freq == 0:
                self.save_checkpoint(e, self.optimizer.state_dict(), model_name=self.model_type)

        print(f"Best Epoch: {best_epoch}, Test Acc: {best_test_acc:.5f}")
        self.plot_metrics(self._epochs)

    @torch.no_grad()
    def evaluate(self, is_eval: bool):
        """
        :return:
        """
        self.model.eval()
        if is_eval:
            data_loader = self.val_loader
        else:
            data_loader = self.test_loader

        total = 0
        correct = 0
        tp, fp, fn = 0., 0., 0.

        for b in iter(data_loader):
            if isinstance(b, list):
                if is_eval:
                    _, val, _ = b
                    batch = val
                else:
                    _, _, test_b = b
                    batch = test_b
            else:
                batch = b

            mask = batch.test_mask

            if is_eval:
                mask = batch.val_mask

            data = batch.to(self.device)
            out = self.model(data)

            if self._is_gin:
                node_idx = torch.arange(out.shape[0]).to(self.device)
                out_sum = torch.zeros((batch.num_nodes, out.shape[1]), dtype=torch.float).to(self.device)
                out_sum.scatter_(0, node_idx.unsqueeze(1).expand(-1, out.shape[1]), out)
                pred_masked = out_sum[mask]
            elif self._is_gat:
                node_idx = torch.arange(out.shape[0]).to(self.device)
                out_sum = torch.zeros((batch.num_nodes, out.shape[1]), dtype=torch.float).to(self.device)
                out_sum.scatter_(0, node_idx.unsqueeze(1).expand(-1, out.shape[1]), out)
                pred_masked = out_sum[mask, -self._num_classes:]
            else:
                pred_masked = out[mask]
            # correct2 = int((out.argmax(dim=-1) == data.y).sum())
            # print(out.argmax(dim=-1))
            # print(f"correct unmasked {correct2}")

            y_masked = batch.y[mask]
            pred_class_idx = torch.argmax(pred_masked, dim=1)
            batch_correct = torch.sum(torch.eq(pred_class_idx, y_masked)).item()
            correct += batch_correct

            # print(f"correct  {batch_correct} "
            #       f"{y_masked.shape[0]} {pred_class_idx.shape }"
            #       f"mask shape batch correct {batch_correct / y_masked.shape[0]}")

            # print(f"correct masked argmax {correct} "
            #       f"{y_masked.shape} mask shape "
            #       f"{pred_class_idx.shape}")

            # _y_masked = batch.y[mask].cpu().numpy()
            # _pred_class_idx = torch.argmax(pred_masked, dim=1).cpu().numpy()
            # _accuracy = accuracy_score(_y_masked, _pred_class_idx)
            # print(_accuracy)
            # calculate tp, fp, fn for F1 score, precision, and recall

            tp += torch.sum((pred_class_idx == 1) & (y_masked == 1)).item()
            fp += torch.sum((pred_class_idx == 1) & (y_masked == 0)).item()
            fn += torch.sum((pred_class_idx == 0) & (y_masked == 1)).item()
            total += y_masked.shape[0]

        accuracy = correct / total
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
        return accuracy, f1, precision, recall

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

    def save_checkpoint(
            self,
            epoch,
            optimizer_state,
            output_dir='checkpoints',
            model_name='model'):
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

    def load_model(self, model_path):
        """Load the saved model from the specified path
        :param model_path:
        :return:
        """
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()


if __name__ == '__main__':
    """
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--model_type', type=str, default='GCN3', choices=['GCN3', 'GIN', 'GAT'])
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--graph_per_instrument', type=bool, default=False)
    parser.add_argument('--wandb', action='store_true', help='Track experiment')
    parser.add_argument('--random_split', type=bool, default=False)

    # init_wandb(name=f'GCN3-{args.dataset}', heads=args.heads, epochs=args.epochs,
    #            hidden_channels=args.hidden_channels, lr=args.lr, device=device)

    # todo add node random split and test.
    args = parser.parse_args()
    if args.random_split:
        transform = T.Compose([
            T.NormalizeFeatures(),
            T.ToDevice(device),
            T.RandomLinkSplit(
                num_val=0.05,
                num_test=0.1,
                is_undirected=False,
                split_labels=True,
                add_negative_train_samples=False)
        ])
    else:
        transform = T.Compose([
            T.NormalizeFeatures(),
            T.ToDevice(device),
        ])

    ds = MidiDataset(root="./data",
                     transform=transform,
                     per_instrument_graph=args.graph_per_instrument,
                     tolerance=0.5)

    example_model = ExampleNodeClassification(
        epochs=args.epochs,
        batch_size=args.batch_size,
        midi_dataset=ds,
        hidden_dim=args.hidden_dim,
        model_type=args.model_type,
        lr=args.lr,
        activation=Activation.PReLU)
    example_model.train()
