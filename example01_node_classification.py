import os
from typing import Optional
import numpy as np
import torch_geometric
from matplotlib import pyplot as plt
from torch_geometric.nn import GCNConv, GINConv, GATConv
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import argparse

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap

from example_shared import Experiments
from neural_graph_composer.midi_dataset import MidiDataset


class GCN2(torch.nn.Module):
    def __init__(self, num_feature, hidden_channels, num_classes):
        super(GCN2, self).__init__()
        self.conv1 = GCNConv(num_feature, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, data):
        """

        :param data:
        :return:
        """
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GCN3(torch.nn.Module):
    def __init__(self, num_feature, hidden_channels, num_classes):
        super(GCN3, self).__init__()
        self.conv1 = GCNConv(num_feature, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)


class GIN(torch.nn.Module):
    """
    """

    def __init__(self, num_feature, hidden_channels, num_classes):
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
    """
    """

    def __init__(self, num_feature, hidden_channels, num_classes,
                 use_edge_weights=True, dropout=0.3):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.use_edge_weights = use_edge_weights
        self.conv1 = GATConv(num_feature, hidden_channels, add_self_loops=True)
        self.conv2 = GATConv(hidden_channels, num_classes, add_self_loops=True)

    def forward(self, data):
        """

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


class Example01(Experiments):
    def __init__(
            self, epochs: int, batch_size: int,
            midi_dataset: MidiDataset, hidden_dim: int,
            model_type: Optional[str] = "GCN3", lr: Optional[float] = 0.01):
        """
           Example experiment for training a graph neural network on MIDI data.
        :param epochs:
        :param batch_size:
        :param hidden_dim:
        :param model_type:
        :param lr:
        """
        super().__init__(epochs, batch_size, midi_dataset)
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
        self._is_gin = False
        self._is_gat = False
        self._lr = lr

        self.train_dataset = midi_dataset
        self.test_dataset = midi_dataset

        self.train_loader = DataLoader(
            midi_dataset, batch_size=self._batch_size, shuffle=True)

        self.val_loader = DataLoader(
            midi_dataset, batch_size=batch_size, shuffle=False)

        self.test_loader = DataLoader(
            midi_dataset, batch_size=batch_size, shuffle=False)

        if model_type == "GCN3":
            print(f"Creating GCN3 {self._feature_dim} {self._hidden_dim} {self._num_classes}")
            self.model = GCN3(
                self._feature_dim, self._hidden_dim, self._num_classes)
        elif model_type == "GAT":
            print(f"Creating GAT {self._feature_dim} {self._hidden_dim} {self._num_classes}")
            self.model = GAT(
                self._feature_dim, self._hidden_dim, self._num_classes)
            self._is_gat = True
        else:
            self.model = GIN(
                self._feature_dim, self._hidden_dim, self._num_classes)
            self._is_gin = True

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self._lr, weight_decay=5e-4)

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

        for i, b in enumerate(self.train_loader):
            train_batch, _, _ = b
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

            # Calculate F1 and accuracy metrics
            pred_train = out_train.argmax(dim=1).cpu().numpy()
            y_train_np = y_train.cpu().numpy()
            pred_train_all.append(pred_train)
            y_train_all.append(y_train_np)

            pred_class_idx = torch.argmax(out_train, dim=1)
            tp += torch.sum((pred_class_idx == 1) & (y_train == 1)).item()
            fp += torch.sum((pred_class_idx == 1) & (y_train == 0)).item()
            fn += torch.sum((pred_class_idx == 0) & (y_train == 1)).item()

        # Calculate F1 and accuracy metrics
        pred_train_all = np.concatenate(pred_train_all, axis=0)
        y_train_all = np.concatenate(y_train_all, axis=0)

        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        train_f1 = f1_score(y_train_all, pred_train_all, average='macro')
        train_acc = (pred_train_all == y_train_all).sum() / len(y_train_all)

        loss_avg = loss_all / len(self.train_loader.dataset)
        return loss_avg, train_f1, train_acc, recall, precision

    def plot_metrics(
            self, train_loss, train_f1, train_acc,
            train_precision, train_recall,
            val_acc, val_f1, val_precision, val_recall,
            test_acc, test_f1, test_precision, test_recall,
            num_epochs, output_dir="metric", run=None):
        """
        :param run:
        :param output_dir:
        :param train_loss:
        :param train_f1:
        :param train_acc:
        :param train_precision:
        :param train_recall:
        :param val_acc:
        :param val_f1:
        :param val_precision:
        :param val_recall:
        :param test_acc:
        :param test_f1:
        :param test_precision:
        :param test_recall:
        :param num_epochs:
        :return:
        """
        print(f"EPOCS {num_epochs}")
        epochs = range(1, num_epochs + 1)
        plt.figure(figsize=(15, 10))
        plt.plot(epochs, train_loss, label="train_loss")
        plt.plot(epochs, train_f1, label="train_f1")
        plt.plot(epochs, train_acc, label="train_acc")
        plt.plot(epochs, train_precision, label="train_precision")
        plt.plot(epochs, train_recall, label="train_recall")
        plt.plot(epochs, val_acc, label="val_acc")
        plt.plot(epochs, val_f1, label="val_f1")
        plt.plot(epochs, val_precision, label="val_precision")
        plt.plot(epochs, val_recall, label="val_recall")
        plt.plot(epochs, test_acc, label="test_acc")
        plt.plot(epochs, test_f1, label="test_f1")
        plt.plot(epochs, test_precision, label="test_precision")
        plt.plot(epochs, test_recall, label="test_recall")
        plt.xlabel("Epoch")
        plt.ylabel("Metric")
        plt.title("Training Metrics")
        plt.legend()

        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            filename = os.path.join(output_dir, "metrics.png")
            plt.savefig(filename)

        plt.show()

    def train(self):
        """
        :return:
        """
        best_val_acc = 0.
        best_epoch = 0.
        best_test_acc = 0.

        train_losses = []
        train_f1s = []
        train_accs = []
        train_recalls = []
        train_precisions = []

        val_accs = []
        val_f1s = []
        val_precisions = []
        val_recalls = []

        test_accs = []
        test_f1s = []
        test_precisions = []
        test_recalls = []

        for e in range(1, self._epochs + 1):

            loss_avg, train_f1, train_acc, train_recall, train_precision = self.train_epoch()
            # train_acc, train_f1, train_precision, train_recall = self.evaluate(self.train_loader)
            val_acc, val_f1, val_precision, val_recall = self.evaluate(self.val_loader)
            test_acc, test_f1, test_precision, test_recall = self.evaluate(self.test_loader)

            train_losses.append(loss_avg)
            train_f1s.append(train_f1)
            train_accs.append(train_acc)
            train_recalls.append(train_recall)
            train_precisions.append(train_precision)

            val_accs.append(val_acc)
            val_f1s.append(val_f1)
            val_precisions.append(val_precision)
            val_recalls.append(val_recall)

            test_accs.append(test_acc)
            test_f1s.append(test_f1)
            test_precisions.append(test_precision)
            test_recalls.append(test_recall)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = e
                best_test_acc = test_acc

            if e % 5 == 0:
                print(
                    f"Epoch: {e}, "
                    f"Loss: {loss_avg:.5f}, "
                    f"Train Acc: {train_acc:.5f}, "
                    f"Train F1: {train_f1:.5f}, "
                    f"Train Precision: {train_precision:.5f}, "
                    f"Train Recall: {train_recall:.5f}")

            if e % 10 == 0:
                val_acc, val_f1, val_precision, val_recall = self.evaluate("val")
                test_acc, test_f1, test_precision, test_recall = self.evaluate("test")

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

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = e
                best_test_acc = test_acc

        print(f"Best Epoch: {best_epoch}, Test Acc: {best_test_acc:.5f}")
        self.plot_metrics(
            train_losses, train_f1s, train_accs, train_precisions,
            train_recalls, val_accs, val_f1s, val_precisions, val_recalls,
            test_accs, test_f1s, test_precisions, test_recalls, self._epochs )

    def evaluate(self, eval_type: str):
        """
        :return:
        """
        if eval_type == "val":
            data_loader = self.val_loader
        else:
            data_loader = self.test_loader

        self.model.eval()

        total = 0
        correct = 0
        tp = 0
        fp = 0
        fn = 0
        for b in data_loader:
            if eval_type == "val":
                _, val, _ = b
                batch = val
            else:
                _, _, test_b = b
                batch = test_b

            test_mask = batch.test_mask
            data = batch.to(self.device)
            out = self.model(data)

            if self._is_gin:
                node_idx = torch.arange(out.shape[0]).to(self.device)
                out_sum = torch.zeros((batch.num_nodes, out.shape[1]), dtype=torch.float).to(self.device)
                out_sum.scatter_(0, node_idx.unsqueeze(1).expand(-1, out.shape[1]), out)
                pred_masked = out_sum[test_mask]
            elif self._is_gat:
                node_idx = torch.arange(out.shape[0]).to(self.device)
                out_sum = torch.zeros((batch.num_nodes, out.shape[1]), dtype=torch.float).to(self.device)
                out_sum.scatter_(0, node_idx.unsqueeze(1).expand(-1, out.shape[1]), out)
                pred_masked = out_sum[test_mask, -self._num_classes:]
            else:
                pred_masked = out[test_mask]

            y_masked = batch.y[test_mask]
            pred_class_idx = torch.argmax(pred_masked, dim=1)
            correct += torch.sum(torch.eq(pred_class_idx, y_masked)).item()

            _y_masked = batch.y[test_mask].cpu().numpy()
            _pred_class_idx = torch.argmax(pred_masked, dim=1).cpu().numpy()
            _accuracy = accuracy_score(_y_masked, _pred_class_idx)

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


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--model_type', type=str, default='GCN3', choices=['GCN3', 'GIN', 'GAT'])
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--graph_per_instrument', type=bool, default=False)

    args = parser.parse_args()
    #

    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        T.RandomLinkSplit(
            num_val=0.05,
            num_test=0.1,
            is_undirected=True,
            split_labels=True,
            add_negative_train_samples=False)
    ])

    ds = MidiDataset(root="./data", transform=transform, per_instrument_graph=args.graph_per_instrument)

    example = Example01(
        epochs=args.epochs,
        batch_size=args.batch_size,
        midi_dataset=ds,
        hidden_dim=args.hidden_dim,
        model_type=args.model_type,
        lr=args.lr)
    example.train()