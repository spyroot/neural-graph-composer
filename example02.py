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


class ExampleOneModel(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        """

        :param num_features:
        :param num_classes:
        """
        super().__init__()
        self.conv1 = GraphConv(num_features, 128)
        self.pool1 = TopKPooling(128, ratio=0.8)
        self.conv2 = GraphConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.8)
        self.conv3 = GraphConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=0.8)

        self.lin1 = torch.nn.Linear(256, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, num_classes)

    def forward(self, data):
        """

        :param data:
        :return:
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        lin3_x = self.lin3(x)
        x = F.log_softmax(lin3_x, dim=-1)
        return x


class Example02(Experiments):
    """
    """

    def __init__(self, epochs, batch_size, midi_dataset):
        """
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
        #
        self.data_loader = DataLoader(self._dataset, batch_size=self._batch_size, shuffle=True)
        self.model = ExampleOneModel(5, dataset.num_classes)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005)

    def train_epoch(self):
        self.model.train()
        epoch_loss = 0
        loss_all = 0

        for i, batch in enumerate(self.data_loader):
            batch.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(batch)

            train_mask = batch.train_mask
            y_train = batch.y[train_mask]
            print(f" {y_train.shape}")
            out_train = out[train_mask]

            loss = F.nll_loss(out_train, y_train)
            print(f"batch loss: {i} {loss}")
            loss.backward()
            self.optimizer.step()
            loss_all += batch.num_graphs * loss.item()
            epoch_loss += loss.item()

        loss_avg = loss_all / len(self.data_loader.dataset)
        return loss_avg

    def train(self):
        """
        :return:
        """
        test_acc = 0
        for e in range(1, self._epochs):
            epoch_loss = self.train_epoch()
            test_acc = self.test()
            print(f"Epoch: {e}, Loss: {epoch_loss:.5f}, Train Acc: {test_acc:.5f}")

    def test(self):
        """
        :return:
        """
        self.model.eval()
        total = 0
        accuracy = 0.0
        for batch in self.data_loader:
            test_mask = batch.test_mask
            data = batch.to(device)
            pred = self.model(data)

            pred_masked = pred[test_mask]
            y_masked = batch.y[test_mask]

            # y_masked = y_masked.unsqueeze(0)  # get to (batch_size, num_class)
            # pred_masked = pred_masked.unsqueeze(0)  # get to (batch_size, num_class)
            pred_class_idx = torch.argmax(pred_masked, dim=1)
            # print(f"Pred idx {pred_class_idx.shape}")
            # print(f"y_masked shape {y_masked.shape}")
            # Pred idx torch.Size([1, 48])
            # y_masked shape torch.Size([1, 10])

            correct = torch.eq(pred_class_idx, y_masked)
            # print("Correct", correct)

            accuracy += torch.mean(correct.float())
            total += y_masked.shape[0]
        return accuracy / total


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1000)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = T.Compose([
        # T.NormalizeFeatures(),
        T.ToDevice(device),
        # T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
        #                   split_labels=True, add_negative_train_samples=False)
    ])

    dataset = MidiDataset(root="./data")

    print(dir(dataset))
    print(dataset.num_features)
    print(f"Dataset size:       {len(dataset)}")
    print("Number of classes", dataset.total_num_classes)
    print("Number of feature", dataset.num_features)
    print(f"x shape:           {dataset[0].x.shape}")
    print(f"y shape:           {dataset[0].y.shape}")
    print(f"Label shape:       {dataset[0].label.shape}")
    print(f"number of classes: {dataset.num_classes}")

    #
    example = Example02(args.epochs, args.batch_size, dataset)
    example.train()
