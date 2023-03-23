"""
Gin like with MLP layer
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, GINConv, global_add_pool
from example_shared import Experiments
from neural_graph_composer.midi_dataset import MidiDataset


class GinLike(torch.nn.Module):
    """
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        """
        :param in_channels:
        :param hidden_channels:
        :param out_channels:
        :param num_layers:
        """
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = MLP([in_channels, hidden_channels, hidden_channels])
            self.convs.append(GINConv(nn=mlp, train_eps=False))
            in_channels = hidden_channels

        self.mlp = MLP([hidden_channels,
                        hidden_channels,
                        out_channels],
                       norm=None, dropout=0.5)

    def forward(self, x, edge_index, batch):
        """
        :param x:
        :param edge_index:
        :param batch:
        :return:
        """
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        x = global_add_pool(x, batch)

        return self.mlp(x)


class Example01(Experiments):
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
        self.model = GinLike(5, 64, dataset.num_classes, 4)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=5e-4)

    def train_epoch(self):
        """
        :return:
        """
        self.model.train()
        epoch_loss = 0
        all_losses = []

        for i, batch in enumerate(self.data_loader):
            batch.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(batch.x, batch.edge_index, batch.batch)

            train_mask = batch.train_mask
            y_train = batch.y[train_mask]

            node_idx = torch.arange(out.shape[0]).to(self.device)
            out_sum = torch.zeros(
                (batch.num_nodes, out.shape[1]), dtype=torch.float).to(self.device)
            out_sum.scatter_(0, node_idx.unsqueeze(1).expand(-1, out.shape[1]), out)
            out_train = out_sum[train_mask]
            loss = F.nll_loss(out_train, y_train)
            loss.backward()
            self.optimizer.step()

            all_losses.append(loss.item())
            epoch_loss += loss.item()

        loss_avg = sum(all_losses) / len(all_losses)
        return loss_avg

    def train(self):
        """
        :return:
        """
        test_acc = 0
        for e in range(1, self._epochs):
            epoch_loss = self.train_epoch()
            accuracy, loss_avg = self.test()
            print(f"Epoch: {e}, Loss: {epoch_loss:.5f}, Test Acc: {accuracy:.5f} Test loss: {loss_avg:.5f}")

    def test(self):
        """
        :return:
        """
        self.model.eval()
        total = 0
        correct = []
        losses = []
        for batch in self.data_loader:
            test_mask = batch.test_mask
            data = batch.to(device)
            out = self.model(data.x, data.edge_index, batch.batch)
            node_idx = torch.arange(out.shape[0]).to(self.device)
            out_sum = torch.zeros((batch.num_nodes, out.shape[1]), dtype=torch.float).to(self.device)
            out_sum.scatter_(0, node_idx.unsqueeze(1).expand(-1, out.shape[1]), out)
            pred_masked = out_sum[test_mask]

            y_masked = batch.y[test_mask]
            pred_class_idx = torch.argmax(pred_masked, dim=1)
            # correct += torch.sum(torch.eq(pred_class_idx, y_masked)).item()
            correct.append(torch.eq(pred_class_idx, y_masked).cpu().numpy())

            loss = F.nll_loss(pred_masked, y_masked)
            losses.append(loss.item())
            total += y_masked.shape[0]

        accuracy = np.mean(np.concatenate(correct))
        loss_avg = np.mean(losses)
        return accuracy, loss_avg


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10000)
    args = parser.parse_args()

    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        # T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
        #                   split_labels=True, add_negative_train_samples=False)
    ])

    dataset = MidiDataset(root="./data", transform=transform, per_instrument_graph=False)
    print(f"Dataset size:       {len(dataset)}")
    print("Number of classes", dataset.total_num_classes)
    print("Number of feature", dataset.num_features)
    print(f"x shape:           {dataset[0].x.shape}")
    print(f"y shape:           {dataset[0].y.shape}")
    print(f"Label shape:       {dataset[0].label.shape}")
    print(f"number of classes: {dataset.num_classes}")

    example = Example01(args.epochs, args.batch_size, dataset)
    example.train()
