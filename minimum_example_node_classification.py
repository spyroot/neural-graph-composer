import argparse
import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from sklearn.metrics import f1_score
from torch.nn import Linear
from torch_geometric.loader import DataLoader
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GATConv, GCN2Conv
from torch_geometric.transforms import RandomNodeSplit

from neural_graph_composer.midi_dataset import MidiDataset
from neural_graph_composer.transforms import AddDegreeTransform
from sklearn.metrics import f1_score

parser = argparse.ArgumentParser()
parser.add_argument('--hidden_channels', type=int, default=128)
parser.add_argument('--heads', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--wandb', action='store_true', help='Track experiment')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# init_wandb(name=f'GAT-{args.dataset}', heads=args.heads, epochs=args.epochs,
#            hidden_channels=args.hidden_channels, lr=args.lr, device=device)

transform = T.Compose([
    AddDegreeTransform(),
    T.NormalizeFeatures(),
])

pre_transform = T.Compose([T.GCNNorm()])

dataset = MidiDataset(root="./data",
                      transform=transform,
                      pre_transform=pre_transform,
                      per_instrument_graph=False,
                      tolerance=0.5,
                      include_velocity=True)

transform.transforms.insert(0, RandomNodeSplit(
    num_splits=2, num_train_per_class=100, num_val=0.1, num_test=0.3))

train_loader = DataLoader(
    dataset[99:120], batch_size=8, shuffle=True)

test_loader = DataLoader(
    dataset[99:120], batch_size=8, shuffle=False)


class NeuralGraphComposer(torch.nn.Module):
    """

    """

    def __init__(self, hidden_channels, num_layers, alpha, theta,
                 shared_weights=True, dropout=0.0):
        super().__init__()

        print(f"Create {dataset.num_features} {dataset.num_classes}")
        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(dataset.num_features, hidden_channels))
        self.lins.append(Linear(hidden_channels, dataset.num_classes))

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                GCN2Conv(hidden_channels, alpha, theta, layer + 1,
                         shared_weights, normalize=False))

        self.dropout = dropout

    def forward(self, x, adj_t):
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()

        for conv in self.convs:
            h = F.dropout(x, self.dropout, training=self.training)
            h = conv(h, x_0, adj_t)
            x = h + x
            x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[1](x)

        return x


model = NeuralGraphComposer(hidden_channels=2048,
                            num_layers=9, alpha=0.5, theta=1.0,
                            shared_weights=False, dropout=0.2).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)


def train(data):
    """train loop with cross entropy
    """
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    losses = []
    for mask_id in range(data.train_mask.shape[1]):
        mask = data.train_mask[:, mask_id]
        loss = F.cross_entropy(out[mask], data.y[mask])
        loss.backward(retain_graph=True)
        losses.append(loss.item())
    optimizer.step()
    return sum(losses) / len(losses)


@torch.no_grad()
def test(data):
    """ Test on 3 set test , train validation. All stats aggregated including F1"""
    model.eval()
    pred = model(data.x, data.edge_index).argmax(dim=-1)
    make_accs = [[0.0] * 3 for _ in range(data.train_mask.shape[1])]

    f1s = [[0.0] * 3 for _ in range(data.train_mask.shape[1])]

    for mask_id in range(data.train_mask.shape[1]):
        for m, mask in enumerate([data.train_mask[:, mask_id], data.val_mask[:, mask_id], data.test_mask[:, mask_id]]):
            mask_sum = int(mask.sum())
            mask_pred = pred[mask]
            mask_y = data.y[mask]
            make_accs[mask_id][m] += int((mask_pred == mask_y).sum()) / mask_sum
            f1s[mask_id][m] += f1_score(mask_y.cpu().numpy(), mask_pred.cpu().numpy(), average='micro')

    return [sum(column) / len(make_accs)
            for column in zip(*make_accs)], [sum(column) / len(f1s)
                                             for column in zip(*f1s)]


def main():
    best_val_acc = final_test_acc = 0
    for epoch in range(1, args.epochs + 1):
        batch_loss = 0.0
        train_num_batches = len(train_loader)
        for i, b in enumerate(train_loader):
            b.to(device)
            loss = train(b)
            batch_loss += loss

        batch_train_acc, batch_val_acc, batch_test_acc = 0.0, 0.0, 0.0
        train_f1, val_f1, test_f1 = 0.0, 0.0, 0.0

        num_batches = len(test_loader)
        for i, b in enumerate(test_loader):
            b.to(device)
            accuracies, f1_scores = test(b)
            train_acc, val_acc, test_acc = accuracies
            train_f1, val_f1, test_f1 = f1_scores
            batch_train_acc += train_acc
            batch_val_acc += val_acc
            batch_test_acc += test_acc
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = test_acc

        batch_loss /= train_num_batches
        batch_train_acc /= num_batches
        batch_val_acc /= num_batches
        batch_test_acc /= num_batches
        print(
            f"Epoch: {epoch}, "
            f"Loss: {batch_loss:.5f}, "
            f"Train acc: {batch_train_acc:.5f}, "
            f"Val acc: {batch_val_acc:.5f}, "
            f"Test acc: {batch_test_acc:.5f} "
            f"Train F1: {train_f1:.5f} ",
            f"Val  F1: {val_f1:.5f} ",
            f"Test F1: {test_f1:.5f} ")


if __name__ == '__main__':
    main()
