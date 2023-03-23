"""
Read dataset

Author
Mus spyroot@gmail.com
    mbayramo@stanford.edu
"""

import argparse

import torch_geometric.transforms as T
from neural_graph_composer.midi_dataset import MidiDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GINConv, global_add_pool
import torch
from torch.nn import Linear as Lin
from torch.nn import ReLU
from torch.nn import Sequential as Seq
from torch_geometric.utils import scatter
from torch_geometric.nn import SuperGATConv
from torch_geometric.nn import GraphConv, TopKPooling

#
#
# def loss_function(output, target):
#     """Compute cross-entropy loss between predicted and target labels.
#     """
#     criterion = nn.CrossEntropyLoss()
#     return criterion(output, target)

# def collate_fn(batch):
#     batch = [d for d in batch if d is not None]
#     if len(batch) == 0:
#         return None
#     return Batch.from_data_list(batch)
#
# train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=1, collate_fn=collate_fn)
#
#
# for i in range(len(train_dataset)):
#     train_dataset[i] = mask_nodes(train_dataset[i])
#
# def test(model, data):
#     model.eval()
#     with torch.no_grad():
#         out = model(data.x_masked, data.edge_index)
#         pred = out[data.mask]
#     return pred
#
# for data in test_dataset:
#     data = mask_nodes(data)
#     pred = test(model, data)
#     print(f"Predicted values: {pred}")
# def mask_nodes(data):
#     mask = torch.zeros(data.num_nodes, dtype=torch.bool)
#     mask[:int(0.5*data.num_nodes)] = True  # mask out first half of nodes
#     data.mask = mask
#     data.x_masked = data.x.clone()
#     data.x_masked[mask] = 0
#     return data

import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, num_feature=5, num_classes=8):
        super().__init__()
        self.conv1 = GCNConv(num_feature, 5)
        self.conv2 = GCNConv(5, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        print(f"forward shape {x.shape} {edge_index.shape}")
        x = self.conv1(x, edge_index)
        print(f"first conv {x.shape} {edge_index.shape}")
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        print(f"last x {x.shape} {edge_index.shape}")

        return F.log_softmax(x, dim=1)


def test_transform(device):
    """

    :param device:
    :return:
    """
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                          split_labels=True, add_negative_train_samples=False),
    ])
    return transform


class HandleNodeAttention(object):
    def __call__(self, data):
        data.attn = torch.softmax(data.x[:, 0], dim=0)
        data.x = data.x[:, 1:]
        return data


class ModelTokPoolingTest(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.conv1 = GINConv(Seq(Lin(in_channels, 64), ReLU(), Lin(64, 64)))
        self.pool1 = TopKPooling(in_channels, min_score=0.05)
        self.conv2 = GINConv(Seq(Lin(64, 64), ReLU(), Lin(64, 64)))
        self.lin = torch.nn.Linear(64, 1)

    def forward(self, data):
        """

        :param data:
        :return:
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # relu
        out = F.relu(self.conv1(x, edge_index))
        # pool
        out, edge_index, _, batch, perm, score = self.pool1(
            out, edge_index, None, batch, attn=x
        )
        ratio = out.size(0) / x.size(0)
        out = F.relu(self.conv2(out, edge_index))
        out = global_add_pool(out, batch)
        out = self.lin(out).view(-1)

        attn_loss = F.kl_div(
            torch.log(score + 1e-14), data.attn[perm],
            reduction='none')

        attn_loss = scatter(attn_loss, batch, reduce='mean')
        return out, attn_loss, ratio


def single_Layer_pass_test():
    """Test that we can take a data
    and pass and all shapes correct.
    :return:
    """
    dataset = MidiDataset(root="./data")
    print("Dataset size", len(dataset))

    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    sampled_data = next(iter(loader))
    num_feature = sampled_data.x.shape[1]
    print(f"Number of features {num_feature}")
    conv = GINConv(Seq(Lin(num_feature, 64), ReLU(), Lin(64, 64)))

    x = sampled_data.x
    edge_index = sampled_data.edge_index
    print(f"x shape {x.shape}")
    print(f"edge index shape {edge_index.shape}")
    forward_pass = conv.forward(x, edge_index)
    print("Forward pass return", forward_pass.shape)

    #
    conv = SuperGATConv(num_feature, 8, heads=8,
                        dropout=0.6, attention_type='MX',
                        edge_sample_ratio=0.8, is_undirected=False)
    super_gat_out = conv.forward(x, edge_index)
    print(f"super gat conv {super_gat_out.shape}")

    graph_conv = GraphConv(num_feature, 128)
    graph_conv_out = graph_conv.forward(x, edge_index)
    print(f"graph_conv_out {graph_conv_out.shape}")


def vanila_gnn_test():
    """
    :return:
    """
    # input_dim = dataset.num_node_features
    hidden_dim = dataset.__hidden_channels
    # output_dim = dataset.num_classes
    epochs = 1
    print(dataset[0].x.shape)
    print(dataset[0].y.shape)

    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    print(dataset[0])
    print("label", dataset[0].label)
    print("label shape", dataset[0].label.shape)
    print(f"x shape {dataset[0].x.shape}")  # (num_nodes, num_feature)
    print(f"y shape {dataset[0].y.shape}")  # (num_nodes)
    print(f"edge {dataset[0].edge_index.shape}")  # (2, num_nodes)

    # loader = DataLoader(dataset, batch_size=1, shuffle=True)
    model = GCN(8, 5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # out = model(pyg_data, pyg_data.edge_attr)  # edge_weight provided

    sampled_data = next(iter(loader))

    for epoch in range(epochs):
        model.train()
        loss_accum = 0
        for batch in loader:
            optimizer.zero_grad()
            feature = sampled_data.x.shape[1]
            out = model(batch)
            print(f"got back {out.shape}")
            break

            # # # x = batch.x  # node feature matrix
            # # edge_index = batch.edge_index  # graph connectivity in COO format
            # print(f"Batch {batch.x.shape} {batch.edge_index.shape}")
            # out = model.forward(batch)
            # y = batch.y  # target labels (if available)
            # print(y.shape)
            # loss = loss_function(out, y)
            # mask = batch.train_mask  # node mask
            # out = model.forward(x, edge_index)
            # loss = loss_function(out[mask], y[mask])
            # loss.backward()
            # optimizer.step()
            # loss_accum += loss.item() * dataset[0].num_graphs
        epoch_loss = loss_accum / len(loader.dataset)
        print(f"Epoch {epoch} - Loss: {epoch_loss:.4f}")


def print_dataset_details(ds):
    """

    :return:
    """
    print(ds[0])
    print("label", ds[0].label)
    print("label shape", ds[0].label.shape)
    print(f"x shape {ds[0].x.shape}")  # (num_nodes, num_feature)
    print(f"y shape {ds[0].y.shape}")  # (num_nodes)
    print(f"edge {ds[0].edge_index.shape}")  # (2, num_nodes)


def gnn_basic_test():
    """
    :return:
    """
    epochs = 1
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    sampled_data = next(iter(loader))
    num_feature = sampled_data.x.shape[1]
    model = GCN(8, num_feature)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # out = model(pyg_data, pyg_data.edge_attr)  # edge_weight provided

    for epoch in range(epochs):
        model.train()
        loss_accum = 0
        for batch in loader:
            optimizer.zero_grad()

            out = model(batch)
            print(f"got back {out.shape}")
            break

            # # # x = batch.x  # node feature matrix
            # # edge_index = batch.edge_index  # graph connectivity in COO format
            # print(f"Batch {batch.x.shape} {batch.edge_index.shape}")
            # out = model.forward(batch)
            # y = batch.y  # target labels (if available)
            # print(y.shape)
            # loss = loss_function(out, y)
            # mask = batch.train_mask  # node mask
            # out = model.forward(x, edge_index)
            # loss = loss_function(out[mask], y[mask])
            # loss.backward()
            # optimizer.step()
            # loss_accum += loss.item() * dataset[0].num_graphs
        epoch_loss = loss_accum / len(loader.dataset)
        print(f"Epoch {epoch} - Loss: {epoch_loss:.4f}")


def topk_test():
    """
    :return:
    """
    epochs = 100

    dataset = MidiDataset(root="./data", transform=HandleNodeAttention())
    print("Dataset size", len(dataset))

    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    sampled_data = next(iter(loader))
    num_feature = sampled_data.x.shape[1]
    model = ModelTokPoolingTest(num_feature)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(epochs):
        for batch in loader:
            model.train()
            optimizer.zero_grad()
            out, attn_loss, _ = model.forward(batch)
            loss = ((out - sampled_data.y).pow(2) + 100 * attn_loss).mean()
            loss.backward()
            optimizer.step()
            print(f" Loss {loss} {attn_loss}")
        # print(f"Epoch {epoch} - Loss: {epoch_loss:.4f}")


#
if __name__ == '__main__':
    """
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()
    dataset = MidiDataset(root="./data")
    print("Dataset size", len(dataset))
    print(dataset[0].label)
    print(dataset[0].y)
    print("Number of classes", dataset.num_classes)
    # single_Layer_pass_test()

    topk_test()
    # gnn_basic_test()
    # gin_test()
