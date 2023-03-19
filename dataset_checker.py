"""
Read dataset

In the context of graph completion, y typically represents the ground truth labels or features for all nodes in the graph (both the observed and missing ones). However, during training, we only use the observed nodes as input and try to predict the missing nodes based on their neighboring observed nodes. Once the missing nodes are predicted, we can evaluate the model's performance by comparing its predictions to the ground truth labels or features stored in y.




Author
Mus spyroot@gmail.com
    mbayramo@stanford.edu
"""

import torch
import argparse

from torch_geometric import nn

from neural_graph_composer.midi_dataset import MidiDataset
from torch_geometric.data import DataLoader

from neural_graph_composer.model import ComposerGCN
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()
    dataset = MidiDataset(root="./data")
    print("Dataset size", len(dataset))

    # input_dim = dataset.num_node_features
    hidden_dim = dataset.hidden_channels
    # output_dim = dataset.num_classes
    epochs = 1

    print(dataset[0].x.shape)
    print(dataset[0].y.shape)

    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    print(dataset[0])
    print("x", dataset[0].x)
    print("y", dataset[0].y)
    print("label", dataset[0].label)
    print("label", dataset[0].label.shape)

    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    model = ComposerGCN((8, 5), dataset.hidden_channels, 8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    #
    # # out = model(pyg_data, pyg_data.edge_attr)  # edge_weight provided

    for epoch in range(epochs):
        model.train()
        loss_accum = 0
        for batch in loader:
            optimizer.zero_grad()
            # # x = batch.x  # node feature matrix
            # edge_index = batch.edge_index  # graph connectivity in COO format
            print(f"Batch {batch.x.shape} {batch.edge_index.shape}")
            out = model.forward(batch)
            y = batch.y  # target labels (if available)
            print(y.shape)
            loss = loss_function(out, y)
            mask = batch.train_mask  # node mask
            out = model.forward(x, edge_index)
            loss = loss_function(out[mask], y[mask])
            loss.backward()
            optimizer.step()
            loss_accum += loss.item() * dataset[0].num_graphs
        epoch_loss = loss_accum / len(loader.dataset)
        print(f"Epoch {epoch} - Loss: {epoch_loss:.4f}")
