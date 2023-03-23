"""

Here we trying
In the context of graph completion, y typically represents the
ground truth labels or features for all nodes in the graph
(both the observed and missing ones). However, during training,
we only use the observed nodes as input and try to predict the missing nodes based on
 their neighboring observed nodes. Once the missing nodes are predicted,
 we can evaluate the model's performance by comparing its predictions to the
  ground truth labels or features stored in y.
"""
import heapq
from typing import Optional

import argparse
import heapq
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv
from torch_geometric.nn import GCNConv

from example_shared import Experiments
from neural_graph_composer.midi_dataset import MidiDataset


class GCN(torch.nn.Module):
    """
        Two-layer Graph Convolutional Network (GCN) that maps node features to node embeddings.
    """

    def __init__(self, num_features: int, hidden_channels: int):
        """

        :param num_features: (int): The number of input features for each node.
        :param hidden_channels: (int): The number of output channels in the GCN layer.
        """
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


class GCN3(torch.nn.Module):
    """
    Three-layer Graph Convolutional Network (GCN) that maps node features to class predictions.
    """

    def __init__(self, num_feature: int,
                 hidden_channels: int,
                 num_classes: int,
                 dropout: float = 0.2):
        """
        :param num_feature: Number of input features per node.
        :param hidden_channels: Size of the output feature space
                                of the first and second convolutional layers.
        :param num_classes: Number of classes to predict.
        :param dropout: Dropout probability.
        """
        super(GCN3, self).__init__()
        self.conv1 = GCNConv(num_feature, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, num_classes)
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """ Perform a forward pass on the GCN3 model to get embeddings.
        :param x: Input feature tensor of shape (num_nodes, num_features)
        :param edge_index: Edge tensor of shape (2, num_edges)
        :return: Output tensor of shape (num_nodes, num_classes)
        """
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)


class GCNEncoder(torch.nn.Module):
    """
        Two-layer GCN used for graph node encoding.
    """

    def __init__(self, in_channels, out_channels):
        """

        :param in_channels:
        :param out_channels:
        """
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        """
        :param x:
        :param edge_index:
        :return:
        """
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class Decoder(nn.Module):
    """Decoder module that maps latent representation to output node features.
    """

    def __init__(self, hidden_size: int,
                 output_size: int,
                 lstm_output_size: int = 32,
                 dropout_prob: float = 0.5):
        """
        :param hidden_size: The size of the hidden layer in the Decoder.
        :param output_size: The size of the output layer in the Decoder.
        :param lstm_output_size: The output size of the LSTM layer in case of LSTM-based Decoder.
        """
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(hidden_size, output_size)
        self.fc2 = nn.Linear(lstm_output_size, hidden_size)
        self.fc3 = nn.Linear(2, hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        """Perform forward pass through the Decoder.
        :param x:
        :return:
        """
        # handle lstm
        # print(f" Forward {x.shape}, {self.fc2.in_features}")
        if x.shape[-1] == self.fc2.in_features:
            print("############## LSTM case")
            return self.forward_lstm(x)
        # handle gnn
        elif x.shape[-1] == self.fc3.in_features:
            return self.forward_generated(x)
        else:
            return self.forward_gcn(x)

    def forward_gcn(self, x) -> Tensor:
        """Perform forward pass through the Decoder in case of a GNN-based Decoder.
        :param x: The input tensor to the Decoder.
        :return:
        """
        # print("gnc case")
        x = self.fc1(x)
        return x

    def forward_lstm(self, x) -> Tensor:
        """ Perform forward pass through the Decoder in case of a LSTM-based Decoder.
        :param x: The input tensor to the Decoder.
        :return:
        """
        x = F.relu(self.fc2(x))
        x = self.fc1(x)
        return x

    def forward_generated(self, x):
        """
        :param x:
        :return:
        """
        x = F.relu(self.fc3(x))
        x = self.fc1(x)
        return x


class GraphLSTM(nn.Module):
    """
    """

    def __init__(self, input_size, hidden_size, num_layers):
        """
        :param input_size:
        :param hidden_size:
        :param num_layers:
        """
        super(GraphLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        return out, hidden


class GAT(torch.nn.Module):
    """
    """

    def __init__(self, num_features, hidden_channels, dropout: float = 0.2):
        """

        :param num_features:
        :param hidden_channels:
        :param dropout:
        """
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_features,
                             hidden_channels,
                             heads=8,
                             dropout=dropout)
        self.conv2 = GATConv(
            hidden_channels * 8, hidden_channels,
            dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        """

        :param x:
        :param edge_index:
        :return:
        """
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GraphGenerationModel(Experiments):
    """
    """

    def __init__(self,
                 midi_dataset: MidiDataset,
                 epochs: Optional[int] = 100,
                 batch_size: Optional[int] = 32,
                 embeddings_lr: Optional[float] = 0.01,
                 lstm_lr: Optional[float] = 0.01,
                 model_type: Optional[str] = "GCN3",
                 gcn_hidden_dim: Optional[int] = 32):
        """
        :param epochs:
        :param batch_size:
        :param midi_dataset:
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
        self.data_loader = DataLoader(
            self._dataset, batch_size=self._batch_size, shuffle=True
        )
        self.model = None

        self.hidden_channels = midi_dataset.num_classes
        self.input_size = self.hidden_channels * 2
        self.hidden_size = midi_dataset.num_classes * 2
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_layers = 2

        print(f"Dataset:            {midi_dataset}")
        print(f"Epochs:             {epochs}")
        print(f"Batch size:         {batch_size}")
        print(f"Embeddings LR:      {embeddings_lr}")
        print(f"LSTM LR:           {lstm_lr}")
        print(f"Model type:         {model_type}")
        print(f"GCN hidden dim:     {gcn_hidden_dim}")
        # print(f"Feature dim:        {self._feature_dim}")
        # print(f"Number of classes:  {self._num_classes}")
        # print(f"Hidden dim:         {self._hidden_dim}")

        if model_type == "GCN3":
            print(f"Creating GCN3 {midi_dataset.num_features} {midi_dataset.num_classes}")
            self.gcn_model = GCN3(
                midi_dataset.num_features, gcn_hidden_dim, midi_dataset.num_classes).to(self.device)
        elif model_type == "GAT":
            print(f"Creating GAT {midi_dataset.num_features} {midi_dataset.num_classes}")
            self.gcn_model = GAT(
                midi_dataset.num_features, midi_dataset.num_classes).to(self.device)
            self._is_gat = True
        else:
            raise ValueError("unk")
            # self.model = GIN(
            #     self._feature_dim, self._hidden_dim, self._num_classes)
            # self._is_gin = True

        # # self.gcn_model = GCN(midi_dataset.num_features, midi_dataset.num_classes).to(self.device)
        # self.gcn_model = GAT(midi_dataset.num_features, midi_dataset.num_classes).to(self.device)

        self.embeddings_lr: float = embeddings_lr
        self.decoder = Decoder(midi_dataset.num_classes, midi_dataset.num_classes,
                               lstm_output_size=self.hidden_size).to(self.device)
        self.lstm_model = GraphLSTM(self.input_size, self.hidden_size, self.num_layers).to(self.device)

        self.optimizer = torch.optim.Adam(
            [
                {'params': self.gcn_model.parameters()},
                {'params': self.lstm_model.parameters()}
            ],
            lr=0.01
        )

    def train_gcn(self, grap_data: Data,
                  epochs: int = 10,
                  learning_rate: float = 0.01) -> tuple[float, float, float]:
        """
        :param grap_data: A PyTorch Geometric `Data` object containing the graph data.
        :param epochs: The number of epochs to train for.
        :param learning_rate: The learning rate to use for optimization.
        :return:
        """
        loss_fn = torch.nn.CrossEntropyLoss()

        gcn_ground_truth_labels = grap_data.y
        correct_gcn_predictions = 0.0
        total_gcn_predictions = 0.0
        gcn_loss = 0.0

        gcn_tp = np.zeros(self.num_classes)
        gcn_fp = np.zeros(self.num_classes)
        gcn_fn = np.zeros(self.num_classes)

        for epoch in range(epochs):
            node_embeddings = self.gcn_model(grap_data.x, grap_data.edge_index)
            output = self.decoder(node_embeddings)
            loss_gcn = loss_fn(output, grap_data.y)

            gcn_loss += loss_gcn.item()
            gcn_predicted_node_labels = torch.argmax(node_embeddings, dim=1)
            correct_gcn_predictions += (gcn_predicted_node_labels == gcn_ground_truth_labels).sum().item()
            total_gcn_predictions += gcn_ground_truth_labels.size(0)

            for i in range(self.num_classes):
                tp = ((gcn_predicted_node_labels == i) & (gcn_ground_truth_labels == i)).sum().item()
                fp = ((gcn_predicted_node_labels == i) & (gcn_ground_truth_labels != i)).sum().item()
                fn = ((gcn_predicted_node_labels != i) & (gcn_ground_truth_labels == i)).sum().item()
                gcn_tp[i] += tp
                gcn_fp[i] += fp
                gcn_fn[i] += fn

            if epoch % 20 == 0:
                print(f"epoch {epoch} gcn avg loss: {round(gcn_loss / (epoch + 1), 3)} "
                      f"avg accuracy: {round(correct_gcn_predictions / total_gcn_predictions, 3)} "
                      f"total gcn: {total_gcn_predictions}")

        gcn_precision = gcn_tp / (gcn_tp + gcn_fp + 1e-8)
        gcn_recall = gcn_tp / (gcn_tp + gcn_fn + 1e-8)
        gcn_f1_score = 2 * gcn_precision * gcn_recall / (gcn_precision + gcn_recall + 1e-8)

        avg_gcn_loss = gcn_loss / epochs
        avg_gcn_accuracy = correct_gcn_predictions / total_gcn_predictions

        avg_gcn_precision = np.mean(gcn_precision)
        avg_gcn_recall = np.mean(gcn_recall)
        avg_gcn_f1_score = np.mean(gcn_f1_score)

        print(f"avg loss for 100 epoch: {avg_gcn_loss:.4f}, "
              f"accuracy: {avg_gcn_accuracy:.4f}, "
              f"precision: {avg_gcn_precision:.4f}, "
              f"recall: {avg_gcn_recall:.4f}, "
              f"F1-score: {avg_gcn_f1_score:.4f} ")
        return gcn_loss, correct_gcn_predictions, total_gcn_predictions
        # print(f"Epoch {epoch}, Loss: {loss_gcn.item():.4f} shape {output.shape} {self._dataset.num_classes}")

    def train_lstm(self, seq_data: List[torch.Tensor], epochs: int = 10, learning_rate: float = 0.01):
        """Train the LSTM model on a list of edge embeddings.

        During training, we consider a true positive to be a correct prediction
         of the next edge embedding in the sequence,
         and a false positive to be an incorrect prediction.

         The ground truth labels are the target edge embeddings that
         come immediately after each input edge embedding in the sequence.

         seq_data: A list of PyTorch tensors, where each tensor represents a sequence
                    of edge and node embeddings. Each tensor has shape (1, seq_len, num_features),
                    where seq_len is the length of the sequence and
                     num_features is the number of features in each edge or node embedding.

                     The tensor is constructed by concatenating the edge and
                     node embeddings for each edge in the sequence.

        :param seq_data: A list of edge embeddings, where each edge
                          embedding has shape (num_edges, 2 * hidden_size).
        :param epochs: The number of epochs to train the LSTM for.
        :param learning_rate: The learning rate to use for optimization.
        :return:
        """

        correct_lstm_predictions = 0
        total_lstm_predictions = 0
        lstm_loss = 0

        true_positives = 0
        false_positives = 0
        false_negatives = 0

        #
        for epoch in range(epochs):
            for seq in seq_data:
                seq = seq.view(1, -1, self.input_size)
                lstm_input = seq[:, :-1, :]
                lstm_target = seq[:, 1:, :]
                hidden = None
                lstm_output, hidden = self.lstm_model(lstm_input, hidden)
                lstm_ground_truth_labels = seq.clone()
                loss_lstm = F.mse_loss(lstm_output, lstm_target)
                lstm_loss += loss_lstm.item()

                lstm_predicted_labels = torch.argmax(lstm_output, dim=1)
                correct_lstm_predictions += (lstm_predicted_labels == lstm_ground_truth_labels).sum().item()
                total_lstm_predictions += lstm_ground_truth_labels.size(0)

                # calculate precision, recall, and F1-score
                true_positives += ((lstm_predicted_labels == lstm_ground_truth_labels)
                                   & (lstm_ground_truth_labels == 1)).sum().item()
                false_positives += ((lstm_predicted_labels != lstm_ground_truth_labels)
                                    & (lstm_predicted_labels == 1)).sum().item()
                false_negatives += ((lstm_predicted_labels != lstm_ground_truth_labels)
                                    & (lstm_ground_truth_labels == 0)).sum().item()

            if epoch % 20 == 0:
                avg_loss = lstm_loss / len(seq_data)
                avg_accuracy = correct_lstm_predictions / total_lstm_predictions
                precision = true_positives / (
                        true_positives + false_positives) if true_positives + false_positives != 0 else 0
                recall = true_positives / (
                        true_positives + false_negatives) if true_positives + false_negatives != 0 else 0

                f1_score = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
                print(f"Epoch: {epoch + 1}, Avg LSTM loss: {avg_loss:.4f}, "
                      f"Avg Accuracy: {avg_accuracy:.4f}, "
                      f"Precision: {precision:.4f}, "
                      f"Recall: {recall:.4f}, "
                      f"F1-score: {f1_score:.4f}")

        precision = true_positives / (true_positives + false_positives) if true_positives + false_positives != 0 else 0
        recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives != 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0

        avg_lstm_loss = lstm_loss / len(seq_data)
        avg_lstm_accuracy = correct_lstm_predictions / total_lstm_predictions

        print(f"LSTM avg loss {avg_lstm_loss:.4f} "
              f"avg accuracy  {avg_lstm_accuracy:.4f} "
              f"total prediction {total_lstm_predictions}, "
              f"Recall {recall:.4f}, "
              f"F1 {f1_score:.4f}")

        return lstm_loss, correct_lstm_predictions, total_lstm_predictions

    def generate_new_nodes_edges(self, edge_embeddings: torch.Tensor, num_new_nodes: int = 1) -> torch.Tensor:
        """ Generates new node embeddings and corresponding edges using the trained LSTM model.
        :param edge_embeddings:  Edge embeddings of the test graph.
        :param num_new_nodes: Number of new nodes to generate.
        :return: Embeddings of the newly generated nodes and their corresponding edges.
        """
        lstm_input = edge_embeddings.view(1, -1, self.input_size)
        hidden = None
        new_nodes_edges = []

        for _ in range(num_new_nodes):
            lstm_output, hidden = self.lstm_model(lstm_input, hidden)
            lstm_output = lstm_output[-1, -1, :]
            new_embedding = self.decoder(lstm_output)
            new_nodes_edges.append(new_embedding.detach())

        # convert the generated embeddings into hash values and edge weights
        new_nodes_edges = torch.stack(new_nodes_edges)
        return new_nodes_edges

    def get_graph_edge_embeddings(self, _subgraph_data: Data) -> torch.Tensor:
        """
        :param _subgraph_data:
        :return:
        """
        node_emb = self.gcn_model(_subgraph_data.x, _subgraph_data.edge_index)
        node_emb = torch.cat([
            node_emb[_subgraph_data.edge_index[0]],
            node_emb[_subgraph_data.edge_index[1]]],
            dim=1)
        return node_emb


def dijkstra_traversal2(graph, start_node):
    """Generates new node embeddings and
       corresponding edges using the trained LSTM model.
    """
    visited = set()
    pq = [(0, start_node)]
    node_sequence = []
    edge_sequence = []

    edge_index = graph.edge_index.view(2, -1)

    while pq:
        dist, node = heapq.heappop(pq)
        if node not in visited:
            visited.add(node)
            node_sequence.append(node)
            neighbors = edge_index[1][edge_index[0] == node].tolist()
            for neighbor in neighbors:
                edge_sequence.append((node, neighbor))
                if neighbor not in visited:
                    edge_indices = (edge_index[0] == node) & (edge_index[1] == neighbor)
                    if edge_indices.any():
                        # use the first matching edge index
                        edge_attr_index = edge_indices.nonzero(as_tuple=True)[0][0].item()
                        edge_weight = graph.edge_attr[edge_attr_index]
                        heapq.heappush(pq, (dist + edge_weight, neighbor))

    return node_sequence, edge_sequence


def compute_node_and_edge_embeddings(model, graph_data):
    """
    :param model:
    :param graph_data:
    :return:
    """
    node_embeddings = model.gcn_model(graph_data.x, graph_data.edge_index)
    edge_embeddings = torch.cat([
        node_embeddings[graph_data.edge_index[0]],
        node_embeddings[graph_data.edge_index[1]]
    ], dim=1)
    return node_embeddings, edge_embeddings


def map_labels_to_hashes(predicted_labels, hash_labels):
    mapped_labels = [hash_labels[i] for i in predicted_labels.tolist()]
    return mapped_labels


def main(args, midi_dataset):
    """
    :return:
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sequence_data = []

    graph_data_loader = DataLoader(
        midi_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    graph_generation_model = GraphGenerationModel(
        midi_dataset=midi_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        embeddings_lr=0.01,
        lstm_lr=0.01,
        model_type=args.gcn,
        gcn_hidden_dim=args.gcn_hidden_dim
    )

    for epoch in range(args.epochs):
        for i, grap_data in enumerate(graph_data_loader):
            node_sequence, edge_sequence = dijkstra_traversal2(grap_data, start_node=0)
            graph_generation_model.gcn_model.train()
            gcn_loss, correct_gcn_predictions, total_gcn_predictions = graph_generation_model.train_gcn(
                grap_data,
                epochs=args.epochs)

            #  node embeddings using the trained GCN model for midi graph
            node_embeddings = graph_generation_model.gcn_model(grap_data.x, grap_data.edge_index)
            #  concatenate src and target node embeddings for each edge
            edge_embeddings = torch.cat([node_embeddings[torch.tensor([src for src, _ in edge_sequence])],
                                         node_embeddings[torch.tensor([dst for _, dst in edge_sequence])]], dim=1)
            sequence_data.append(edge_embeddings.detach())

        # train LSTM
        # The node_embeddings tensor has the shape (3, 16)
        # which means it has 3 rows (corresponding to the 3 nodes in the input graph)
        # and 16 columns (representing the 16-dimensional embedding for each node).

        graph_generation_model.lstm_model.train()
        lstm_loss, correct_lstm_predictions, total_lstm_predictions = graph_generation_model.train_lstm(sequence_data)
        total_loss = gcn_loss + lstm_loss
        graph_generation_model.optimizer.backward()
        graph_generation_model.optimizer.step()
        graph_generation_model.optimizer.zero_grad()

        print(f"Total loss: {total_loss.item():.4f}")

    grap_data = grap_data.clone()
    # generate new nodes and edges
    new_node_embeddings = graph_generation_model.gcn_model(
        grap_data.x, grap_data.edge_index)

    edge_embeddings = torch.cat([node_embeddings[torch.tensor([src for src, _ in grap_data.edge_index.T])],
                                 node_embeddings[torch.tensor([dst for _, dst in grap_data.edge_index.T])]], dim=1)

    new_nodes_edges = graph_generation_model.generate_new_nodes_edges(edge_embeddings)
    decoded_node = graph_generation_model.decoder(new_nodes_edges)

    # new_nodes_edges = graph_generation_model.generate_new_nodes_edges(new_node_embeddings)
    print(f"node embeddings {new_node_embeddings.shape} {new_nodes_edges.shape}")
    predicted_labels = torch.argmax(decoded_node, dim=1)
    print(f"predicted_labels {predicted_labels}")
    mapped_labels = [dataset.hash_labels[i] for i in predicted_labels.tolist()]
    print(f"Mapped labels: {mapped_labels}")
    return decoded_node


if __name__ == '__main__':
    """
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--emb_lr', type=float, default=0.01)
    parser.add_argument('--lstm_lr', type=float, default=0.01)
    parser.add_argument('--gcn_hidden_dim', type=int, default=32)
    parser.add_argument('--graph_per_instrument', type=bool, default=False)
    parser.add_argument('--random_split', type=bool, default=False)
    parser.add_argument('--gcn', type=str, default='GCN3', choices=['GCN', 'GAT'])
    parser.add_argument('--num_val', type=float, default=0.05)
    parser.add_argument('--num_test', type=float, default=0.1)
    parser.add_argument('--add_negative_train_samples', type=bool, default=False)

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
    ])

    if args.random_split:
        transform = T.Compose([
            T.NormalizeFeatures(),
            T.ToDevice(device),
            T.RandomLinkSplit(
                num_val=args.num_val,
                num_test=args.num_test,
                is_undirected=True,
                split_labels=True,
                add_negative_train_samples=args.add_negative_train_samples)
        ])
    else:
        transform = T.Compose([
            T.NormalizeFeatures(),
            T.ToDevice(device),
        ])

    dataset = MidiDataset(
        root="./data",
        per_instrument_graph=args.graph_per_instrument,
        transform=transform
    )

    print(dataset.num_features)
    print(f"Dataset size:       {len(dataset)}")
    print("Number of classes", dataset.total_num_classes)
    print("Number of feature", dataset.num_features)
    print(f"x shape:           {dataset[0].x.shape}")
    print(f"y shape:           {dataset[0].y.shape}")
    print(f"Label shape:       {dataset[0].label.shape}")
    print(f"number of classes: {dataset.num_classes}")

    main(args, dataset)
