"""
This data set checker.

The Graph Auto-Encoder (GAE) model is then used to process the custom MidiDataset.
 It takes the graph structure of each MIDI composition and learns a latent
 representation of the chords and their relationships. The GAE can be used
 for various tasks, such as generating new MIDI compositions or predicting
 the next chords in a given sequence. The encoder and decoder parts of the GAE can be any combination of
the provided classes, such as GCNEncoder, VariationalGCNEncoder, LinearEncoder, and VariationalLinearEncoder.

Chord prediction: GAE can be used to predict the next chord in a sequence based on the learned graph structure.
By training the model on the relationships between chords in the MIDI dataset, it can generate new chord
 progressions that follow the learned patterns.

Music genre classification: GAE can be used for music genre classification by learning to
 differentiate between different styles and structures of music. By training the model on a
 labeled dataset of MIDI files, it can identify the characteristics that define various
 genres and classify new MIDI compositions accordingly.

Chord embeddings: GAE can be used to learn meaningful representations (embeddings) of chords.
These embeddings can then be used to compare and analyze chords, identify similarities
 and differences, and visualize the chord space.

Music generation: GAE can be utilized to generate new MIDI compositions.
By sampling from the latent space of chords and their relationships, GAE can create new chord
progressions and, consequently, new musical pieces that exhibit similar characteristics
 to the training dataset.

Anomaly detection: GAE can be employed for detecting anomalous patterns or chords in a MIDI
composition. By comparing the input MIDI file's graph structure with the learned patterns
in the dataset, the model can identify chords or sequences that deviate from the norm.

Music recommendation: GAE can be used to develop a music recommendation system based
on the learned relationships between chords and MIDI compositions. By comparing the latent
representations of different MIDI files, the model can suggest similar compositions or pieces
that share a common structure or style.

Author Mus spyroot@gmail.com
"""
import argparse
from typing import Optional, Tuple

import torch
import torch_geometric.transforms as T
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GAE, VGAE, GCNConv
from neural_graph_composer.midi_dataset import MidiDataset
from sklearn.cluster import KMeans


class RandomNodeDrop(T.BaseTransform):
    """Randomly drop nodes from the graph."""

    def __init__(self, p=0.5):
        """
        :param p: p (float): The probability of dropping each node.
        """
        self.p = p

    def __call__(self, data: Data) -> Data:
        """

        :param data:
        :return:
        """
        # print(f"Got data {data}")
        num_nodes = data.num_nodes
        node_idx = torch.arange(num_nodes)
        drop_idx = node_idx[torch.randperm(num_nodes)[:int(self.p * num_nodes)]]
        remain_idx = node_idx[~torch.isin(node_idx, drop_idx)]

        data.edge_index, data.edge_attr = RandomNodeDrop.filter_adj(
            data.edge_index, data.edge_attr, remain_idx, num_nodes)
        data.x = data.x[remain_idx]
        data.y = data.y[remain_idx]

        data.train_mask = data.train_mask[remain_idx]
        data.val_mask = data.val_mask[remain_idx]
        data.test_mask = data.test_mask[remain_idx]
        data.num_nodes = data.x.shape[0]

        # print(f"called return data {data}")
        return data

    @staticmethod
    def filter_adj(
            edge_index: torch.Tensor,
            edge_attr: Optional[torch.Tensor], keep_idx: torch.Tensor,
            num_nodes: Optional[int] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        if num_nodes is None:
            num_nodes = int(edge_index.max()) + 1

        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[keep_idx] = 1
        row, col = edge_index
        mask_row = mask[row]
        mask_col = mask[col]
        mask_all = mask_row & mask_col
        if edge_attr is not None:
            return edge_index[:, mask_all], edge_attr[mask_all]
        else:
            return edge_index[:, mask_all]

    def __repr__(self):
        return f'{self.__class__.__name__}(p={self.p})'


class RandomEdgeDrop(T.BaseTransform):
    def __init__(self, p=0.5):
        """
        :param p:
        """
        self.p = p

    def __call__(self, data):
        """
        :param data:
        :return:
        """
        num_edges = data.edge_index.shape[1]
        mask = torch.rand(num_edges) >= self.p

        edge_index = data.edge_index[:, mask]
        edge_attr = None
        if data.edge_attr is not None:
            edge_attr = data.edge_attr[mask]

        pos_edge_mask = None
        neg_edge_mask = None
        if data.pos_edge_label is not None and data.neg_edge_label is not None:
            pos_edge_mask = data.pos_edge_label_index[:, mask]
            neg_edge_mask = data.neg_edge_label_index[:, mask]

        return Data(
            x=data.x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=data.y,
            pos_edge_label=data.pos_edge_label,
            neg_edge_label=data.neg_edge_label,
            train_mask=data.train_mask,
            val_mask=data.val_mask,
            test_mask=data.test_mask,
            pos_edge_label_index=pos_edge_mask,
            neg_edge_label_index=neg_edge_mask,
            node_hash=data.node_hash
        )

    def __repr__(self):
        return f'{self.__class__.__name__}(p={self.p})'


class GCNEncoder(torch.nn.Module):
    """GCN Encoder
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class VariationalGCNEncoder(torch.nn.Module):
    """VAE GCN Encoder
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class LinearEncoder(torch.nn.Module):
    """ Liner GCN
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class VariationalLinearEncoder(torch.nn.Module):
    """ VariationalLinearEncoder
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_mu = GCNConv(in_channels, out_channels)
        self.conv_logstd = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


def train(train_data, model, optimizer, args):
    """
    :param train_data:
    :param model:
    :param optimizer:
    :param args:
    :return:
    """
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)
    # z = model.encode(train_data.x, train_data.edge_index.long())
    loss = model.recon_loss(z, train_data.pos_edge_label_index)
    if args.variational:
        loss = loss + (1 / train_data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(data, model):
    """
    :param data:
    :param model:
    :return:
    """
    model.eval()
    z = model.encode(data.x, data.edge_index)

    pos_indices = data.pos_edge_label_index.t()
    neg_indices = data.neg_edge_label_index.t()

    all_indices = torch.cat([pos_indices, neg_indices], dim=0)
    y_true = torch.zeros((len(all_indices),), dtype=torch.long)
    y_true[:pos_indices.size(0)] = 1

    all_indices = all_indices.to(torch.long)
    edge_index = torch.stack([all_indices[:, 0], all_indices[:, 1]], dim=0)

    #
    y_pred = model.decoder(z, edge_index)
    y_pred = (y_pred > 0.5).type(torch.long)
    acc = (y_pred == y_true).sum().item() / y_true.size(0)
    return model.test(z, data.pos_edge_label_index, data.neg_edge_label_index), acc


def predict_link(model, x, edge_index, node_a, node_b):
    """
    :param model:
    :param x:
    :param edge_index:
    :param node_a:
    :param node_b:
    :return:
    """
    model.eval()
    z = model.encode(x, edge_index)
    similarity = torch.sigmoid(torch.dot(z[node_a], z[node_b]))
    return similarity.item()


def node_clustering(model, x, edge_index, num_clusters):
    """
    :param model:
    :param x:
    :param edge_index:
    :param num_clusters:
    :return:
    """
    model.eval()
    z = model.encode(x, edge_index)
    embeddings = z.detach().cpu().numpy()
    kmeans = KMeans(n_clusters=num_clusters)
    clusters = kmeans.fit_predict(embeddings)
    return clusters


def visualize_embeddings(model, x, edge_index):
    """
    :param model:
    :param x:
    :param edge_index:
    :return:
    """
    model.eval()
    z = model.encode(x, edge_index)
    embeddings = z.detach().cpu().numpy()

    tsne = TSNE(n_components=2)
    reduced_embeddings = tsne.fit_transform(embeddings)

    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], s=20)
    plt.show()


def main(args):
    """
    :param args:
    :return:
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        RandomNodeDrop(p=0.1),
        T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                          split_labels=True, add_negative_train_samples=False),
    ])

    dataset = MidiDataset(root="./data", transform=transform)
    batch_size = args.batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    in_channels, out_channels = dataset.num_features, 32

    if not args.variational and not args.linear:
        model = GAE(GCNEncoder(in_channels, out_channels))
        print("Creating GAE with GCNEncoder")
    elif not args.variational and args.linear:
        model = GAE(LinearEncoder(in_channels, out_channels))
        print("Creating GA with LinearEncoder")
    elif args.variational and not args.linear:
        model = VGAE(VariationalGCNEncoder(in_channels, out_channels))
        print("Creating VGAE with VariationalGCNEncoder")
    elif args.variational and args.linear:
        model = VGAE(VariationalLinearEncoder(in_channels, out_channels))
        print("Creating VGAE with VariationalLinearEncoder")

    models = [
        ("GAE with GCNEncoder", GAE(GCNEncoder(in_channels, out_channels))),
        ("GAE with LinearEncoder", GAE(LinearEncoder(in_channels, out_channels))),
        ("VGAE with VariationalGCNEncoder", VGAE(VariationalGCNEncoder(in_channels, out_channels))),
        ("VGAE with VariationalLinearEncoder", VGAE(VariationalLinearEncoder(in_channels, out_channels))),
    ]

    for model_name, model in models:
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        test_x_list = []
        test_edge_index_list = []
        for epoch in range(1, args.epochs + 1):
            train_loss = 0
            for batch in dataloader:
                train_data, val_data, test_data = batch
                loss = train(train_data, model, optimizer, args)
                train_loss += loss
                metric, acc = test(test_data, model)
                auc, ap = metric
                print(f'Epoch: {epoch:03d}, Model: {model_name: <40}, Loss: {loss:.4f}, '
                      f'AUC: {auc:.4f}, AP: {ap:.4f}, ACC: {acc:.4f}')
                if epoch == 1:
                    test_x_list.append(test_data.x)
                    test_edge_index_list.append(test_data.edge_index)

            # for batch in dataloader:
            #     train_data, val_data, test_data = batch
            # train_loss /= len(dataloader)
            # metric_val, acc_val = test(val_data, model)
            # auc_val, ap_val = metric_val

            print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, ')
            if epoch % args.save_interval == 0:
                torch.save(model.state_dict(), f'{model_name}_epoch_{epoch}.pt')

    test_x = torch.cat(test_x_list, dim=0)
    test_edge_index = torch.cat(test_edge_index_list, dim=1)
    combined_test_data = Data(x=test_x, edge_index=test_edge_index)

    node_a_index, node_b_index = 5, 10
    node_a_hash = dataset.index_to_hash[node_a_index]
    node_b_hash = dataset.index_to_hash[node_b_index]

    note_a = dataset.notes_to_hash[node_a_hash]
    note_b = dataset.notes_to_hash[node_b_hash]
    print(dataset.hash_to_notes)

    similarity = predict_link(
        model, combined_test_data.x,
        combined_test_data.edge_index,
        node_a_index, node_b_index)

    print(f'Similarity between node with hash {note_a} and node with hash {note_b}: {similarity}')

    # num_clusters = 5
    # clusters = node_clustering(model, combined_test_data.x, combined_test_data.edge_index, num_clusters)
    # print(f'Node clusters: {clusters}')
    # visualize_embeddings(model, combined_test_data.x, combined_test_data.edge_index)


def random_edge_drop_checker():
    """Validate random edge  drop.  It should be less num edges.
    :return:
    """
    transform = RandomEdgeDrop(p=0.5)
    dataset = MidiDataset(root="./data")
    dataset.transform = transform
    dataloader = DataLoader(dataset, batch_size=2)
    # Check number of nodes before and after RandomNodeDrop transformation
    for batch in dataloader:
        data = batch[0]
        print("Number of nodes before transformation:", data.edge_index)
        data = transform(data)
        print("Number of nodes after transformation:", data.edge_index)


def random_node_drop_checker():
    """Validate random drop is working.  It should be less number of nodes
    :return:
    """
    transform = RandomNodeDrop(p=0.5)
    dataset = MidiDataset(root="./data")
    dataset.transform = transform
    dataloader = DataLoader(dataset, batch_size=2)
    # Check number of nodes before and after RandomNodeDrop transformation
    for batch in dataloader:
        data = batch[0]
        print("Number of nodes before transformation:", data.num_nodes)
        data = transform(data)
        print("Number of nodes after transformation:", data.num_nodes)


try:
    import google.colab
    IN_COLAB = True
except:
    IN_COLAB = False

if __name__ == '__main__':
    """
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--variational', action='store_true')
    parser.add_argument('--linear', action='store_true')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--save_interval', type=int, default=200)
    parser.add_argument('--wandb', action='store_true', help='Track experiment')
    args = parser.parse_args()
    main(args)

    # # Predict link between node_a and node_b
    # node_a, node_b = 5, 10
    # similarity = predict_link(model, test_data.x, test_data.edge_index, node_a, node_b)
    # print(f'Similarity between node {node_a} and {node_b}: {similarity}')
    #
    # # Perform node clustering
    # num_clusters = 5
    # clusters = node_clustering(model, test_data.x, test_data.edge_index, num_clusters)
    # print(f'Node clusters: {clusters}')
    #
    # # Visualize embeddings
    # visualize_embeddings(model, test_data.x, test_data.edge_index)
