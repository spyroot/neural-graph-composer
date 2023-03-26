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
import glob
import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch_geometric.transforms as T
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GAE, VGAE, GCNConv
from neural_graph_composer.midi_dataset import MidiDataset
from sklearn.cluster import KMeans
from example_shared import Experiments, Activation

try:
    import google.colab

    IN_COLAB = True
except:
    IN_COLAB = False


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
        """
        :return:
        """
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


class VariationGae(Experiments):
    def __init__(
            self, epochs: int,
            batch_size: int,
            midi_dataset: MidiDataset, hidden_dim: int,
            model_type: Optional[str] = "",
            lr: Optional[float] = 0.01,
            activation: Optional[Activation] = Activation.ReLU,
            train_update_rate: Optional[int] = 1,
            test_update_freq: Optional[int] = 10,
            eval_update_freq: Optional[int] = 20,
            save_freq: Optional[int] = 20):
        """Example experiment for training a graph neural network on MIDI data.

        :param epochs: num epochs
        :param batch_size: default batch (for colab on GPU use 4)
        :param hidden_dim: hidden for all models.
        :param model_type:
         :param lr: learning rate.
        """
        super().__init__(
            epochs, batch_size, midi_dataset, train_update_rate=train_update_rate,
            test_update_freq=test_update_freq,
            eval_update_freq=eval_update_freq,
            save_freq=save_freq
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        assert self.device is not None, "Device is not set."
        assert self.datasize is not None, "Datasize is not set."
        assert self.test_size is not None, "Test size is not set."
        assert self._num_workers is not None, "Number of workers is not set."
        assert self._batch_size is not None, "Batch size is not set."

        self.datasize = 0
        self.start_epoch = 0
        self.save_freq = save_freq

        self.test_size = 0
        self._num_workers = 0
        self._batch_size = batch_size
        self._hidden_dim = hidden_dim
        self._feature_dim = midi_dataset.num_node_features
        self._num_classes = midi_dataset.total_num_classes
        self._lr = lr

        self.train_dataset = midi_dataset
        self.test_dataset = midi_dataset

        self.train_loader = DataLoader(
            midi_dataset, batch_size=self._batch_size, shuffle=True)

        self.val_loader = DataLoader(
            midi_dataset, batch_size=batch_size, shuffle=False)

        self.test_loader = DataLoader(
            midi_dataset, batch_size=batch_size, shuffle=False)

        self.model_type = model_type

        # self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.1)

    def train(self, train_data, is_variational):
        """
        :param train_data:
        :param is_variational:
        :return:
        """
        self.model.train()
        self.optimizer.zero_grad()
        z = self.model.encode(train_data.x, train_data.edge_index)
        # z = model.encode(train_data.x, train_data.edge_index.long())
        loss = self.model.recon_loss(z, train_data.pos_edge_label_index)
        if is_variational:
            loss = loss + (1 / train_data.num_nodes) * self.model.kl_loss()
        loss.backward()
        self.optimizer.step()
        return float(loss)

    @torch.no_grad()
    def test(self, data, model):
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
        y_pred = self.model.decoder(z, edge_index)
        y_pred = (y_pred > 0.5).type(torch.long)
        acc = (y_pred == y_true).sum().item() / y_true.size(0)
        return self.model.test(z, data.pos_edge_label_index, data.neg_edge_label_index), acc

    @staticmethod
    def predict_link(model, x, edge_index, node_a, node_b):
        """take edge index model x , node_a and node_b and predict similarity
        between node a and node b.
        :param model: trained model
        :param x: original x from the input graph
        :param edge_index: edge index
        :param node_a: node a from a graph
        :param node_b: node b from graph
        :return:
        """
        model.eval()
        z = model.encode(x, edge_index)
        similarity = torch.sigmoid(torch.dot(z[node_a], z[node_b]))
        return similarity.item()

    @staticmethod
    def node_clustering(model, x, edge_index, num_clusters):
        """ Compute k mean clustering
        :param model: trained model
        :param x: x from original graph
        :param edge_index: edge index
        :param num_clusters: number of clusters
        :return:
        """
        model.eval()
        z = model.encode(x, edge_index)
        embeddings = z.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=num_clusters)
        clusters = kmeans.fit_predict(embeddings)
        return clusters

    @staticmethod
    def visualize_embeddings(model, x, edge_index):
        """Take model x and edge index and visualize embedding space.
        Output tsne.
        :param model: trained model
        :param x: x from that graph
        :param edge_index: edge index.
        :return:
        """
        model.eval()
        z = model.encode(x, edge_index)
        embeddings = z.detach().cpu().numpy()
        tsne = TSNE(n_components=2)
        reduced_embeddings = tsne.fit_transform(embeddings)
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], s=20)
        plt.show()

    def load_model(self, model_path):
        """Load the saved model from the specified path
        :param model_path:
        :return:
        """
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def trainer(self, cmd, train_all: Optional[bool] = True):
        """
        :param train_all: train all models otherwise take from args (Default True)
        :param cmd:
        :return:
        """
        in_channels, out_channels = dataset.num_features, 32

        models = []
        if train_all:
            models = [
                ("GAE_GCNEncoder", GAE(GCNEncoder(in_channels, out_channels))),
                ("GAE_LinearEncoder", GAE(LinearEncoder(in_channels, out_channels))),
                ("VGAE_VariationalGCNEncoder", VGAE(VariationalGCNEncoder(in_channels, out_channels))),
                ("VGAE_VariationalLinearEncoder", VGAE(VariationalLinearEncoder(in_channels, out_channels))),
            ]
        else:
            if not cmd.variational and not cmd.linear:
                model = GAE(GCNEncoder(in_channels, out_channels))
                models = [model]
                print("Creating GAE with GCNEncoder")
            elif not cmd.variational and cmd.linear:
                model = GAE(LinearEncoder(in_channels, out_channels))
                models = [model]
                print("Creating GA with LinearEncoder")
            elif cmd.variational and not cmd.linear:
                model = VGAE(VariationalGCNEncoder(in_channels, out_channels))
                models = [model]
                print("Creating VGAE with VariationalGCNEncoder")
            elif cmd.variational and cmd.linear:
                model = VGAE(VariationalLinearEncoder(in_channels, out_channels))
                models = [model]
                print("Creating VGAE with VariationalLinearEncoder")

        for model_name, model in models:
            self.model_type = model_name

            self.optimizer = torch.optim.Adam(model.parameters(), lr=self._lr)
            self.model = model
            self.load_checkpoint(model_name)
            self.model = model.to(device)

            best_val_acc = 0.
            best_epoch = 0.
            best_test_acc = 0.

            test_x_list = []
            test_edge_index_list = []

            current_epoch = len(self.train_losses)
            print(f"current_epoch {current_epoch}")
            for e in range(current_epoch, self._epochs + 1):

                train_loss = 0.
                num_batches = len(self.train_loader)
                test_mean_acc = np.zeros(num_batches)
                test_mean_auc = np.zeros(num_batches)
                test_mean_ap = np.zeros(num_batches)

                for i, batch in enumerate(self.train_loader):
                    train_data, val_data, test_data = batch
                    loss = self.train(train_data, cmd.variational)
                    train_loss += loss
                    metric, acc = self.test(test_data, model)
                    auc, ap = metric
                    test_mean_acc[i] = acc
                    test_mean_auc[i] = auc
                    test_mean_ap[i] = ap

                    print(f'Epoch: {e:03d}, Model: {model_name: <40}, Loss: {loss:.4f}, '
                          f'AUC: {auc:.4f}, AP: {ap:.4f}, ACC: {acc:.4f}')
                    num_batches += 1

                    if e == 1:
                        test_x_list.append(test_data.x)
                        test_edge_index_list.append(test_data.edge_index)

                loss_avg = train_loss / num_batches
                self.update_metrics(loss_avg)

                if test_mean_acc.mean() > best_val_acc:
                    best_val_acc = test_mean_acc.mean()
                    best_epoch = e
                    best_test_acc = test_mean_acc.mean()

                self.update_test_metric(test_acc=test_mean_acc.mean(),
                                        test_auc=test_mean_auc.mean(),
                                        test_ap=test_mean_ap.mean())

                print(f'Epoch: {e:03d}, avg train loss: {loss_avg:.4f}')
                if e % self.save_freq == 0:
                    self.save_checkpoint(e, self.optimizer.state_dict(), model_name=model_name)

            print(f"Best Epoch: {best_epoch}, Test Acc: {best_test_acc:.5f}")
            self.plot_metrics()
            self.clear_metric()
            current_epoch = 0
            best_val_acc = 0

    def inference(self, model, test_x_list, test_edge_index_list):
        """
        :return:
        """
        test_x = torch.cat(test_x_list, dim=0)
        test_edge_index = torch.cat(test_edge_index_list, dim=1)
        combined_test_data = Data(x=test_x, edge_index=test_edge_index)

        node_a_index, node_b_index = 5, 10
        node_a_hash = dataset.index_to_hash[node_a_index]
        node_b_hash = dataset.index_to_hash[node_b_index]

        note_a = dataset.notes_to_hash[node_a_hash]
        note_b = dataset.notes_to_hash[node_b_hash]
        print(dataset.hash_to_notes)

        similarity = self.predict_link(
            model, combined_test_data.x,
            combined_test_data.edge_index,
            node_a_index, node_b_index)

        print(f'Similarity between node with hash {note_a} and node with hash {note_b}: {similarity}')

        # this for inference after we trained.
        # num_clusters = 5
        # clusters = node_clustering(model, combined_test_data.x, combined_test_data.edge_index, num_clusters)
        # print(f'Node clusters: {clusters}')
        # visualize_embeddings(model, combined_test_data.x, combined_test_data.edge_index)

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


def random_edge_drop_checker():
    """Validate random edge  drop.  It should be less num edges.
    :return:
    """
    _transform = RandomEdgeDrop(p=0.5)
    _dataset = MidiDataset(root="./data")
    dataset.transform = _transform
    dataloader = DataLoader(_dataset, batch_size=2)
    # check number of nodes before and after RandomNodeDrop transformation
    for batch in dataloader:
        data = batch[0]
        print("Number of nodes before transformation:", data.edge_index)
        data = transform(data)
        print("Number of nodes after transformation:", data.edge_index)


def random_node_drop_checker():
    """Validate random drop is working.  It should be less numb of nodes
    :return:
    """
    _transform = RandomNodeDrop(p=0.5)
    _dataset = MidiDataset(root="./data")
    dataset.transform = _transform
    dataloader = DataLoader(_dataset, batch_size=2)
    # Check number of nodes before and after RandomNodeDrop transformation
    for batch in iter(dataloader):
        data = batch[0]
        print("Number of nodes before transformation:", data.num_nodes)
        data = transform(data)
        print("Number of nodes after transformation:", data.num_nodes)


if __name__ == '__main__':
    """
    AUC (Area Under the ROC Curve) and AP (Average Precision) 
    are two commonly used evaluation metrics in binary classification tasks. 
    AUC measures the ability of the model to distinguish between 
    positive and negative examples.
    
    It is the area under the Receiver Operating Characteristic (ROC) 
    curve, which is created by plotting the True Positive Rate (TPR) against 
    the False Positive Rate (FPR) at different classification thresholds. 
    AUC ranges from 0 to 1, where 1 indicates a perfect classifier.

    AP measures the precision of the model at different recall levels. 
    It is the average of the precision values calculated at different recall 
    levels, where recall is the fraction of true positive examples that were 
    correctly identified by the model out of all the true positive examples in the 
    dataset. AP ranges from 0 to 1, where 1 indicates a perfect classifier.

    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument('--variational', action='store_true')
    parser.add_argument('--linear', action='store_true')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=200)
    parser.add_argument('--wandb', action='store_true', help='Track experiment')
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--graph_per_instrument', type=bool, default=False)
    parser.add_argument('--random_split', type=bool, default=False)

    args = parser.parse_args()

    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        RandomNodeDrop(p=0.1),
        T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                          split_labels=True, add_negative_train_samples=False),
    ])

    dataset = MidiDataset(root="./data",
                          transform=transform)

    graph_vaes = VariationGae(
        epochs=args.epochs,
        batch_size=args.batch_size,
        midi_dataset=dataset,
        hidden_dim=args.hidden_dim,
        model_type="",
        lr=args.lr,
        activation=Activation.PReLU)
    graph_vaes.trainer(args)
