"""
It minimum implementation and model to train for node classification task.
it uses subset of dataset you should get 90+ accuracy and expect
simular F1 score.

Author Mus spyroot@gmail.com
           mbayramo@stanford.edu
"""
import argparse
from math import log
from typing import Optional, Tuple

import numpy as np
import wandb

import torch
from torch import Tensor
from torch.nn import Linear
from torch.nn import Parameter
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn.inits import glorot
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.loader import DataLoader
from torch_geometric.logging import init_wandb
from torch_geometric.nn import GCN2Conv
from torch_geometric.typing import OptTensor, Adj

from torch_sparse import matmul
from torch_sparse import SparseTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score

from neural_graph_composer.midi_dataset import MidiDataset
from neural_graph_composer.transforms import AddDegreeTransform
from ngc_shared import Activation, Experiments


class NgcGCN2Conv(MessagePassing):
    r"""The graph convolutional operator with initial residual connections and
    identity mapping (GCNII) from the `"Simple and Deep Graph Convolutional
    Networks" <https://arxiv.org/abs/2007.02133>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \left( (1 - \alpha) \mathbf{\hat{P}}\mathbf{X} +
        \alpha \mathbf{X^{(0)}}\right) \left( (1 - \beta) \mathbf{I} + \beta
        \mathbf{\Theta} \right)

    with :math:`\mathbf{\hat{P}} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
    \mathbf{\hat{D}}^{-1/2}`, where
    :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the adjacency
    matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix,
    and :math:`\mathbf{X}^{(0)}` being the initial feature representation.
    Here, :math:`\alpha` models the strength of the initial residual
    connection, while :math:`\beta` models the strength of the identity
    mapping.
    The adjacency matrix can include other values than :obj:`1` representing
    edge weights via the optional :obj:`edge_weight` tensor.

    Args:
        channels (int): Size of each input and output sample.
        alpha (float): The strength of the initial residual connection
            :math:`\alpha`.
        theta (float, optional): The hyperparameter :math:`\theta` to compute
            the strength of the identity mapping
            :math:`\beta = \log \left( \frac{\theta}{\ell} + 1 \right)`.
            (default: :obj:`None`)
        layer (int, optional): The layer :math:`\ell` in which this module is
            executed. (default: :obj:`None`)
        shared_weights (bool, optional): If set to :obj:`False`, will use
            different weight matrices for the smoothed representation and the
            initial residual ("GCNII*"). (default: :obj:`True`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        normalize (bool, optional): Whether to add self-loops and apply
            symmetric normalization. (default: :obj:`True`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F)`,
          initial node features :math:`(|\mathcal{V}|, F)`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge weights :math:`(|\mathcal{E}|)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F)`
    """

    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, channels: int, alpha: float, theta: float = None,
                 layer: int = None, shared_weights: bool = True,
                 cached: bool = False, add_self_loops: bool = True,
                 normalize: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.channels = channels
        self.alpha = alpha
        self.beta = 1.
        if theta is not None or layer is not None:
            assert theta is not None and layer is not None
            self.beta = log(theta / layer + 1)
        self.cached = cached
        self.normalize = normalize
        self.add_self_loops = add_self_loops

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.weight1 = Parameter(torch.Tensor(channels, channels))

        if shared_weights:
            self.register_parameter('weight2', None)
        else:
            self.weight2 = Parameter(torch.Tensor(channels, channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight1)
        glorot(self.weight2)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, x_0: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, self.flow, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, self.flow, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        x = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)

        x.mul_(1 - self.alpha)
        x_0 = self.alpha * x_0[:x.size(0)]

        if self.weight2 is None:
            out = x.add_(x_0)
            out = torch.addmm(out, out, self.weight1, beta=1. - self.beta,
                              alpha=self.beta)
        else:
            out = torch.addmm(x, x, self.weight1, beta=1. - self.beta,
                              alpha=self.beta)
            out = out + torch.addmm(x_0, x_0, self.weight2,
                                    beta=1. - self.beta, alpha=self.beta)

        return out

    def message(self, x_j: Tensor, edge_weight: Optional[Tensor] = None) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.channels}, '
                f'alpha={self.alpha}, beta={self.beta})')


class NeuralGraphComposer(torch.nn.Module):
    """
    """
    def __init__(
            self,
            num_features, num_classes,
            hidden_channels: Optional[int] = 2048,
            num_layers: Optional[int] = 9,
            alpha: Optional[float] = 0.5,
            theta: Optional[float] = 1.0,
            shared_weights: Optional[bool] = True,
            dropout: Optional[float] = 0.0):
        """
        :param hidden_channels:
        :param num_layers:
        :param num_features:
        :param num_classes:
        :param alpha:
        :param theta:
        :param shared_weights: weight matrices for the smoothed representation and the initial residua
        :param dropout:
        """
        super().__init__()
        print(f"Create {num_features} {num_classes}")
        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(num_features, hidden_channels))
        self.lins.append(Linear(hidden_channels, num_classes))

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                GCN2Conv(hidden_channels, alpha, theta, layer + 1,
                         shared_weights, normalize=False))

        self.dropout = dropout

    def forward(self, x, adj_t):
        """
        :param x:
        :param adj_t:
        :return:
        """
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


class NgcTrainer(Experiments):
    """
    In first experiment node classification task.
    We have a MIDI dataset and want a classifier to classify each node in the input graph.
    Model allow to swap activation and experiment with GCN , GIN and GAT.
    """

    def __init__(
            self,
            args,
            midi_dataset: MidiDataset,
            device: Optional[torch.device] = "cpu",
            model_type: Optional[str] = "GCN3",
            activation: Optional[Activation] = Activation.ReLU,
            train_update_rate: Optional[int] = 1,
            test_update_freq: Optional[int] = 10,
            eval_update_freq: Optional[int] = 20,
            save_freq: Optional[int] = 100,
            is_data_split: Optional[bool] = False):
        """
        This a node classification trainer.

        :param args:
        :param midi_dataset:
        :param device:
        :param model_type:
        :param activation:
        :param train_update_rate:
        :param test_update_freq:
        :param eval_update_freq:
        :param save_freq:
        :param is_data_split:
        """
        super().__init__(
            midi_dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            train_update_rate=train_update_rate,
            test_update_freq=test_update_freq,
            eval_update_freq=eval_update_freq,
            save_freq=save_freq)

        self.args = args
        print(args)

        self.dataset = midi_dataset
        self.train_loader = DataLoader(
            self.dataset[99:120], batch_size=self._batch_size, shuffle=True)

        self.test_loader = DataLoader(
            self.dataset[99:120], batch_size=self._batch_size, shuffle=False)

        self.model = NeuralGraphComposer(
            midi_dataset.num_features, midi_dataset.num_classes,
            hidden_channels=args.hidden_channels,
            num_layers=args.num_layers, alpha=args.alpha, theta=args.theta,
            shared_weights=False, dropout=args.dropout).to(device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def train(self, data):
        """train loop with cross entropy
        """
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(data.x, data.edge_index)
        losses = []
        for mask_id in range(data.train_mask.shape[1]):
            mask = data.train_mask[:, mask_id]
            loss = F.cross_entropy(out[mask], data.y[mask])
            loss.backward(retain_graph=True)
            losses.append(loss.item())
        self.optimizer.step()
        return torch.tensor(losses).mean().item()

    @torch.no_grad()
    def test(self, data):
        """ Test on 3 set test , train validation. All stats aggregated including F1"""
        self.model.eval()
        pred = self.model(data.x, data.edge_index).argmax(dim=-1)
        # accuracy metric table for train, val , test set.
        accuracy_metrics = [[0.0] * 3 for _ in range(data.train_mask.shape[1])]
        f1_scores = [[0.0] * 3 for _ in range(data.train_mask.shape[1])]
        precision_scores = [[0.0] * 3 for _ in range(data.train_mask.shape[1])]
        recall_scores = [[0.0] * 3 for _ in range(data.train_mask.shape[1])]

        for mask_id in range(data.train_mask.shape[1]):
            for m, mask in enumerate([data.train_mask[:, mask_id],
                                      data.val_mask[:, mask_id],
                                      data.test_mask[:, mask_id]]):
                mask_sum = int(mask.sum())
                mask_pred = pred[mask]
                mask_y = data.y[mask]
                accuracy_metrics[mask_id][m] += int((mask_pred == mask_y).sum()) / mask_sum
                y_cpu = mask_y.cpu().numpy()
                pred_cpu = mask_pred.cpu().numpy()
                f1_scores[mask_id][m] += f1_score(y_cpu, pred_cpu, average='micro')
                precision_scores[mask_id][m] = precision_score(y_cpu, pred_cpu, average='micro')
                recall_scores[mask_id][m] = recall_score(y_cpu, pred_cpu, average='micro')

        return \
            [sum(column) / len(accuracy_metrics)
             for column in zip(*accuracy_metrics)], \
                [sum(column) / len(f1_scores)
                 for column in zip(*f1_scores)], \
                [sum(column) / len(precision_scores)
                 for column in zip(*precision_scores)], \
                [sum(column) / len(recall_scores)
                 for column in zip(*recall_scores)]

    def trainer(self):
        """
        :param args:
        :return:
        """

        device = self.device
        best_val_acc = final_test_acc = best_test_f1 = best_val_f1 = 0
        for epoch in range(1, self._epochs + 1):
            batch_loss = 0.0
            train_num_batches = len(self.train_loader)
            losses = np.zeros((train_num_batches, 1))
            for i, b in enumerate(self.train_loader):
                b.to(device)
                loss = self.train(b)
                losses[i] = loss

            num_batches = len(self.test_loader)
            f1_arr = np.zeros((num_batches, 3))
            precision_arr = np.zeros((num_batches, 3))
            recall_arr = np.zeros((num_batches, 3))
            accuracy_arr = np.zeros((num_batches, 3))

            for i, b in enumerate(self.test_loader):
                b.to(device)
                accuracies, f1_scores, precision_scores, recall_scores = self.test(b)
                accuracy_arr[i] = accuracies
                precision_arr[i] = precision_scores
                recall_arr[i] = recall_scores
                f1_arr[i] = f1_scores

            mean_loss = losses.mean()
            mean_accuracy = accuracy_arr.mean(axis=0)
            mean_f1 = f1_arr.mean(axis=0)
            mean_precision = precision_arr.mean(axis=0)
            mean_recall = recall_arr.mean(axis=0)

            if mean_accuracy[1] > final_test_acc:
                final_test_acc = mean_accuracy[1]
            if mean_accuracy[2] > best_val_acc:
                best_val_acc = mean_accuracy[2]
            if mean_f1[1] > best_test_f1:
                best_test_f1 = mean_f1[1]
            if mean_f1[2] > best_val_f1:
                best_val_f1 = mean_f1[2]

            if self.args.wandb:
                wandb.log({
                    "epoch": epoch,
                    "loss": mean_loss,
                    "train_accuracy": mean_accuracy[0],
                    "val_accuracy": mean_accuracy[1],
                    "test_accuracy": mean_accuracy[2],
                    "train_f1": mean_f1[0],
                    "val_f1": mean_f1[1],
                    "test_f1": mean_f1[2],
                    "train_precision": mean_precision[0],
                    "val_precision": mean_precision[1],
                    "test_precision": mean_precision[2],
                    "train_recall": mean_recall[0],
                    "val_recall": mean_recall[1],
                    "test_recall": mean_recall[2]
                })

            print(f"Epoch: {epoch}, Loss: {batch_loss:.5f}")
            print(
                f"{'':<8}{'Accuracy':<14}{'F1':<14}{'Precision':<14}{'Recall':<14}"
                f"\n{'Train:':<8}{mean_accuracy[0]:.5f}{'':<7}"
                f"{mean_f1[0]:.5f}{'':<7}{mean_precision[0]:.5f}{'':<7}{mean_recall[0]:.5f}"
                f"\n{'Val:':<8}{mean_accuracy[1]:.5f}{'':<7}{mean_f1[1]:.5f}{'':<7}"
                f"{mean_precision[1]:.5f}{'':<7}{mean_recall[1]:.5f}"
                f"\n{'Test:':<8}{mean_accuracy[2]:.5f}{'':<7}{mean_f1[2]:.5f}{'':<7}"
                f"{mean_precision[2]:.5f}{'':<7}{mean_recall[2]:.5f}"
            )

        return final_test_acc, best_val_acc, best_test_f1, best_val_f1


def main():
    """
    :return:

    num_features, num_classes,
            hidden_channels: Optional[int] = 2048,
            num_layers: Optional[int] = 9,
            alpha: Optional[float] = 0.5,
            theta: Optional[float] = 1.0,
            shared_weights: Optional[bool] = True,
            dropout: Optional[float] = 0.0):
    """

    parser = argparse.ArgumentParser()
    model_params = parser.add_argument_group("Model Parameters")
    model_params.add_argument('--hidden_channels', type=int, default=2048)
    model_params.add_argument('--heads', type=int, default=8)
    model_params.add_argument('--num_layers', type=int, default=9)
    model_params.add_argument('--alpha', type=float, default=0.5)
    model_params.add_argument('--theta', type=float, default=1.0)
    model_params.add_argument('--shared_weights', action='store_true', default=False)
    model_params.add_argument('--dropout', type=float, default=0.2)
    model_params.add_argument('--add_self_loops', action='store_true', default=True)
    model_params.add_argument('--normalize', action='store_true', default=True)

    optimizer_params = parser.add_argument_group("Optimizer Parameters")
    optimizer_params.add_argument('--lr', type=float, default=0.005)
    optimizer_params.add_argument('--epochs', type=int, default=1000)
    optimizer_params.add_argument('--weight_decay', type=float, default=5e-4)

    dataset_misc = parser.add_argument_group("Dataset and Experiment")
    dataset_misc.add_argument('--wandb', action='store_true', help='Track experiment')
    dataset_misc.add_argument('--ds_location', type=str, default="./data", help="Track experiment")
    dataset_misc.add_argument('--tolerance', type=float, default=0.5, help="tolerance for midi timing")
    dataset_misc.add_argument('--batch_size', type=int, default=8, help="batch size")

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    init_wandb(name=f'NGC-node-classification',
               epochs=args.epochs,
               hidden_channels=args.hidden_channels,
               lr=args.lr, device=device)

    transform = T.Compose([
        AddDegreeTransform(),
        T.NormalizeFeatures(),
    ])

    pre_transform = T.Compose([T.GCNNorm()])

    dataset = MidiDataset(root=args.ds_location,
                          transform=transform,
                          pre_transform=pre_transform,
                          per_instrument_graph=False,
                          tolerance=args.tolerance,
                          include_velocity=True)

    transform.transforms.insert(0, RandomNodeSplit(
        num_splits=2, num_train_per_class=100, num_val=0.1, num_test=0.3))

    ngc_trainer = NgcTrainer(args, dataset, device=device)

    try:
        best_val_acc, final_test_acc, best_f1_test, best_f1_val = ngc_trainer.trainer()
        print(
            f"Best validation accuracy: {best_val_acc:.5f}, "
            f"Best test  accuracy: {final_test_acc:.5f}, "
            f"Best val f1: {best_f1_val:.5f} "
            f"Best test f1: {best_f1_test:.5f}, "
        )
    except KeyboardInterrupt:
        print("Training interrupted. Finishing wandb run...")
        if args.wandb:
            wandb.finish()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Training interrupted. Finishing wandb run...")
        wandb.finish()
