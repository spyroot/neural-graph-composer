"""
Several transform used in different spots.

Add Degree Transform used to add node degree
to feature vector. I found this method provide better result.

Randomly drop nodes or edges.

GraphIDTransform add graph id. It useful
for graph classification task.

Author Mus spyroot@gmail.com
"""
import torch

from typing import Optional, Tuple

import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.data import Data

import torch
from torch_geometric.transforms import BaseTransform


class AddDegreeTransform(BaseTransform):
    """Add degree to x.
    """
    def __call__(self, data):
        x = data.x.reshape(data.num_nodes, -1)
        degree = torch_geometric.utils.degree(data.edge_index[0], num_nodes=data.num_nodes)
        degree = degree.reshape(-1, 1)
        data.directed = True
        data.x = torch.cat([x, degree], dim=1)
        return data

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class GraphIDTransform(BaseTransform):
    """This one I created to add graph_id.  Note I got good result on graphs but that
    was not a point.
    """

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.num_classes = dataset.num_classes
        print(f"Creating Graph ID transform Node proper {self.num_classes}")

    def __call__(self, data):
        """Transform add graph id.

        It requires dataset to have public method that it can access.
        For example what I did in my dataset.

        def get_graph_id(self):
            return self._current_index

        def __getitem__(self, index):
            self._current_index = index
        return super().__getitem__(index)

        :param data:
        :return:
        """
        graph_id = self.dataset.get_graph_id()
        data.graph_id = torch.tensor(graph_id, dtype=torch.long)
        return data

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class RandomNodeDrop(T.BaseTransform):
    """Randomly drop nodes from the graph."""

    def __init__(self, p=0.5):
        """
        :param p: p (float): The probability of dropping each node.
        """
        print(f"Creating Random Node proper with p {p}")
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
        """Filter adj
        :param edge_index:
        :param edge_attr:
        :param keep_idx:
        :param num_nodes:
        :return:
        """
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


def example_normalize(y_hash_values):
    # unique hash values from your dataset.
    unique_hash_values = set(y_hash_values)
    # transform = HashToIndexTransform(unique_hash_values)
    # x_normalized, y_index = transform(x, y_hash)  # Apply the transform to a single data point (x, y_hash).

    # @property
    # def num_classes(self):
    #     """
    #     :return:
    #     """
    #     if self.__num_classes is None:
    #         self.__num_classes = self.calculate_num_classes()
    #     return self.__num_classes
