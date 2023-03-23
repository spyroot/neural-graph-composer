import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from neural_graph_composer.midi_dataset import MidiDataset


class Experiments:
    """
    """
    def __init__(self, epochs, batch_size, midi_dataset):
        """
        """
        if epochs is None:
            raise ValueError("epochs cannot be None")
        if batch_size is None:
            raise ValueError("batch_size cannot be None")
        if midi_dataset is None:
            raise ValueError("midi_dataset cannot be None.")
        if not hasattr(midi_dataset, '__getitem__') or not hasattr(midi_dataset, '__len__'):
            raise ValueError("midi_dataset must have __getitem__ and __len__ methods.")

        assert isinstance(batch_size, int) and batch_size > 0, "batch_size must be a positive integer"
        assert isinstance(epochs, int) and epochs > 0, "epochs must be a positive integer"

        self._dataset = midi_dataset
        self.num_classes = self._dataset.total_num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        assert isinstance(self.device, torch.device), "device must be a torch.device object"

        self.datasize = 0
        self.test_size = 0
        self._num_workers = 0
        self._batch_size = batch_size
        self._epochs = epochs

        #
        self.data_loader = None
        self.train_ration = 0.8
        self.test_dataset = None
        self.train_dataset = None
        self.model = None
        self.optimizer = None

    def split(self, ds):
        """
        :return:
        """
        if ds is None:
            raise ValueError("ds cannot be None")

        ds = ds.shuffle()
        # Define the train-test split ratio (e.g., 80% train, 20% test)
        train_size = int(len(ds) * self.train_ration)
        test_size = len(ds) - train_size
        self.train_dataset, self.test_dataset = torch.utils.data.random_split(ds, [train_size, test_size])

    @staticmethod
    def remove_nodes_and_edges(data, remove_ratio=0.2):
        """remove nodes and edges.
        :param data:
        :param remove_ratio:
        :return:
        """
        if data is None:
            raise ValueError("data cannot be None")

        num_nodes_to_remove = int(data.num_nodes * remove_ratio)
        nodes_to_remove = np.random.choice(data.num_nodes, num_nodes_to_remove, replace=False)
        mask = torch.ones(data.num_nodes, dtype=torch.bool)
        mask[nodes_to_remove] = False
        data.x = data.x[mask]
        data.edge_index = data.edge_index[:, mask[data.edge_index[0]] & mask[data.edge_index[1]]]
        return data

    def dataset_split_mask(self, split_mask_ration=0.8):
        """Compute split mask
        :return:
        """
        if self._dataset is None:
            raise ValueError("ds cannot be None")

        # Split the dataset into a train set and a test set
        train_size = int(split_mask_ration * len(self._dataset))
        test_size = len(self._dataset) - train_size
        train_dataset, test_dataset = random_split(self._dataset, [train_size, test_size])
        # create binary masks for the train set and test set
        train_mask = torch.zeros(len(self._dataset), dtype=torch.bool)
        test_mask = torch.zeros(len(self._dataset), dtype=torch.bool)
        train_indices = train_dataset.indices
        test_indices = test_dataset.indices
        train_mask[train_indices] = True
        test_mask[test_indices] = True


def complex_synthetic():
    """geneerate graph useful to test LSTM and djakstra.
    :return:
    """
    subgraph_x = torch.tensor([
        [1, 0],
        [0, 1],
        [1, 1],
        [0, 0],
        [1, 0]
    ], dtype=torch.float)

    subgraph_edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 0],
        [1, 0, 2, 1, 3, 2, 4, 3, 0, 4]
    ], dtype=torch.long)

    subgraph_edge_attr = torch.tensor([0.5, 0.6, 0.3,
                                       0.2, 0.9, 0.4, 0.7,
                                       0.8, 0.1, 0.5], dtype=torch.float)
    subgraph_data = Data(
        x=subgraph_x,
        edge_index=subgraph_edge_index,
        edge_attr=subgraph_edge_attr)
    subgraph_data_list = [subgraph_data]
    return subgraph_data_list


def create_synthetic_data():
    """
    :return:
    """
    subgraph_x = torch.tensor([
        [1, 0],
        [0, 1],
        [1, 1]
    ], dtype=torch.float)

    subgraph_edge_index = torch.tensor([
        [0, 1, 1, 2],
        [1, 0, 2, 1]
    ], dtype=torch.long)

    # 3 edges
    subgraph_edge_attr = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float)
    subgraph_data = Data(
        x=subgraph_x,
        edge_index=subgraph_edge_index,
        edge_attr=subgraph_edge_attr,
        y=torch.tensor([1, 0, 0]))

    return subgraph_data
