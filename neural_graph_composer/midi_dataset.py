import argparse
import logging
import os
import os.path as osp
import shutil
from typing import Optional, List, Dict, Callable
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, extract_tar
from torch_geometric.data import download_url

from .midi_graph_builder import MidiGraphBuilder
from .midi_reader import MidiReader
from torch_geometric.data import InMemoryDataset, download_url, extract_zip

from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


class NodeHashIndexMapper:
    def __init__(self):
        self.hash_to_index = {}
        self.index_to_hash = {}
        self.current_index = 0

    def merge_with(self, other_mapper: 'NodeHashIndexMapper'):
        for node_hash in other_mapper.hash_to_index:
            self.add_node_hash(node_hash)

    def add_node_hash(self, node_hash):
        if node_hash not in self.hash_to_index:
            self.hash_to_index[node_hash] = self.current_index
            self.index_to_hash[self.current_index] = node_hash
            self.current_index += 1

    def get_index(self, node_hash):
        return self.hash_to_index[node_hash]

    def get_node_hash(self, index):
        return self.index_to_hash[index]


class HashToIndexTransform:
    def __init__(self, unique_hash_values):
        self.hash_to_index = {hash_val: idx for idx, hash_val in enumerate(unique_hash_values)}

    def __call__(self, x, y_hash):
        y_index = self.hash_to_index[y_hash]
        x_normalized = self.normalize(x)
        return x_normalized, y_index

    def normalize(self, x):
        # Implement your normalization method here, e.g., min-max scaling or standardization.
        # x_normalized = ...
        x_normalized = {}
        return x_normalized


class MidiDataset(InMemoryDataset):
    """Create dataset from list of MIDI files

    per_graph_slit dictates if we want threat each instrument as seperate graph.
    or we want merge


    """
    def __init__(self,
                 root,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 default_node_attr: str = 'attr',
                 file_names: Optional[List[str]] = None,
                 default_webserver: Optional[str] = 'http://localhost:9000',
                 train_ratio: Optional[float] = 0.7,
                 val_ratio: Optional[float] = 0.15,
                 per_instrument_graph: Optional[bool] = True,
                 per_graph_slit: Optional[bool] = True,
                 ):
        """
        :param root: Root directory where the dataset should be saved.
        :param transform:  A function/transform that takes in an
                `torch_geometric.data.Data` object and returns a transformed version.
                The data object will be transformed before every access.
        :param pre_transform:  A function/transform that takes in an
                `torch_geometric.data.Data` object and returns a transformed version.
                The data object will be transformed before being saved to disk.
        :param pre_filter: A function that takes in an
                `torch_geometric.data.Data` object and returns a boolean value, indicating
                whether the data object should be included in the final dataset.
        :param default_node_attr:  Default node attribute name. Name graph builder use to add node attribute
        :param per_instrument_graph:  each instrument is separate graph
        :param default_webserver:   This mainly for debug  we can poll local webserver midi files
        :param train_ratio: Training ratio (default 0.7).
        :param val_ratio: Validation ratio (default 0.15).
        :param per_graph_slit: Whether to split the dataset into graphs.
        """
        if file_names is not None:
            if not isinstance(file_names, list):
                raise ValueError("file_names should be a list of strings.")
            if not all(isinstance(file_name, str) for file_name in file_names):
                raise ValueError("All elements in file_names should be strings.")

        if not (0 <= train_ratio <= 1):
            raise ValueError("train_ratio should be between 0 and 1.")

        if not (0 <= val_ratio <= 1):
            raise ValueError("val_ratio should be between 0 and 1.")

        if train_ratio + val_ratio > 1:
            raise ValueError("The sum of train_ratio and val_ratio should be less than or equal to 1.")

        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)
        self.logger.setLevel(logging.WARNING)

        # root = osp.expanduser(osp.normpath(root))
        # self.root = root

        self.__url = default_webserver
        self.__node_attr_name = default_node_attr
        self.__train_ratio = train_ratio
        self.__val_ratio = val_ratio
        self.__is_instrument_graph = per_instrument_graph
        self.__mask_instruments = per_graph_slit
        self.__hidden_channels = 64
        self.__node_attr_name = default_node_attr
        # self.__dataset_length = len(self.__data_list)
        self.all_classes = set()

        self.data_list = None
        self.__data_list = None
        self._graph_builder = None

        self._num_instruments = {}

        print(f"default_node_attr: {self.__train_ratio}")
        print(f"transform: {self.__val_ratio}")
        print(f"per_instrument_graph: {self.__is_instrument_graph}")
        print(f"pre_filter: {pre_filter}")
        super().__init__(root, transform, pre_transform, pre_filter)

        for path in self.processed_paths:
            print(f"Using processed data at {path}")

        # all read-only properties
        self._notes_to_hash = {}
        self._hash_to_notes = {}
        self._hash_to_index = {}
        self._index_to_hash = {}

        self.__num_classes = None
        if self.processed_file_exists():
            self.load_processed()
        else:
            print("File not found.")
            # out = torch.load(self.processed_paths[0])
            # if not isinstance(out, tuple) or len(out) != 3:
            #     raise RuntimeError("The 'data' object must a tuple.")
            # self.data, self.slices, additional_data = out
            # self.__total_num_classes = len(torch.unique(self.data.y))
            # self._notes_to_hash = additional_data['hash_to_notes']
            # self._hash_to_notes = additional_data['notes_to_hash']
            # self._hash_to_index = additional_data['hash_to_index']
            # self._index_to_hash = additional_data['index_to_hash']

    @property
    def hash_to_index(self) -> Dict[int, int]:
        """  A read-only dictionary mapping hash values to their respective indices.
        :return:
        """
        return self._hash_to_index.copy()

    @property
    def index_to_hash(self) -> Dict[int, int]:
        """  A read-only dictionary mapping indices to their respective hash values.
        :return:
        """
        return self._index_to_hash.copy()

    @property
    def notes_to_hash(self):
        """ A read-only dictionary mapping MIDI notes to their respective hash values.
        :return:
        """
        return self._notes_to_hash.copy()

    @property
    def hash_to_notes(self):
        """ A read-only dictionary mapping hash values to their respective MIDI notes.
        :return:
        """
        return self._hash_to_notes.copy()

    def load_processed(self):
        """Load the processed data from disk,  depending on the flag.
        :return:
        """
        print("Loading processed")

        if self.__is_instrument_graph:
            print(f"Loading {self.processed_dir} per_instrument_data.pt")
            data_path = osp.join(self.processed_dir, 'per_instrument_data.pt')
        else:
            print(f"Loading {self.processed_dir} data.pt")
            data_path = osp.join(self.processed_dir, 'data.pt')

        data = torch.load(data_path)
        if not isinstance(data, tuple) or len(data) != 3:
            raise RuntimeError("The 'data' object must be a tuple.")

        self.data, self.slices, additional_data = data
        self.__num_classes = len(torch.unique(self.data.y))
        self.__data_list = None

        self._hash_to_notes = additional_data['hash_to_notes']
        self._notes_to_hash = additional_data['notes_to_hash']
        self._hash_to_index = additional_data['hash_to_index']
        self._index_to_hash = additional_data['index_to_hash']

        self.__dataset_length = len(self.data)
        self.__num_classes = len(torch.unique(self.data.y))
        self.__total_num_classes = len(torch.unique(self.data.y))

        print(f"Loaded data from {data_path}")
        if self._hash_to_notes is not None:
            print(f"Loaded hash_to_notes with {len(self._hash_to_notes)} entries")
        if self._notes_to_hash is not None:
            print(f"Loaded notes_to_hash with {len(self._notes_to_hash)} entries")
        if self._hash_to_index is not None:
            print(f"Loaded hash_to_index with {len(self._hash_to_index)} entries")
        if self._index_to_hash is not None:
            print(f"Loaded index_to_hash with {len(self._index_to_hash)} entries")

    @property
    def graph_builder(self) -> MidiGraphBuilder:
        """
        :return:
        """
        return self._graph_builder

    def total_num_classes(self):
        """
        :return: Total number of unique classes across all graphs
        """
        return len(self.all_classes)

    @property
    def raw_file_names(self):
        """
        :return:
        """
        # module_dir = os.path.dirname(os.path.abspath(__file__))
        # midi_dir = os.path.join(module_dir, 'data', 'midi')
        midi_files_dir = '/Users/spyroot/dev/neural-graph-composer/neural_graph_composer/dataset'
        return [f for f in os.listdir(midi_files_dir)
                if os.path.isfile(os.path.join(midi_files_dir, f)) and f.endswith('.mid')
                ]

    def processed_file_exists(self):
        """Check if processed file already exists.
        :return: boolean
        """
        print("Checking files.")

        if not osp.exists(self.processed_dir):
            return False

        if self.__is_instrument_graph:
            processed_files = ['per_instrument_data.pt']
            # osp.exists(osp.join(self.processed_dir, file_name))
        else:
            processed_files = ['data.pt']

        # check each file if all processed files exist
        for file_name in self.processed_file_names:
            if not osp.exists(osp.join(self.processed_dir, file_name)):
                return False

        return True

    @property
    def processed_file_names(self):
        """
        :return:
        """
        return ['per_instrument_data.pt', 'data.pt']
        # files = []
        # p = Path(self.processed_dir)
        # if not p.exists():
        #     return files
        #
        # for x in os.listdir(self.processed_dir):
        #     if x.startswith("data") and x.endswith(".pt"):
        #         files.append(x)
        #
        # return files

    def compute_statistics(self):
        """
        :return:
        """
        num_nodes, num_edges, num_labels = 0, 0, 0
        for data in self:
            num_nodes += data.num_nodes
            num_edges += data.num_edges
            num_labels += torch.unique(data.label).size(0)

        return {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "num_labels": num_labels,
        }

    @property
    def total_num_classes(self):
        """
        :return: Total number of unique classes across all graphs
        """
        return len(self.index_to_hash)

    def has_edge_weights(pyg_data, default_key="edge_attr"):
        """
        :param default_key:
        :return:
        """
        return hasattr(pyg_data, default_key) \
            and pyg_data.edge_attr is not None and pyg_data.weight.size(0) > 0

    def download(self):
        """
        :return:
        """
        print(f"raw_file_names {self.raw_file_names}")
        for raw_file in self.raw_file_names:
            print(f"Downloading {raw_file}")
            path = download_url(f"{self.__url}/{raw_file}", self.raw_dir)
            if raw_file.endswith(".zip"):
                extract_zip(path, self.raw_dir)
            if raw_file.endswith(".tar"):
                extract_tar(path, self.raw_dir)

    def _mask_per_instrument(self):
        """Process each midi file construct graph per each instrument,
        create masks each instrument.
        :return:
        """
        if self.__data_list is None:
            self.__data_list = []

        self._graph_builder = MidiGraphBuilder(
            None, is_instrument_graph=True)

        if self._graph_builder is None:
            print("Warning: self.graph_builder is None")

        for raw_path in self.raw_paths:
            print(f"Reading {raw_path}")
            try:
                # read file and construct graph
                midi_seqs = MidiReader.read(raw_path)
                if raw_path not in self._num_instruments:
                    self._num_instruments[raw_path] = midi_seqs.num_instruments()
                self._graph_builder.build(midi_seqs)

                # graph_builder output iterator
                for midi_data in self._graph_builder.graphs():
                    self.all_classes.update(torch.unique(midi_data.y).tolist())
                    # first we apply pre-filter then apply mask
                    if self.pre_filter is not None and not self.pre_filter(midi_data):
                        continue

                    # split mask
                    num_nodes = midi_data.x.size(0)
                    train_ratio, val_ratio = self.__train_ratio, self.__val_ratio
                    train_size = int(train_ratio * num_nodes)
                    val_size = int(val_ratio * num_nodes)

                    indices = np.random.permutation(num_nodes)
                    # Assign training, validation, and testing masks
                    midi_data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
                    midi_data.train_mask[torch.tensor(indices[:train_size])] = True

                    midi_data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
                    midi_data.val_mask[torch.tensor(indices[train_size:train_size + val_size])] = True

                    midi_data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
                    midi_data.test_mask[torch.tensor(indices[train_size + val_size:])] = True

                    if self.pre_transform is not None:
                        midi_data = self.pre_transform(midi_data)

                    self.__data_list.append(midi_data)
            except KeyError as ker_err:
                print(f"Error in file {raw_path} error: {ker_err}")

    def process(self):
        """We compute the mask for each graph separately,
        since  graphs have different sizes or structures.
        This way, you ensure that you have a balanced split for each graph,
        allowing your model to learn and validate on a variety of examples.
        :return:
        """

        if self.processed_file_exists():
            print(f"Processed files found in {self.processed_dir}. Loading...")
            self.load_processed()
            return

        self._mask_per_instrument()

        # Save additional data to a separate file
        _hash_to_notes = self.graph_builder.hash_to_notes
        _notes_to_hash = self.graph_builder.notes_to_hash
        _hash_to_index = self.graph_builder.hash_to_index
        _index_to_hash = self.graph_builder.index_to_hash
        additional_data = {
            'hash_to_notes': _hash_to_notes,
            'notes_to_hash': _notes_to_hash,
            'hash_to_index': _hash_to_index,
            'index_to_hash': _index_to_hash,
        }

        _data, _slices = self.collate(self.__data_list)
        _dataset_length = len(self.__data_list)
        print(f"Saving {self.processed_dir} per_instrument_data.pt")
        torch.save((_data, _slices, additional_data), osp.join(self.processed_dir, f'per_instrument_data.pt'))
        del self.__data_list
        self.__data_list = None
        self._graph_builder = None

        self._mask_entire_graph()
        _hash_to_notes = self.graph_builder.hash_to_notes
        _notes_to_hash = self.graph_builder.notes_to_hash
        _hash_to_index = self.graph_builder.hash_to_index
        _index_to_hash = self.graph_builder.index_to_hash
        additional_data = {
            'hash_to_notes': _hash_to_notes,
            'notes_to_hash': _notes_to_hash,
            'hash_to_index': _hash_to_index,
            'index_to_hash': _index_to_hash,
        }

        print(f"Saving {self.processed_dir} data.pt")
        torch.save((_data, _slices, additional_data), osp.join(self.processed_dir, f'data.pt'))
        del self.__data_list
        self.__data_list = None
        self._graph_builder = None

        self.load_processed()

    def _mask_entire_graph(self):
        """Process and mask all graphs
        :return:
        """
        self._graph_builder = MidiGraphBuilder(
            None, is_instrument_graph=True)

        if self.__data_list is None:
            self.__data_list = []

        for raw_path in self.raw_paths:
            print(f"Reading {raw_path}")
            try:
                midi_seqs = MidiReader.read(raw_path)
                self._graph_builder.build(midi_seqs)
                graphs = self._graph_builder.graphs()
                if not graphs:
                    raise ValueError("No subgraphs found in graph builder.")

                midi_data = next(graphs)

                self.all_classes.update(torch.unique(midi_data.y).tolist())
                if self.pre_filter is not None and not self.pre_filter(midi_data):
                    continue

                # we mask entire graph
                num_graphs = len(midi_data)
                train_ratio, val_ratio = self.__train_ratio, self.__val_ratio
                train_size = int(train_ratio * num_graphs)
                val_size = int(val_ratio * num_graphs)

                # split mask
                num_nodes = midi_data.x.size(0)
                train_ratio, val_ratio = self.__train_ratio, self.__val_ratio
                train_size = int(train_ratio * num_nodes)
                val_size = int(val_ratio * num_nodes)

                indices = np.random.permutation(num_nodes)
                # Assign training, validation, and testing masks
                midi_data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
                midi_data.train_mask[torch.tensor(indices[:train_size])] = True

                midi_data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
                midi_data.val_mask[torch.tensor(indices[train_size:train_size + val_size])] = True

                midi_data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
                midi_data.test_mask[torch.tensor(indices[train_size + val_size:])] = True

                if self.pre_transform is not None:
                    midi_data = self.pre_transform(midi_data)
                self.__data_list.append(midi_data)

                #
                # for idx, data in enumerate(midi_data):
                #     # training, validation, and testing masks
                #     data.train_mask = torch.tensor([idx in indices[:train_size]])
                #     data.val_mask = torch.tensor([idx in indices[train_size:train_size + val_size]])
                #     data.test_mask = torch.tensor([idx in indices[train_size + val_size:]])
                #     # apply pre transform for train, val and test.
                #     if self.pre_transform is not None:
                #         data = self.pre_transform(data)
                #     self.__data_list.append(data)

            except KeyError as ker_err:
                print(f"Error in file {raw_path} error: {ker_err}")
            except FileNotFoundError as file_not_found:
                print(f"Error in file {raw_path} error: {file_not_found}")

    def split(self, train_size, val_size, test_size):
        """
        :param train_size:
        :param val_size:
        :param test_size:
        :return:
        """
        train_dataset = torch.utils.data.Subset(self, self.train_idx[:train_size])
        val_dataset = torch.utils.data.Subset(self, self.val_idx[:val_size])
        test_dataset = torch.utils.data.Subset(self, self.test_idx[:test_size])
        return train_dataset, val_dataset, test_dataset

    def get(self, idx):
        """
        :param idx:
        :return:
        """
        if self.__is_instrument_graph:
            file_name = f'per_instrument_data.pt'
        else:
            file_name = f'data.pt'

        self.data, self.slices, _ = torch.load(osp.join(self.processed_dir, file_name))
        return self.data


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
