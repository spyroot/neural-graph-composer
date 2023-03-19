import argparse
import logging
import os.path as osp
from typing import Optional, List

import numpy as np
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import download_url

from .midi_graph_builder import MidiGraphBuilder
from .midi_reader import MidiReader


class MidiDataset(InMemoryDataset):
    """Create dataset from list of MIDI files
    """

    def __init__(self, root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 default_node_attr='attr',
                 file_names: Optional[List[str]] = None,
                 default_webserver: Optional[str] = 'http://localhost:9000',
                 train_ratio: Optional[float] = 0.7,
                 val_ratio: Optional[float] = 0.15,
                 per_instrument_graph: Optional[bool] = True,
                 per_graph_slit: Optional[bool] = True,
                 ):
        """
        :param root:
        :param transform:
        :param pre_transform:
        :param pre_filter:
        :param default_node_attr:
        :param default_webserver:
        :param train_ratio:
        :param val_ratio:
        :param per_instrument_graph:
        :param per_graph_slit:
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

        self.url = default_webserver
        self.node_attr_name = default_node_attr
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.per_instrument_graph = per_instrument_graph
        self.per_graph_slit = per_graph_slit
        self.hidden_channels = 64

        print("called")
        super().__init__(root, transform, pre_transform, pre_filter)
        self.node_attr_name = default_node_attr

    @property
    def raw_file_names(self):
        return ['midi_test01.mid', 'midi_test02.mid', 'midi_test03.mid']

    @property
    def processed_file_names(self):
        return ['midi_test01.mid.pt', 'midi_test02.mid.pt', 'midi_test03.mid.pt']

    def compute_statistics(self):
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

        # zip_file_name = "midi_files.zip"
        # download_url(f"{self.url}/{zip_file_name}", self.raw_dir)

        for raw_file in self.raw_file_names:
            print(f"Downloading {raw_file}")
            download_url(f"{self.url}/{raw_file}", self.raw_dir)

            # with zipfile.ZipFile(os.path.join(self.raw_dir, zip_file_name), "r") as zip_ref:
            #     zip_ref.extractall(self.raw_dir)
            # os.remove(os.path.join(self.raw_dir, zip_file_name))

    def process_mask_per_graph(self):
        """Process and mask each graphs
        :return:
        """
        idx = 0
        for raw_path in self.raw_paths:
            print(f"Reading {raw_path}")
            try:
                # read file and construct graph
                midi_seqs = MidiReader.read(raw_path)
                # we build per instrument
                graph_builder = MidiGraphBuilder(
                    midi_seqs, per_instrument=self.per_instrument_graph)
                graph_builder.build()

                # graph_builder output iterator
                for midi_data in graph_builder.graphs():
                    # first we apply pre-filter then apply mask
                    if self.pre_filter is not None and not self.pre_filter(midi_data):
                        continue

                    # split mask
                    num_nodes = midi_data.x.size(0)
                    train_ratio, val_ratio = 0.7, 0.15
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

                    torch.save(midi_data, osp.join(self.processed_dir, f'data_{idx}.pt'))
                    idx += 1
            except KeyError as ker_err:
                print(f"Error in file {raw_path} error: {ker_err}")

    def process_mask_entire_graph(self):
        """Process and mask all graphs
        :return:
        """
        all_data = []
        for raw_path in self.raw_paths:
            print(f"Reading {raw_path}")
            midi_seqs = MidiReader.read(raw_path)

            # we build per instrument
            graph_builder = MidiGraphBuilder(
                midi_seqs, per_instrument=self.per_instrument_graph)

            #
            graph_builder.build()
            midi_data = graph_builder.pyg_data()

            # first we apply pre-filter then apply mask
            if self.pre_filter is not None and not self.pre_filter(midi_data):
                continue

            all_data.append(midi_data)
            # first we apply pre-filter then apply mask
            if self.pre_filter is not None and not self.pre_filter(midi_data):
                continue

        # we mask all graph
        num_graphs = len(all_data)
        train_ratio, val_ratio = 0.7, 0.15
        train_size = int(train_ratio * num_graphs)
        val_size = int(val_ratio * num_graphs)

        indices = np.random.permutation(num_graphs)
        for idx, data in enumerate(all_data):
            # training, validation, and testing masks
            data.train_mask = torch.tensor([idx in indices[:train_size]])
            data.val_mask = torch.tensor([idx in indices[train_size:train_size + val_size]])
            data.test_mask = torch.tensor([idx in indices[train_size + val_size:]])
            # apply pre transform
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            # save each separately
            torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))

    def process(self):
        """We compute the mask for each graph separately,
        since  graphs have different sizes or structures.
        This way, you ensure that you have a balanced split for each graph,
        allowing your model to learn and validate on a variety of examples.
        :return:
        """
        if self.per_graph_slit:
            self.process_mask_per_graph()
        else:
            self.process_mask_entire_graph()

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data

