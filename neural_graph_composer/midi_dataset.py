import argparse
import logging
import os
import os.path as osp
import pathlib
import shutil
import warnings
from typing import Optional, List, Dict, Callable
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, extract_tar

from .midi_graph_builder import MidiGraphBuilder
from .midi_reader import MidiReader
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch.utils.data import Dataset


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


class MidiDataset(InMemoryDataset):
    """Create dataset from list of MIDI files

    per_graph_slit dictates if we want threat each instrument as separate graph
    we want to merge each to single graph.
    """
    _mirror = ['https://github.com/spyroot/neural-graph-composer/tree/main/data/processed']

    def __init__(self,
                 root,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 default_node_attr: str = 'attr',
                 midi_files: Optional[List[str]] = None,
                 default_webserver: Optional[str] = _mirror[0],
                 train_ratio: Optional[float] = 0.7,
                 val_ratio: Optional[float] = 0.15,
                 per_instrument_graph: Optional[bool] = True,
                 per_graph_slit: Optional[bool] = True,
                 default_midi_loc: str = "neural_graph_composer/dataset",
                 feature_vec_size: Optional[int] = 12,
                 velocity_num_buckets: Optional[int] = 8,
                 tolerance: float = 0.2,
                 filter_single_notes: Optional[bool] = False,
                 include_velocity: Optional[bool] = False,
                 do_split_mask: Optional[bool] = True,
                 remove_label: Optional[bool] = True,
                 do_sanity_check: Optional[bool] = False):
        """Default_midi_loc used to indicate a directory where all MIDI files.

            Example usage offline:

            raw_paths = [
            'data/raw/a_night_in_tunisia_2_jc.mid',
            'data/raw/a_night_in_tunisia_2_jc.mid'
            ]

            midi_dataset = MidiDataset(root="./data_test",
                               midi_files=raw_paths,
                               per_instrument_graph=False)

            It will crete data_test dir copy all files
            to raw folder as any dataset.

            Note it will always copy, so we have same behavior as online.
            If caller indicate files it will use this file and copy
            row and produce Dataset.

            Example usage online:  This will check raw dir
            and if files not present will download.

            midi_dataset = MidiDataset(root="./data_test",
                   per_instrument_graph=False)

            So you can first create and then call without
            it will essentially create custom dataset.

            midi_dataset = MidiDataset(root="./data_test",
                   midi_files=raw_paths,
                   per_instrument_graph=False)

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
        :param default_midi_loc: Default location where we search for a files.
        :param train_ratio: Training ratio (default 0.7).
        :param val_ratio: Validation ratio (default 0.15).
        :param do_split_mask will create masks for test , train.
        :param per_graph_slit: will compute and include train and test mask
        :param: do_sanity_check will do some sanity check including check for
                all node classes and inverse checks.
        :return: 'MidiDataset'
        :rtype: 'MidiDataset'
        """
        if not isinstance(root, str):
            raise TypeError("root must be a string.")
        if transform is not None and not callable(transform):
            raise TypeError("transform must be a callable.")
        if pre_transform is not None and not callable(pre_transform):
            raise TypeError("pre_transform must be a callable.")
        if pre_filter is not None and not callable(pre_filter):
            raise TypeError("pre_filter must be a callable.")
        if not isinstance(default_node_attr, str):
            raise TypeError("default_node_attr must be a string.")
        if midi_files is not None and not isinstance(midi_files, list):
            raise TypeError("midi files must be a list of strings.")
        if not isinstance(default_webserver, str):
            raise TypeError("default webserver must be a string.")
        if not isinstance(train_ratio, (float, int)):
            raise TypeError("train ratio must be a float or an integer.")
        if not isinstance(val_ratio, (float, int)):
            raise TypeError("validation ratio must be a float or an integer.")
        if not isinstance(per_instrument_graph, bool):
            raise TypeError("per instrument graph must be a boolean.")
        if not isinstance(per_graph_slit, bool):
            raise TypeError("per graph slit must be a boolean.")
        if not isinstance(default_midi_loc, str):
            raise TypeError("default midi loc must be a string.")
        if feature_vec_size is not None and not isinstance(feature_vec_size, int):
            raise TypeError("feature vec size must be an integer.")
        if velocity_num_buckets is not None and not isinstance(velocity_num_buckets, int):
            raise TypeError("velocity num buckets must be an integer.")
        if not isinstance(tolerance, float):
            raise TypeError("tolerance must be a float.")
        if not isinstance(filter_single_notes, bool):
            raise TypeError("filter single_notes must be a boolean.")
        if not isinstance(include_velocity, bool):
            raise TypeError("include velocity must be a boolean.")

        if midi_files is not None:
            if not isinstance(midi_files, list):
                raise ValueError("midi_files should be a list of strings.")
            if not all(isinstance(file_name, str) for file_name in midi_files):
                raise ValueError("All elements in midi_files should be strings.")

        if not (0 <= train_ratio <= 1):
            raise ValueError("train_ratio should be between 0 and 1.")

        if not (0 <= val_ratio <= 1):
            raise ValueError("val_ratio should be between 0 and 1.")

        if train_ratio + val_ratio > 1:
            raise ValueError("The sum of train_ratio and val_ratio "
                             "should be less than or equal to 1.")

        if feature_vec_size is not None and (feature_vec_size < 1 or feature_vec_size > 128):
            raise ValueError("feature_vec_size must be between 1 and 128 (inclusive)")

        if velocity_num_buckets is not None and velocity_num_buckets < 1:
            raise ValueError("velocity_num_buckets must be greater than or equal to 1")

        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)
        self.logger.setLevel(logging.WARNING)

        # p = pathlib.Path(root)
        # if not p.exists() or not p.is_dir():
        #     raise ValueError("Please indicate valid path to root directory.")
        # we set absolute for win32
        root = str(pathlib.Path(root).absolute())

        default_loc_path = pathlib.Path(default_midi_loc)
        if not default_loc_path.exists() or not default_loc_path.is_dir():
            raise ValueError("Please indicate valid path where to search for a midi files.")

        default_midi_loc = str(default_loc_path.absolute())

        # this value set if caller created
        # dataset from a list of files.
        self._offline = False

        # list offline files.
        self.files = []
        if midi_files is not None:
            self.files = midi_files
            self._offline = True

        self._remove_label = remove_label
        self._is_sanity_check = do_sanity_check
        self._do_split_mask = do_split_mask
        self._default_loc = Path(default_midi_loc).expanduser().resolve()
        self.__url = default_webserver
        self.__train_ratio = train_ratio
        self.__val_ratio = val_ratio
        self.__is_instrument_graph = per_instrument_graph
        self.__mask_instruments = per_graph_slit
        self.__hidden_channels = 64
        self._current_index = 0

        #
        self.node_attr_name = default_node_attr
        self.feature_vec_size = feature_vec_size
        self.velocity_num_buckets = velocity_num_buckets
        self.filter_single_notes = filter_single_notes
        self.include_velocity = include_velocity
        self.tolerance = tolerance
        self.default_trim_time = 3
        self.all_classes = set()

        self.data_list = None
        self.__data_list = None
        self._graph_builder = None
        self._num_instruments = {}

        logging.debug(f"default_node_attr:      {self.__train_ratio}")
        logging.debug(f"transform:              {self.__val_ratio}")
        logging.debug(f"per_instrument_graph:   {self.__is_instrument_graph}")
        logging.debug(f"pre_filter:             {pre_filter}")
        print(f"processed_file_names:           {self.processed_file_names}")

        self._graph_builder = MidiGraphBuilder(None)

        super().__init__(root, transform, pre_transform, pre_filter)

        for path in self.processed_paths:
            logging.debug(f"Using processed data at {path}")

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
        if self.__is_instrument_graph:
            print(f"Loading {self.processed_dir} {self.processed_file_names[0]}")
            data_path = osp.join(self.processed_dir, self.processed_file_names[0])
        else:
            print(f"Loading {self.processed_dir} {self.processed_file_names[1]}")
            data_path = osp.join(self.processed_dir, self.processed_file_names[1])

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

    @property
    def raw_file_names(self):
        """Method by default uses self._default_loc to get list of all midi files.
        if user provide list of files , it will return that list
        :return:
        """
        # module_dir = os.path.dirname(os.path.abspath(__file__))
        # midi_dir = os.path.join(module_dir, 'data', 'midi')
        if self.files is not None and len(self.files) > 0:
            return self.files
        else:
            # location we use to get list of all midi files
            midi_files_dir = self._default_loc
            return [f for f in os.listdir(midi_files_dir)
                    if os.path.isfile(os.path.join(midi_files_dir, f))
                    and f.endswith('.mid')
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
        return [f'per_instrument_'
                f'vel_{self.include_velocity}_'
                f'feature_size_{self.feature_vec_size}_'
                f'tolerance_{str(self.tolerance)}.pt',
                f'data_'
                f'vel_{self.include_velocity}_'
                f'feature_size_{self.feature_vec_size}_'
                f'tolerance_{str(self.tolerance)}.pt']

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
        """ Downloads the MIDI files from the given URL and extracts them to the `raw_dir`.
         If a list of files is provided, it copies the files to the `raw_dir`
         instead of downloading them.
        :return:
        """
        logging.debug(f"raw_file_names {self.raw_file_names}")
        if len(self.files) > 0 and self._offline:
            for i, raw_file in enumerate(self.raw_file_names):
                p = Path(raw_file).expanduser().resolve()
                if not p.exists():
                    raise RuntimeError(f"File not found: {raw_file}")
                if p.is_file():
                    dst_path = os.path.join(self.raw_dir, p.name)
                    self.files[i] = p.name
                    src_path = os.path.abspath(str(p))
                    logging.debug(f"Downloading {src_path} to {dst_path}")
                    try:
                        shutil.copy(src_path, dst_path)
                    except shutil.SameFileError:
                        warnings.warn(f"Data raw folder already containers {src_path} {dst_path}")
                        pass
        else:
            for raw_file in self.raw_file_names:
                logging.debug(f"Downloading {raw_file}")
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

        if self._graph_builder is None:
            print("Warning: self.graph_builder is None")

        for raw_path in self.raw_paths:
            print(f"Reading {raw_path}")
            try:
                # read file and construct graph
                midi_seqs = MidiReader.read(raw_path)
                if raw_path not in self._num_instruments:
                    self._num_instruments[raw_path] = midi_seqs.num_instruments()
                self._graph_builder.build(
                    midi_seqs,
                    default_trim_time=self.default_trim_time,
                    feature_vec_size=self.feature_vec_size,
                    velocity_num_buckets=self.velocity_num_buckets,
                    filter_single_notes=self.filter_single_notes,
                    tolerance=self.tolerance,
                    is_include_velocity=self.include_velocity,
                    is_per_instrument=True
                )

                # graph_builder output iterator, is_sanity_check if we want to do sanity
                # check during this phase.
                for midi_data in self._graph_builder.graphs(
                        is_sanity_check=self._is_sanity_check,
                        skip_label=self._remove_label):
                    self.all_classes.update(torch.unique(midi_data.y).tolist())
                    # first we apply pre-filter then apply mask
                    if self.pre_filter is not None and not self.pre_filter(midi_data):
                        continue

                    # split mask
                    if self._do_split_mask:
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

    @staticmethod
    def __sanity_checker(self, data, index_to_hash, hash_to_notes):
        """
        :return:
        """
        data_x = data.x[data.train_mask]
        data_y = data.y[data.train_mask]

        for i in range(data_x.shape[0]):
            node_features = data_x[i]
            original_index = data_y[i].item()
            hash_of_index = index_to_hash[original_index]
            original_set_of_notes = hash_to_notes[hash_of_index]
            original_set_tensor = torch.tensor(list(original_set_of_notes))
            original_set_zero = torch.zeros((data_x.shape[0],))
            original_set_tensor = torch.cat((original_set_tensor, original_set_zero), dim=0)[
                                  :data_x.shape[0]].unsqueeze(0)
            node_features = node_features.unsqueeze(0)

            sorted_node_features, _ = torch.sort(node_features)
            sorted_original_set_tensor, _ = torch.sort(original_set_tensor)
            if not torch.equal(sorted_node_features, sorted_original_set_tensor):
                print(f"Error for index {i}, hash {hash_of_index}, notes {original_set_of_notes}:")
                print(node_features, original_set_tensor)

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
        torch.save((_data, _slices, additional_data),
                   osp.join(self.processed_dir,
                            f'per_instrument_'
                            f'vel_{self.include_velocity}_'
                            f'feature_size_{self.feature_vec_size}_'
                            f'tolerance_{str(self.tolerance)}.pt')
                   )
        del self.__data_list
        self.__data_list = None
        # self._graph_builder = None

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
        torch.save((_data, _slices, additional_data),
                   osp.join(
                       self.processed_dir,
                       f'data_'
                       f'vel_{self.include_velocity}_'
                       f'feature_size_{self.feature_vec_size}_'
                       f'tolerance_{str(self.tolerance)}.pt'
                   ))
        del self.__data_list
        self.__data_list = None

        # release all memory
        del self._graph_builder
        self._graph_builder = None

        self.load_processed()

    def _mask_entire_graph(self):
        """Process and mask all graphs
        :return:
        """
        if self.__data_list is None:
            self.__data_list = []

        for raw_path in self.raw_paths:
            print(f"Reading {raw_path}")
            try:
                midi_seqs = MidiReader.read(raw_path)
                self._graph_builder.build(
                    midi_seqs,
                    default_trim_time=self.default_trim_time,
                    feature_vec_size=self.feature_vec_size,
                    velocity_num_buckets=self.velocity_num_buckets,
                    filter_single_notes=self.filter_single_notes,
                    tolerance=self.tolerance,
                    is_include_velocity=self.include_velocity,
                    is_per_instrument=False
                )
                graphs = self._graph_builder.graphs()
                if not graphs:
                    raise ValueError("No sub graphs found in graph builder.")

                midi_data = next(graphs)

                self.all_classes.update(torch.unique(midi_data.y).tolist())
                if self.pre_filter is not None and not self.pre_filter(midi_data):
                    continue

                # split mask
                if self._do_split_mask:
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

                del midi_seqs

            except KeyError as ker_err:
                print(ker_err)
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

    # def get(self, idx):
    #     """
    #     :param idx:
    #     :return:
    #     """
    #     if self.__is_instrument_graph:
    #         file_name = self.processed_file_names[0]
    #     else:
    #         file_name = self.processed_file_names[1]
    #
    #     self.data, self.slices, _ = torch.load(osp.join(self.processed_dir, file_name))
    #     return self.data

    def get_graph_id(self):
        return self.current_index

    def __getitem__(self, index):
        self._current_index = index
        return super().__getitem__(index)


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
