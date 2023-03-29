"""
Main interface consumed to get MIDI information from file to internal representation.
Currently, it only supports PrettyMIDI.

A class for constructing PyTorch Geometric graphs from MIDI data.
The `MidiGraphBuilder` generate graph from MIDI sequence and converts graph representation data
into PyG graphs, with the option to create separate graphs for each instrument.
The graph nodes represent musical events, such as notes or chords, and can include
features such as pitch and velocity. The class supports two types of node attribute encoding:
one-hot vectors and fixed-size float tensors. The `MidiGraphBuilder` also provides options
for controlling the feature size and tolerance for determining when notes
are considered simultaneous.

Author Mus spyroot@gmail.com
           mbayramo@stanford.edu

"""
import itertools
import logging
import math
import pathlib
import warnings
from collections import defaultdict
from enum import auto, Enum
from typing import Optional, Generator, Dict, Tuple, Iterator
from typing import Union, List, Any

import librosa
import networkx as nx
import numpy as np
import torch
from numpy import ndarray
from torch import Tensor
from torch_geometric.data import Data

from neural_graph_composer.midi.midi_note import MidiNote
from torch_geometric.data import Data as PygData
from networkx.classes.digraph import Graph

from neural_graph_composer.midi.midi_sequence import MidiNoteSequence
from neural_graph_composer.midi.midi_sequences import MidiNoteSequences
from neural_graph_composer.midi_reader import MidiReader


class NodeAttributeType(Enum):
    """This a type how we encode represent node attribute
    """
    Tensor = auto()
    OneHotTensor = auto()


class MidiGraphBuilder:
    """
    A class for constructing PyTorch Geometric graphs from MIDI data.
    The `MidiGraphBuilder` generate graph from MIDI sequence and converts graph representation data
    into PyG graphs, with the option to create separate graphs for each instrument.
    The graph nodes represent musical events, such as notes or chords, and can include
    features such as pitch and velocity. The class supports two types of node attribute encoding:
    one-hot vectors and fixed-size float tensors. The `MidiGraphBuilder` also provides options
    for controlling the feature size and tolerance for determining when notes
    are considered simultaneous.
    """
    ENCODINGS = {
        NodeAttributeType.Tensor: lambda n: torch.FloatTensor(list(n)),
        NodeAttributeType.OneHotTensor: lambda n: torch.nn.functional.one_hot(
            torch.FloatTensor(list(n)), num_classes=127)
    }

    __slots__ = ["logger", "is_include_velocity", "node_attr_type", "tolerance",
                 "_max_pitches", "feature_size", "node_attr_name",
                 "_notes_to_hash", "_hash_to_notes", "_hash_to_index", "_index_to_hash",
                 "pre_instrument", "_pyg_data", "_sub_graphs", "midi_sequences"]

    def __init__(
            self,
            midi_data: Union[MidiNoteSequence, MidiNoteSequences,
            Generator[MidiNoteSequence, None, None]] = None,
            is_instrument_graph: Optional[bool] = True,
            feature_size: int = 12,
            is_include_velocity: Optional[bool] = False,
            tolerance: Optional[float] = 0.2,
            node_attr_type: NodeAttributeType = NodeAttributeType.Tensor):

        """
        :param midi_data: midi_sequences object
        :param feature_size: The size of the feature vector to use for the graph embeddings.
        If None, the maximum MIDI note number (127) will be used as the upper bound for the feature size.
        By default, the feature size is 12, which means that each note or chord is represented
        as a tensor of shape (feature_size, ) if velocity is not included, or (2, feature_size)
        if velocity is included.

        :param is_instrument_graph (bool, optional):  If True, will create a graph for each instrument
        in the `MidiNoteSequence`. Otherwise, a single graph will be constructed using all instruments.
        Defaults to True.
        :param is_include_velocity: If True, velocity will be included as an attribute of the graph nodes.
        :param tolerance (float, optional): The maximum allowable difference between two note
        start times for them to be considered simultaneous. It main criterion we
        use to form a chord vs note. Default value is 0.5.
        """
        if midi_data is not None:
            if isinstance(midi_data, MidiNoteSequence):
                midi_data = MidiNoteSequences(midi_seq=[midi_data])
            if not isinstance(midi_data, MidiNoteSequences) or isinstance(midi_data, Generator):
                raise TypeError("midi_data must be an instance "
                                "of Generator, MidiNoteSequence or MidiNoteSequences")
        else:
            midi_data = None

        if not isinstance(tolerance, (int, float)) or tolerance < 0:
            raise ValueError("tolerance must be a non-negative number")

        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)
        self.logger.setLevel(logging.WARNING)

        self.is_include_velocity = is_include_velocity
        self.node_attr_type = node_attr_type
        self.tolerance = tolerance

        self._max_pitches = 127

        if feature_size is None:
            self.feature_size = 12
        elif isinstance(feature_size, int) and 0 < feature_size <= self._max_pitches:
            self.feature_size = feature_size
        else:
            raise ValueError("feature_size should be a positive integer less "
                             "than or equal to the maximum number "
                             "of pitches in MIDI (127)")

        # this a default name that we use for node attributes
        self.node_attr_name = ["attr"]

        self.feature_size = feature_size
        self.pre_instrument = is_instrument_graph
        self.midi_sequences = midi_data

        # all read-only properties. notes to hash and all mappings
        self._notes_to_hash = {}
        self._hash_to_notes = {}
        self._hash_to_index = {}
        self._index_to_hash = {}
        self._pyg_data = []
        self._sub_graphs = []

    @classmethod
    def from_file(cls, file_path: str,
                  feature_size: Optional[int] = 12,
                  is_include_velocity: Optional[bool] = False) -> 'MidiGraphBuilder':
        """Constructs a MidiGraphBuilder object from a MIDI file path.:
        :param cls:
        :param file_path: A string representing the path to the MIDI file to be processed.
        :param is_include_velocity: A boolean indicating whether to build a graph for each instrument in the MIDI file.
        :param feature_size: The size of the  feature vector to use for the PyTorch Geometric graph.
        :param is_include_velocity:
        :return: A MidiGraphBuilder object constructed from the specified MIDI file.
        """
        midi_file = pathlib.Path(file_path)
        if not midi_file.is_file():
            raise ValueError(f"Invalid file path: {file_path}")

        if midi_file.suffix not in ['.mid', '.midi']:
            raise ValueError(f"File {file_path} is not a MIDI file.")

        if not midi_file.exists():
            raise ValueError(f"File {file_path} does not exist.")

        midi_seq = MidiReader.read(file_path)
        return cls(midi_seq, feature_size=feature_size, is_instrument_graph=is_include_velocity)

    @classmethod
    def from_midi_sequence(cls,
                           midi_sequence: MidiNoteSequence,
                           feature_size: Optional[int] = 12,
                           is_include_velocity: Optional[bool] = True) -> 'MidiGraphBuilder':
        """Constructs a new instance of `MidiGraphBuilder`
        using a `MidiNoteSequence` object.

        :param cls:
        :param midi_sequence: A `MidiNoteSequence` object containing MIDI data to construct a graph from.
        :param is_include_velocity: If True, will create a graph for each instrument in the `MidiNoteSequence`.
                               Otherwise, a single graph will be constructed using all instruments.
        :param feature_size: The size of the feature vector to use for the graph embeddings.
        :return: A new instance of `MidiGraphBuilder`.
        :rtype: MidiGraphBuilder
        """
        if not isinstance(midi_sequence, MidiNoteSequence):
            raise ValueError("midi_sequence should be an instance of MidiNoteSequence")
        if not midi_sequence.notes or midi_sequence.notes == []:
            raise ValueError("MidiNoteSequence is empty")

        midi_sequences = MidiNoteSequences(midi_seq=midi_sequence)
        return cls(midi_sequences, feature_size=feature_size, is_instrument_graph=is_include_velocity)

    @property
    def sub_graphs(self) -> List[Graph]:
        """All graph add to internal list. Note when graph() generated emit
        graph , graph removed from internal list.
        :return:
        """
        return self._sub_graphs

    @property
    def pyg_data(self) -> List[PygData]:
        """All graph serialized to pyg_data. Note when generator emit graph
        Object released from memory.
        :return:
        """
        return self._pyg_data

    @classmethod
    def from_midi_sequences(cls, midi_sequences: MidiNoteSequences):
        """Creates a MidiNoteSequence from a list of MidiNote objects.
        :param midi_sequences:
        :return:
        """
        if not isinstance(midi_sequences, MidiNoteSequences):
            raise TypeError("midi_sequences must be an instance of MidiNoteSequences")
        graph_builder = cls(midi_sequences)
        return graph_builder

    @staticmethod
    def from_midi_networkx(
            midi_graph: Any,
            group_node_attrs: Optional[Union[List[str], all]] = None,
            group_edge_attrs: Optional[Union[List[str], all]] = None) -> PygData:
        """Convert a MIDI graph represented as a NetworkX graph
        to a PyTorch Geometric Data object. This simular to pyg from networkx but it has some critical
        changes because original semantically does something very different.

        :param midi_graph: The MIDI graph as a NetworkX graph object.
        :param group_node_attrs: A list of node attribute names to group into a single tensor.
                                 If set to 'all', all node attributes will be grouped into a tensor.
        :param group_edge_attrs: A list of edge attribute names to group into a single tensor.
                                 If set to 'all', all edge attributes will be grouped into a tensor.
        :return: A PyTorch Geometric Data object representing the MIDI graph.
        """
        # default attr we expect
        if group_node_attrs is None:
            group_node_attrs = ["attr"]

        if group_edge_attrs is None:
            group_edge_attrs = ["weight"]

        midi_graph = nx.convert_node_labels_to_integers(midi_graph)
        midi_graph = midi_graph.to_directed() if not nx.is_directed(midi_graph) else midi_graph

        if isinstance(midi_graph, (nx.MultiGraph, nx.MultiDiGraph)):
            edges = list(midi_graph.edges(keys=False))
        else:
            edges = list(midi_graph.edges)

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        data = defaultdict(list)

        if midi_graph.number_of_nodes() > 0:
            node_attrs = list(next(iter(midi_graph.nodes(data=True)))[-1].keys())
        else:
            node_attrs = {}

        if midi_graph.number_of_edges() > 0:
            edge_attrs = list(next(iter(midi_graph.edges(data=True)))[-1].keys())
        else:
            edge_attrs = {}

        for i, (_, feat_dict) in enumerate(midi_graph.nodes(data=True)):
            if set(feat_dict.keys()) != set(node_attrs):
                raise ValueError('Not all nodes contain the same attributes')
            for key, value in feat_dict.items():
                data[str(key)].append(value)

        for i, (_, _, feat_dict) in enumerate(midi_graph.edges(data=True)):
            if set(feat_dict.keys()) != set(edge_attrs):
                raise ValueError('Not all edges contain the same attributes')
            for key, value in feat_dict.items():
                key = f'edge_{key}' if key in node_attrs else key
                data[str(key)].append(value)

        # node attr
        if group_node_attrs is not None:
            for key, value in data.items():
                if key not in group_node_attrs:
                    continue
                try:
                    if isinstance(value, (tuple, list)) and all(isinstance(v, (Tensor, np.ndarray)) for v in value):
                        data[key] = value
                    elif isinstance(value, np.ndarray):
                        data[key] = torch.from_numpy(value)
                    elif isinstance(value, (tuple, list)) and all(isinstance(v, Tensor) for v in value):
                        logging.debug(
                            f" -> Adding data key {key} case two shape len {len(value)} data[key] len {data[key]}")
                        data[key] = torch.stack(value, dim=0)
                    elif isinstance(value, (tuple, list)) and all(isinstance(v, float) for v in value):
                        data[key] = torch.tensor(value).unsqueeze(1)
                    else:
                        logging.debug(f" -> Adding key {key} to data, values", value)
                        if isinstance(value, (tuple, list)) and all(isinstance(v, Tensor) for v in value):
                            data[key] = value
                        else:
                            data[key] = torch.tensor(value)
                except (ValueError, TypeError):
                    logging.warning("Expected type tensor")
                    pass

        data['edge_index'] = edge_index.view(2, -1)
        data = Data.from_dict(data)

        if group_node_attrs is not None:
            xs = []
            for key in group_node_attrs:
                x = data[key]
                x = x.view(-1, 1) if x.dim() <= 1 else x
                logging.debug(f"Node attribute '{key}' shape: {x.shape}")  # Add this line
                xs.append(x)
                del data[key]
            data.x = torch.cat(xs, dim=-1)

        if group_edge_attrs == 'all':
            group_edge_attrs = list(edge_attrs)

        if group_edge_attrs is not None:
            xs = []
            for key in group_edge_attrs:
                key = f'edge_{key}' if key in node_attrs else key
                x = torch.tensor(data[key])
                logging.debug(f"Edge attribute '{key}' shape: {x.shape}")
                x = x.view(-1, 1) if x.dim() <= 1 else x
                xs.append(x)
                del data[key]
            data.edge_attr = torch.cat(xs, dim=-1)

        if data.x is None and data.pos is None:
            data.num_nodes = midi_graph.number_of_nodes()

        return data

    @staticmethod
    def neighbors(midi_graph: nx.Graph, u: List[Union[MidiNote, int]]) -> Iterator:
        """Take list of pitch value or just int and construct a set. The hash of set is node
        and return all neighbors nodes of this hash.
        :param midi_graph: The MIDI graph in which to look for neighbors.
        :param u: A list of `MidiNote`s or integers representing pitch values.
        :return: An iterator over all neighbors of the given pitch set.
        """
        u_hash = 0
        if len(u) > 0:
            if isinstance(u[0], MidiNote):
                u_hash = hash(frozenset([n.pitch for n in u]))
            elif isinstance(u[0], int):
                u_hash = hash(frozenset(u))
            elif isinstance(u, int):
                u_hash = hash(frozenset([u]))
            else:
                raise ValueError("invalid type for node u")

        return midi_graph.neighbors(u_hash)

    @staticmethod
    def is_connected(
            midi_graph: nx.Graph,
            u: Union[List[Union[MidiNote, int]], int],
            v: Union[List[Union[MidiNote, int]], int]):
        """ Take a midi graph and check if u and v connected.
        u and v must a hash of set. and set it number of pitch values
        i.e.  {51, 61] etc.

        Usage:
        >>> mg_builder = MidiGraphBuilder.from_file("path/to/midi/file.mid")
        >>> mg = mg_builder.build()
        >>> u = hash(frozenset([60, 64, 67]))
        >>> v = hash(frozenset([60, 64, 68]))
        >>> mg_builder.is_connected(mg, u, v)
        True
        :param midi_graph:
        :param u: The hash of the set of pitch values for node u, or a list of `MidiNote`s or pitch values.
        :param v: The hash of the set of pitch values for node v, or a list of `MidiNote`s or pitch values.
        :return: True if u and v are connected, False otherwise.
        """
        if isinstance(u, list):
            if isinstance(u[0], MidiNote):
                u = hash(frozenset([n.pitch for n in u]))
            elif isinstance(u[0], int):
                u = hash(frozenset(u))
            else:
                raise ValueError("invalid type for node u")

        if isinstance(v, list):
            if isinstance(v[0], MidiNote):
                v = hash(frozenset([n.pitch for n in v]))
            elif isinstance(v[0], int):
                v = hash(frozenset(v))
            else:
                raise ValueError("invalid type for node v")
        return u in midi_graph.neighbors(v)

    @staticmethod
    def get_edge_connected(midi_graph: nx.Graph, u: List[int], v: List[int]):
        """Take two list of pitch u=[51,52] v=[61,62], if two sets connected return the edge.
        :param midi_graph: a midi graph
        :param u: List of pitches
        :param v: List of pitches
        :return: a tuple representing the edge if u and v are connected, otherwise None
        """
        hash_of_u = hash(frozenset(u))
        hash_of_v = hash(frozenset(v))
        if hash_of_u in midi_graph and hash_of_v in midi_graph:
            if midi_graph.has_edge(hash_of_u, hash_of_v):
                return hash_of_u, hash_of_v
        return None

    @staticmethod
    def merge_nodes(graph: nx.Graph, nodes: List[int]) -> int:
        """Merges a list of nodes in the graph into a single node.
        The new node will have an attribute `merged_nodes` that
        stores the original nodes that were merged.
        :param graph:
        :param nodes:
        :return:
        """
        # Create a new node representing the merged nodes
        new_node = max(graph.nodes) + 1
        graph.add_node(new_node, merged_nodes=nodes)

        # add edges from new node to all neighbors of the merged nodes
        for node in nodes:
            for neighbor in graph.neighbors(node):
                if neighbor not in nodes:
                    graph.add_edge(new_node, neighbor, **graph.edges[node, neighbor])

        # remove the original nodes from the graph
        for node in nodes:
            graph.remove_node(node)

        return new_node

    @staticmethod
    def connect_instruments(graph: nx.Graph, midi_seqs: MidiNoteSequences):
        """
        Connects sub-graphs of different instruments based on their overlapping notes.

        For each pair of neighboring nodes from different instruments, checks if their notes overlap in time
        and connects the sub-graphs they belong to if they do.

        :param graph: The MIDI graph to connect the sub-graphs of.
        :param midi_seqs: The MIDI note sequences corresponding to the MIDI graph.
        """
        # Create a dictionary mapping pitch and start time to the node it belongs to
        pitch_start_to_node = {}
        for node in graph.nodes:
            pitch_start = (graph.nodes[node]["label"], graph.nodes[node]["start_time"])
            pitch_start_to_node[pitch_start] = node

        # iterate over each pair of neighboring nodes from different instruments
        for u, v in graph.edges:
            if midi_seqs[u].instrument != midi_seqs[v].instrument:
                # check if the notes of the two neighboring nodes overlap in time
                u_notes = [n for n in midi_seqs[u].notes if n.start_time == graph.nodes[u]["start_time"]]
                v_notes = [n for n in midi_seqs[v].notes if n.start_time == graph.nodes[v]["start_time"]]
                for u_note in u_notes:
                    for v_note in v_notes:
                        if u_note.overlaps(v_note):
                            # if the notes overlap, connect the sub-graphs they belong to
                            u_pitch_start = (u_note.pitch, u_note.start_time)
                            v_pitch_start = (v_note.pitch, v_note.start_time)
                            u_node = pitch_start_to_node.get(u_pitch_start)
                            v_node = pitch_start_to_node.get(v_pitch_start)
                            if u_node is not None and v_node is not None:
                                graph.add_edge(u_node, v_node, weight=1.0)

    def build(self,
              midi_seqs: Optional[MidiNoteSequences] = None,
              default_trim_time: Optional[int] = 3,
              feature_vec_size: Optional[int] = 12,
              velocity_num_buckets: Optional[int] = 8,
              node_attr_type: NodeAttributeType = NodeAttributeType.Tensor,
              filter_single_notes: Optional[bool] = False,
              tolerance: float = 0.2,
              is_per_instrument: Optional[bool] = True,
              is_include_velocity: Optional[bool] = False):
        """
         Build a graph for one or more midi sequences.

        :param is_per_instrument: if False all instrument add to a single larger graph.
        :param tolerance:
        :param filter_single_notes:
        :param midi_seqs: A  MidiNoteSequence objects, If `None`, use the sequences passed to the constructor.
        :param default_trim_time: The time resolution to use for each note. For  example,
                                 if `default_trim_time=3`, notes played at `0.001` and`0.00001`
                                 would both be rounded to `0.000`

        :param feature_vec_size: size of feature vector. (Default 12)
        :param velocity_num_buckets: number of bucket for velocity. ( Default 8)
        :param is_include_velocity:  include or not velocity in feature vector.
        :param is_include_velocity:  include or not velocity information in feature vector. (Default False)
        :param node_attr_type:  The type of attribute to use for each node in the graph.
                                Can be either `NodeAttributeType.Tensor`
                                or `NodeAttributeType.OneHotTensor`.
        :return:
        """
        _midi_seq = None
        if midi_seqs is None and self.midi_sequences is None:
            raise ValueError("No MIDI sequences provided.")

        if not isinstance(default_trim_time, int):
            raise TypeError("default_trim_time must be an integer")

        if not isinstance(feature_vec_size, int):
            raise TypeError("feature_vec_size must be an integer")

        if not isinstance(velocity_num_buckets, int):
            raise TypeError("velocity_num_buckets must be an integer")

        _midi_seq = midi_seqs if midi_seqs is not None else self.midi_sequences
        large_graph = None

        total_nodes = 0

        for s in _midi_seq:
            # skip all drum instruments
            if _midi_seq[s].instrument.is_drum:
                continue
            if len(_midi_seq[s].notes) == 1:
                continue
            g = self.build_sequence(
                _midi_seq[s],
                feature_vec_size=feature_vec_size,
                velocity_num_buckets=velocity_num_buckets,
                node_attr_type=node_attr_type,
                filter_single_notes=filter_single_notes,
                tolerance=tolerance,
                is_include_velocity=is_include_velocity,
                g=large_graph if not is_per_instrument else None
            )

            logging.debug(f"Number of nodes: {len(g.nodes)}, "
                          f"Number of edges: {len(g.edges)}")

            if not g.nodes:
                continue
            has_attrs = all('attr' in g.nodes[n] for n in g)
            if not has_attrs:
                raise ValueError("Not all nodes have attributes in the graph")
            has_attrs = all('node_hash' in g.nodes[n] for n in g)
            if not has_attrs:
                raise ValueError("Not all nodes have attributes in the graph")
            has_attrs = all('label' in g.nodes[n] for n in g)
            if not has_attrs:
                logging.warning("Not all nodes have 'label' attribute in the graph.")

            # for each instrument construct graph
            if is_per_instrument:
                pyg_data = self.from_midi_networkx(g)
                self._pyg_data.append(pyg_data)
                self._sub_graphs.append(g)
                delta_nodes = len(g.nodes)
                total_nodes += delta_nodes
            else:
                # otherwise append to existing graph
                if large_graph is None:
                    large_graph = g
                    delta_nodes = len(g.nodes)
                    total_nodes += delta_nodes
                else:
                    num_existing_nodes = len(large_graph.nodes)
                    large_graph = nx.compose(large_graph, g)
                    delta_nodes = len(large_graph.nodes) - num_existing_nodes
                    total_nodes += delta_nodes

            logging.debug(f"Added {delta_nodes} new nodes to the "
                          f"graph, total number of nodes: {total_nodes}.")

        if not is_per_instrument and large_graph is not None:
            pyg_data = self.from_midi_networkx(large_graph)
            self._pyg_data.append(pyg_data)
            self._sub_graphs.append(large_graph)

        logging.debug(f"Total number of nodes in the dataset: {total_nodes}.")

    @staticmethod
    def same_start_time(n: MidiNote, tolerance=1e-6) -> bool:
        """Check if two MidiNote objects have the same start time
        within a certain tolerance.
        :param n: MidiNote object
        :param tolerance: tolerance value for comparing start times
        :return: bool
        """
        return abs(n.start_time - n.start_time) < tolerance

    @staticmethod
    def instrument_time_differences(midi_seq: MidiNoteSequence) -> ndarray:
        """Compute the time differences between consecutive notes in a
        MidiNoteSequence object for non-drum instruments.
        :param midi_seq: MidiNoteSequence object
        :return: list of time differences
        """
        start_times = sorted([n.start_time for n in midi_seq.notes
                              if not midi_seq.instrument.is_drum and len(midi_seq.notes) > 0])
        return np.diff(start_times)

    @staticmethod
    def rescale_velocities(
            velocities: List[int],
            target_min_velocity: Optional[int] = 32,
            target_max_velocity: Optional[int] = 127) -> List[int]:
        """Rescale a list of velocity values to fit within a specified range.
        # Example usage:
            velocities = [20, 26, 23, 20, 40, 20, 60, 20, 30, 20, 20, 30]
            rescaled_velocities = rescale_velocities(velocities)
            print(rescaled_velocities)

        :param velocities: list of velocity values
        :param target_min_velocity: minimum velocity value for rescaled values
        :param target_max_velocity: maximum velocity value for rescaled values
        :return: list of rescaled velocity values
        :return:
        """
        velocities = np.array(velocities)

        if len(velocities) == 1:
            rescaled_velocity = np.clip(
                np.round(velocities * (target_max_velocity - target_min_velocity) / 127 + target_min_velocity),
                target_min_velocity, target_max_velocity
            )[0]
            return [rescaled_velocity]

        # current min and max velocities
        current_min_velocity = np.min(velocities)
        current_max_velocity = np.max(velocities)

        denominator = current_max_velocity - current_min_velocity
        if denominator <= 0:
            rescaled_velocities = np.full_like(velocities, target_min_velocity)
        else:
            rescaled_velocities = ((velocities - current_min_velocity) * (
                    target_max_velocity - target_min_velocity) / denominator) + target_min_velocity

        # clip the rescaled velocities to the valid MIDI range (0 to 127)
        clipped_velocities = np.clip(rescaled_velocities, 0, 127).astype(np.int32)
        return clipped_velocities.tolist()

    @staticmethod
    def scale_relative_velocities(
            velocities: List[int], scaling_factor: Optional[float] = 1.0) -> List[int]:
        """Rescale a list of velocity values to fit within a specified range.
        where velocity very low.

        # Example usage:
            velocities = [20, 26, 23, 20, 40, 20, 60, 20, 30, 20, 20, 30]
            rescaled_velocities = rescale_velocities(velocities)
            print(rescaled_velocities)

        :param velocities: list of velocity values
        :param scaling_factor: scaling factor to apply to the velocity values
        :return: list of rescaled velocity values
        """
        velocities = np.array(velocities)
        # the average velocity
        avg_velocity = np.mean(velocities)
        # the ratio between each note's velocity and the average velocity
        velocity_ratios = velocities / avg_velocity
        # scale the velocities using the scaling factor
        scaled_velocities = velocity_ratios * scaling_factor * avg_velocity
        # clip
        clipped_velocities = np.clip(scaled_velocities, 0, 127).astype(int)

        return clipped_velocities.tolist()

    @staticmethod
    def compute_tolerance(midi_seq: MidiNoteSequence, percentile: Optional[int] = 95) -> float:
        """Compute the percentile time difference between consecutive
        notes in a MidiNoteSequence object.

        midi_notes = [
            MidiNote(pitch=60, start_time=0.0, end_time=0.5),
            MidiNote(pitch=62, start_time=0.5, end_time=1.0),
            MidiNote(pitch=64, start_time=1.5, end_time=2.0),
            MidiNote(pitch=67, start_time=2.0, end_time=2.5),
            MidiNote(pitch=69, start_time=2.5, end_time=3.0),
            MidiNote(pitch=71, start_time=3.5, end_time=4.0)
        ]
        midi_seq = MidiNoteSequence(midi_notes, instrument=0)
        tolerance = MidiGraphBuilder.compute_tolerance(midi_seq)
        print(tolerance)  # expected output: 0.5

        :param midi_seq: MidiNoteSequence object
        :param percentile: percentile value to compute for the time differences
        :return: float
        """
        time_differences = MidiGraphBuilder.instrument_time_differences(midi_seq)
        if len(time_differences) < 2:
            return 0.0
        tolerance_percentile = np.percentile(time_differences, percentile)
        return tolerance_percentile

    @staticmethod
    def compute_data(seq: MidiNoteSequence,
                     tolerance: float,
                     filter_single_notes: bool,
                     tolerance_auto: Optional[bool] = False,
                     precision: Optional[int] = 2) -> Tuple[Dict[float, List[MidiNote]], int]:
        """Compute the dictionary 'data', which maps start times to lists of notes for a given
        MidiNoteSequence, and compute the length of the longest sequence
        of notes played at the same time.

        :param precision:
        :param seq:  MidiNoteSequence a midi sequence for particular instrument.
        :param tolerance: fixed tolerance
        :param filter_single_notes:  will filter single notes from seq.
        :param tolerance_auto:  will compute tolerance for indirection in time.
        :return:
        """
        # compute tolerance based on percentile.
        if tolerance_auto:
            tolerance = MidiGraphBuilder.compute_tolerance(seq)
            tolerance = float(tolerance)

        # sorted_notes = sorted(seq.notes, key=lambda n: n.start_time)

        data = {}
        longest_note_sequence = 0
        for n in seq.notes:
            # all drum are skipped, we only care about instrument that produce harmony
            if n.is_drum or n.velocity == 0:
                continue

            assert n.start_time >= 0, "n.start_time must be non-negative"

            if tolerance == 0.0:
                adjusted_time_hash = n.start_time
            elif tolerance_auto and math.isfinite(tolerance):
                adjusted_time_hash = round(round(n.start_time / tolerance) * tolerance, precision)
            else:
                adjusted_time_hash = round(n.start_time / tolerance, precision) * tolerance

            assert adjusted_time_hash >= 0, "n.start_time must be non-negative"

            time_hash = hash(adjusted_time_hash)
            if time_hash not in data:
                data[time_hash] = []

            data[time_hash].append(n)
            if len(data[time_hash]) > longest_note_sequence:
                longest_note_sequence = len(data[time_hash])

        if filter_single_notes:
            # Filter out single notes
            data = {k: notes for k, notes in data.items() if len(notes) > 1}

        return data, longest_note_sequence

    @staticmethod
    def create_tensor(
            node_attr_type: NodeAttributeType,
            pitch_set,
            velocity_set: Optional[Union[set, frozenset, list]] = None,
            feature_vec_size: Optional[int] = 12,
            num_classes: Optional[int] = None,
            velocity_num_buckets: Optional[int] = 8) -> torch.Tensor:
        """Create tensor for each node, node might have a set of pitch attach to it or set of pitches
        and velocity where velocity is number of bucket.

        For example
        [60, 62, 64, 65, 67, 69, 71, 72]
        [1, 0, 0, 0, 0, 0, 0, 3]

        where the numbers in the velocity set correspond to the
        number of notes with velocity in the corresponding bucket.
        In this example, notes with velocity 0-9 are in bucket 0,
        notes with velocity 20-29 are in bucket 2, and notes
        with velocity 70-79 are in bucket 7, etc.

        Example usage:
        # create tensor for node with pitch set only
        node_tensor = create_tensor(NodeAttributeType.OneHotTensor, pitch_set={60, 62, 64})

        # create tensor for node with pitch and velocity sets
        node_tensor = create_tensor(
            NodeAttributeType.OneHotTensor,
            pitch_set={60, 62, 64},
            velocity_set={1, 3, 4, 5}, velocity_num_buckets=5
            )

        for example

        tensor([[64., 60., 62.,  0.],
                [ 1.,  2.,  3.,  0.]])

        :param node_attr_type: type of node attribute, can be either NodeAttributeType.Tensor
                               or NodeAttributeType.OneHotTensor
        :param pitch_set: a set of pitch values
        :param velocity_set: a set of velocity values
        :param feature_vec_size: size of the feature vector
        :param num_classes: number of classes for one-hot encoding
        :param velocity_num_buckets: number of buckets for velocity values
        :return: a tensor of size feature_vec_size x 2 (if velocity_set is not None)
                 or feature_vec_size x num_classes
        :return:
        """
        # in case we want move to torch bucketize. need to test this before.
        # velocity_labels = torch.FloatTensor(list(velocity_set))
        # velocity_buckets = torch.linspace(start=0, end=127, steps=velocity_num_buckets)
        # velocity_bucket_indices = torch.bucketize(velocity_labels, boundaries=velocity_buckets)
        # velocity_tensor = torch.tensor(velocity_bucket_indices).unsqueeze(1).float()
        # print("vec vecttor", velocity_tensor)
        # print("vec velocity_set", velocity_set)

        if node_attr_type == NodeAttributeType.Tensor:
            # print(f" velocity_set set {velocity_set}")
            pitch_attr = torch.FloatTensor(list(pitch_set))
            velocity_attr = torch.FloatTensor(velocity_set) if velocity_set is not None and not np.isscalar(
                velocity_set) else None

            if pitch_attr.shape[0] > feature_vec_size:
                pitch_attr = pitch_attr[:feature_vec_size]
            if velocity_set is not None:
                pitch_tensor = torch.zeros(feature_vec_size, dtype=torch.float)
                vel_tensor = torch.zeros(feature_vec_size, dtype=torch.float)
                pitch_tensor[:pitch_attr.shape[0]] = pitch_attr
                vel_tensor[:velocity_attr.shape[0]] = velocity_attr
                pitch_tensor = pitch_tensor.unsqueeze(0)
                vel_tensor = vel_tensor.unsqueeze(0)
                assert torch.all(pitch_tensor >= 0), "pitch_tensor must be non-negative"
                assert torch.all(vel_tensor >= 0), "vel_tensor must be non-negative"
                new_x = torch.cat((pitch_tensor, vel_tensor), dim=0)
            else:
                new_x = torch.zeros(feature_vec_size, dtype=torch.float)
                new_x[:pitch_attr.shape[0]] = pitch_attr
        elif node_attr_type == NodeAttributeType.OneHotTensor:
            if num_classes is None:
                num_classes = len(pitch_set)
            if velocity_set is not None:
                pitch_labels = torch.FloatTensor(list(pitch_set)).long()
                velocity_labels = torch.FloatTensor(list(velocity_set))
                pitches_one_hot = torch.nn.functional.one_hot(pitch_labels, num_classes=num_classes)
                velocity_labels = velocity_labels.unsqueeze(1)
                new_x = torch.cat([pitches_one_hot, velocity_labels], dim=-1)
            else:
                labels_tensor = torch.FloatTensor(list(pitch_set)).long()
                new_x = torch.nn.functional.one_hot(
                    labels_tensor, num_classes=num_classes
                )
        else:
            raise ValueError("Unknown encoder type")

        return new_x

    def build_sequence(
            self,
            seq: MidiNoteSequence,
            feature_vec_size: Optional[int] = 12,
            velocity_num_buckets: Optional[int] = 8,
            node_attr_type: NodeAttributeType = NodeAttributeType.Tensor,
            filter_single_notes: bool = False,
            tolerance: float = 0.2,
            is_include_velocity: Optional[bool] = False,
            g: Optional[nx.DiGraph] = None) -> nx.DiGraph:

        """Build a graph for single midi sequence for particular instrument.

        If we need to merge all instruments into a single graph, the caller should
        use the `build` method which will merge all graphs to a single large graph.

        - First, we sort all notes based on start time and build a group.
        - A group is a `frozenset`, so no pitch value has duplicates.
        - We use all 12 octave.
        - We have two options for how we compute tolerance:
            - The first option is to use a fixed size tolerance.
              This option might not produce good results for all cases since it
              may group notes that are not meant to be played at the same time, such as notes played
             one after another that do not form a chord.

           - The second option is to compute tolerance based on percentile.
             We estimate the percentile and use this as tolerance, so if two
             notes should be played at the same time (perfectly, not human time), we fix the error
            and have a good representation of chords vs notes that are separate sequences of notes.

           - A group might be larger than the feature vector size, especially for
             classical pieces with notes played very tightly in time.
             We split groups into subgroups.
           - After we compute a group, we fix the velocity for each note.

            - The number of velocities must correspond to the number of notes in the group (set).

            - If in the original group we had two notes with the same pitch value, e.g. C0 and C0, after
             we set them, become just one note, hence we choose the maximum velocity.

            - In many pieces, especially classical music, velocity is a very important factor
            that captures nuances, but in many MIDI files, values are in the range 20-30,
            hence we rescale all values so that the lower bound is 32 and the upper bound is 127.

            - For each subgroup (which in most cases is just one set of 12 notes),
            we compute a feature vector consisting of pitch values [56, 60, ...] and the respective velocity vector.

            - We compute a hash for a node. A node hash is `HASH(PITCH_GROUP)`.

            - If the new hash is not yet in graph G, we add it to the graph and add a
              self-edge with weight 1.

            - If the hash is already in the graph, we increment the self-weight by 1
             (i.e. capture importance to self).

            - We add an edge from the node before the current node to the current node.
             This captures the transition from note to note or chord to chord.

            - If A and B are already connected, we increment their weight to capture importance.

            - In all edge cases, the method always returns a graph. If there are no notes,
             it will return an empty graph so that the upper-layer method can use an iterator structure.

        For example, right now the `build` method could technically be linked to the `graph`
        method and emit a graph by graph, so it would build and emit a generator, so we wouldn't
        need to store an internal representation in a list. This would save memory.

        :param seq: note seq object that store information about note seq.
        :param feature_vec_size: dictate a feature vector size.
        :param g: if you need add nodes to existing graph.
        :param filter_single_notes: filter out single notes if True, otherwise include them in the graph
        :param tolerance: dedicates tolerance between notes drift.
        :param filter_single_notes:  Will filter all single note. so
                                     final grap will have nodes only with more than > 1 note
        :param node_attr_type: dictates how we want re-present a node attribute.
                               as Tensor fixed size or as One hot vector
        :param feature_vec_size: dictate a maximum size of tensor.
                                  for example note play at 0.001 and next note .00001
        :param velocity_num_buckets: some pitch are loud some are not.
                                     We use bins to present that.
                                     Midi vel is value from 0 to 255.
                                     We use 8 bins to represent that
        :param is_include_velocity:   include velocity or not
        :return:
        """
        if not isinstance(seq, MidiNoteSequence):
            raise TypeError(f"midi_sequences must be an instance of "
                            f"MidiNoteSequence but received {type(seq)}")
        if feature_vec_size is not None:
            if not isinstance(feature_vec_size, int) or feature_vec_size <= 0 or feature_vec_size > 128:
                raise ValueError("Feature vector size must be an integer between 1 and 128.")
        if velocity_num_buckets is not None:
            if not isinstance(velocity_num_buckets, int) or velocity_num_buckets <= 0:
                raise ValueError("Number of velocity buckets must be a positive integer.")
        if not isinstance(node_attr_type, NodeAttributeType):
            raise ValueError("Node attribute type must be a NodeAttributeType object.")
        if not isinstance(filter_single_notes, bool):
            raise ValueError("The flag 'filter_single_notes' must be a boolean.")
        if not isinstance(tolerance, float) or tolerance < 0 or tolerance > 1:
            raise ValueError("Tolerance must be a float between 0 and 1.")
        if is_include_velocity is not None and not isinstance(is_include_velocity, bool):
            raise ValueError("The flag 'is_include_velocity' must be a boolean.")

        data, longest_note_sequence = self.compute_data(
            seq, tolerance, filter_single_notes, tolerance_auto=True)

        if not data:
            if g is None:
                return nx.DiGraph()
            else:
                return g

        sorted_keys = list(data.keys())
        sorted_keys.sort()
        if not sorted_keys:
            if g is None:
                return nx.DiGraph()
            else:
                return g

        # Create directed graph.
        #  - a chord that already played will point itself.
        #    i.e. if we play chord 5 time it edge to self with respected weight
        # -  next chord that play after prev time step connected back to chord at t-1
        #    note we don't care about time, we only care how chords connected
        if g is None:
            midi_graph = nx.DiGraph()
            last_node_hash = None
        else:
            midi_graph = g
            last_node_hash = list(g.nodes())[-1]

        mapping = {}
        # For each chord that hold set of pitches.
        # ie notes played same time form a node.
        node_weights = {}
        for k in sorted_keys:
            if k not in data:
                continue
            notes = data[k]

            # a pitch a set of pitches and velocity for each pitch
            # 1033w_hungarian_rhapsody_12_(nc)smythe.mid is good example.
            # feature vector even with very tight timing we get 18-20 notes.
            pitch_group = frozenset(n.pitch for n in notes)
            pitch_sliced = []
            if len(pitch_group) > feature_vec_size:
                pitch_iter = iter(pitch_group)
                while True:
                    slice_iter = itertools.islice(pitch_iter, feature_vec_size)
                    slice_items = list(slice_iter)
                    if not slice_items:
                        break
                    pitch_sliced.append(frozenset(slice_items))
            else:
                pitch_sliced = [pitch_group]

            for pitch_sub_group in pitch_sliced:
                velocity = None
                # compute velocity for each group.
                #   note each group reduce so number of velocity might be >
                #   is larger than all subgroup since we group based on set.
                if is_include_velocity:
                    max_velocity = {}
                    for n in notes:
                        if n.pitch not in pitch_sub_group:
                            continue
                        if n.pitch not in max_velocity:
                            max_velocity[n.pitch] = n.velocity
                        else:
                            max_velocity[n.pitch] = max(max_velocity[n.pitch], n.velocity)
                    velocity = [max_velocity[p] for p in pitch_sub_group]
                    velocity = MidiGraphBuilder.rescale_velocities(velocity)

                pitch_names = frozenset(librosa.midi_to_note(n.pitch) for n in notes)
                # a hash of set is node  {C0, F0, E0} respected MIDI num {x,y,z} form a hash
                # we compute hash if group > feature_size we do for each group.
                new_node_hash = hash(pitch_sub_group)
                # we add hash for a given pitch_set to dict,
                # so we can recover if we need 2
                if new_node_hash not in self._notes_to_hash:
                    self._notes_to_hash[pitch_sub_group] = new_node_hash
                    self._hash_to_notes[new_node_hash] = pitch_sub_group

                if new_node_hash not in self.hash_to_index:
                    current_length = len(self._hash_to_notes)
                    self._hash_to_index[new_node_hash] = current_length
                    self._index_to_hash[current_length] = new_node_hash

                # mapping map hash to pitch name
                mapping[new_node_hash] = pitch_names
                if node_attr_type not in self.ENCODINGS:
                    raise ValueError("Unknown encoder type")

                new_x = self.create_tensor(
                    node_attr_type,
                    pitch_sub_group,
                    velocity_set=velocity,
                    feature_vec_size=feature_vec_size,
                    num_classes=127,
                    velocity_num_buckets=velocity_num_buckets
                )

                if last_node_hash is None:
                    midi_graph.add_node(
                        new_node_hash, attr=new_x,
                        label=new_node_hash, node_hash=new_node_hash
                    )
                    # self edge.
                    midi_graph.add_edge(new_node_hash, new_node_hash, weight=1.0)
                else:
                    # if node already connected update weight
                    if midi_graph.has_edge(new_node_hash, last_node_hash):
                        if 'weight' not in midi_graph[new_node_hash][last_node_hash]:
                            warnings.warn(f"new hash {new_node_hash} has no weight to {last_node_hash}")
                        midi_graph[new_node_hash][last_node_hash]['weight'] += 1.0
                    elif midi_graph.has_edge(last_node_hash, new_node_hash):
                        if 'weight' not in midi_graph[last_node_hash][new_node_hash]:
                            warnings.warn(f"{last_node_hash} has no weight to {last_node_hash}")
                        midi_graph[last_node_hash][new_node_hash]['weight'] += 1.0
                    else:
                        midi_graph.add_node(new_node_hash, attr=new_x, label=pitch_names, node_hash=new_node_hash)
                        midi_graph.add_edge(last_node_hash, new_node_hash, weight=1.0)
                        node_weights[new_node_hash] = {last_node_hash: 1.0}
                last_node_hash = new_node_hash

        if node_attr_type == NodeAttributeType.OneHotTensor:
            for node in midi_graph.nodes:
                midi_graph.nodes[node]["attr"] = midi_graph.nodes[node]["attr"]

        return midi_graph

    @property
    def hidden(self):
        return self.hidden

    def __iter__(self):
        """Generator for iterating over the graphs."""
        yield from self.graphs()

    @property
    def hash_to_index(self) -> Dict[int, int]:
        """ A read-only dictionary mapping hash values to their respective indices.
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

    def graphs(self,
               skip_label: Optional[bool] = False,
               is_sanity_check: Optional[bool] = False) -> Generator[PygData, None, None]:
        """Generator that yields the PygData objects for each sub-graph.

        :param: skip_label will skip label in PygData object if you need
        :param: is_sanity_check does a reverse check for y indices.
        :return: a generator of PygData objects for each sub-graph
        """
        if not self._hash_to_notes:
            raise ValueError("hash to notes is empty.")

        # check idx for each hash value.
        if is_sanity_check:
            for node_hash in self._hash_to_notes:
                assert node_hash in self.hash_to_index

        for g in self._pyg_data or []:
            node_hash = [self._hash_to_index[node_hash] for node_hash in g.node_hash]
            target_indices = [self._hash_to_index[hash_val] for hash_val in g.node_hash]

            g.label = node_hash
            g.y = torch.tensor(target_indices, dtype=torch.long)

            # reverse sanity-check
            if is_sanity_check:
                original_labels = [self._index_to_hash[label.item()] for label in g.label]
                original_target_indices = [self._index_to_hash[index] for index in target_indices]
                original_y_values = [self._index_to_hash[y.item()] for y in g.y]
                original_notes = [self._hash_to_notes[h] for h in original_labels]

                for h in original_labels:
                    assert h in self._hash_to_notes, f"Note {h} is not in self.hash_to_notes."
                for h in original_target_indices:
                    assert h in self._hash_to_notes, f"hash {hash} is not in self.hash_to_notes."
                for h in original_y_values:
                    assert h in self._hash_to_notes, f"hash {hash} is not in self.hash_to_notes."

                for note in original_notes:
                    if isinstance(note, frozenset):
                        assert note in self._notes_to_hash, f"Note {note} is not in hash."

            if skip_label:
                g.label = None

            yield g
            del g

            del self._pyg_data[:]
            del self._sub_graphs[:]
            self._pyg_data = []
            self._sub_graphs = []
