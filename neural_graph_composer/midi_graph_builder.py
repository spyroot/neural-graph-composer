"""
Main interface consumed to get MIDI information from file to internal representation.
Currently, it only supports PrettyMIDI.

Author Mus spyroot@gmail.com
           mbayramo@stanford.edu

"""
import logging
from collections import defaultdict
from enum import auto, Enum
from typing import Optional
from typing import Union, List, Any

import librosa
import networkx as nx
import torch
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
    """
    ENCODINGS = {
        NodeAttributeType.Tensor: lambda n: torch.FloatTensor(list(n)),
        NodeAttributeType.OneHotTensor: lambda n: torch.nn.functional.one_hot(
            torch.FloatTensor(list(n)), num_classes=127)
    }

    def __init__(self,
                 midi_data: Union[MidiNoteSequence, MidiNoteSequences],
                 per_instrument: Optional[bool] = True,
                 hidden_feature_size: int = 64):
        """
        :param midi_data: midi_sequences object
        :param per_instrument:  Will build graph for each instrument.
        """
        if isinstance(midi_data, MidiNoteSequence):
            midi_data = MidiNoteSequences(midi_seq=[midi_data])

        if not isinstance(midi_data, MidiNoteSequences):
            raise TypeError("midi_data must be an instance of MidiNoteSequence or MidiNoteSequences")

        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)
        self.logger.setLevel(logging.WARNING)

        # this a default name that we use for node attributes
        self.node_attr_name = ["attr"]
        # in case we need map hashed value back to midi notes set
        self.hash_to_midi_set = {}
        self.midi_sequences = midi_data
        #
        self.hidden_feature_size = hidden_feature_size

        #
        self.pre_instrument = per_instrument
        # if per_instrument is true and number of instrument is > 0 each wil have own graph
        self._pyg_data = []
        self._sub_graphs = []

    @property
    def sub_graphs(self) -> List[Graph]:
        return self._sub_graphs

    @property
    def pyg_data(self) -> List[PygData]:
        return self._pyg_data

    @classmethod
    def from_midi_sequences(cls, midi_sequences: MidiNoteSequences):
        """Creates a MidiNoteSequence from a list of MidiNote objects.
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
        """This simular to pyg from networkx but it has some critical
        changes because original semantically does something very different.

        :param midi_graph:
        :param group_node_attrs:
        :param group_edge_attrs:
        :return:
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

                if isinstance(value, (tuple, list)) and isinstance(value[0], Tensor):
                    logging.debug(f" -> Adding data key {key} case two shape len {len(value)} data[key] len {data[key]}")
                    data[key] = torch.stack(value, dim=0)
                else:
                    try:
                        logging.debug(f" -> Adding key {key} to data, values", value)
                        if isinstance(value, torch.Tensor):
                            data[key] = value
                        else:
                            data[key] = torch.tensor(value)
                    except (ValueError, TypeError):
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

        if group_edge_attrs is all:
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
    def neighbors(midi_graph, u: List[Any]):
        """Take list of pitch value and construct a set.
        Hash and return based on hash all neighbors of given pitch set
        :param midi_graph:
        :param u:
        :return:
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
    def nodes_connected(midi_graph, u, v):
        """ Take a midi graph and check if u and v connected.
        u and v must a hash of set. and set it number of pitch values
        i.e.  {51, 61] etc.

        :param midi_graph:
        :param u:
        :param v:
        :return:
        """
        return u in midi_graph.neighbors(v)

    @staticmethod
    def get_edge_connected(midi_graph, u: List[int], v: List[int]):
        """Take two list of pitch u=[51,52] v=[61,62], if two sets connected return edge.
        :param midi_graph: a midi graph
        :param u: List of pitches
        :param v: List of pitches
        :return:
        """
        hash_of_u = hash(frozenset(u))
        hash_of_v = hash(frozenset(v))
        return midi_graph.get_edge_data(hash_of_u, hash_of_v)

    def build(self,
              default_trim_time: Optional[int] = 3,
              max_vector_size: Optional[int] = 5,
              velocity_num_buckets: Optional[int] = 8,
              node_attr_type: NodeAttributeType = NodeAttributeType.Tensor,
              is_debug: Optional[bool] = False):
        """
        :param default_trim_time:
        :param max_vector_size:
        :param velocity_num_buckets:
        :param node_attr_type:
        :param is_debug:
        :return:
        """

        for s in self.midi_sequences:
            g = self.build_sequence(
                self.midi_sequences[s], default_trim_time=default_trim_time, max_vector_size=max_vector_size,
                velocity_num_buckets=velocity_num_buckets, node_attr_type=node_attr_type,
                is_debug=is_debug
            )

            # g = nx.convert_node_labels_to_integers(g)
            pyg_data = self.from_midi_networkx(g)
            self._pyg_data.append(pyg_data)
            self._sub_graphs.append(g)

    def build_sequence(
            self,
            seq: MidiNoteSequence,
            default_trim_time: Optional[int] = 3,
            max_vector_size: Optional[int] = 5,
            velocity_num_buckets: Optional[int] = 8,
            node_attr_type: NodeAttributeType = NodeAttributeType.Tensor,
            is_debug: Optional[bool] = False):
        """Build a graph for single midi sequence for particular instrument.
        If we need merge all instrument to a single graph, caller
        need use build method that will merge all graph to a single large graph.

        :param is_debug: for debug purpose will move out
        :param node_attr_type: dictates how we want re-present a node attribute.
                               as Tensor fixed size or as One hot vector
        :param max_vector_size: dictate a maximum size of tensor.
        :param seq:
        :param default_trim_time: dictate time resolution
                                  for example note play at 0.001 and next note .00001
        :param velocity_num_buckets: some pitch are loud some are not.
                                     We use bins to present that.
                                     Midi vel is value from 0 to 255.
                                     We use 8 bins to represent that
        :return:
        """
        if not isinstance(seq, MidiNoteSequence):
            raise TypeError(f"midi_sequences must be an instance of "
                            f"MidiNoteSequence but received {type(seq)}")

        # sort all pitches played at same time.
        # we truncated float time. Long notes that span different chord
        # will be present in different node, and it ok
        data = {}
        # read seq of notes and form a hash list where
        # key is start time and value a note.
        longest_note_sequence = 0
        for n in seq.notes:
            # all drum are skipped, we only care about instrument
            # produce harmony and velocity > 0
            if n.is_drum or n.velocity == 0:
                continue
            time_hash = round(n.start_time, default_trim_time)
            if time_hash not in data:
                data[time_hash] = []
            data[time_hash].append(n)
            if len(data[time_hash]) > longest_note_sequence:
                longest_note_sequence = len(data[time_hash])

        # sort by time
        sorted_keys = list(data.keys())
        sorted_keys.sort()

        # Create directed graph.
        #  - a chord that already played will point itself.
        #    i.e. if we play chord 5 time it edge to self with respected weight
        # -  next chord that play after prev time step connected back to chord at t-1
        #    note we don't care about time, we only care how chords connected
        midi_graph = nx.DiGraph()
        last_node_hash = None
        last_node_name = ""
        mapping = {}

        # For each chord that hold set of pitches.
        # ie notes played same time form a node.
        for k in sorted_keys:
            notes = data[k]
            # a pitch a set of pitches and velocity for each pitch
            pitch_set = frozenset(n.pitch for n in notes)
            pitch_vel_set = frozenset(n.velocity // velocity_num_buckets for n in notes)

            # pitch name are attribute of node
            pitch_names = frozenset(librosa.midi_to_note(n.pitch) for n in notes)
            # a hash of set is node
            # i.e  {C0, F0, E0} respected MIDI num {x,y,z} form a hash
            new_node_hash = hash(pitch_set)
            # we add hash for a given pitch_set to dict,
            # so we can recover if we need 2
            if new_node_hash not in self.hash_to_midi_set:
                self.hash_to_midi_set[pitch_set] = new_node_hash

            # mapping map hash to pitch name
            mapping[new_node_hash] = pitch_names
            if node_attr_type not in self.ENCODINGS:
                raise ValueError("Unknown encoder type")

            # new_x = self.ENCODINGS[node_attr_type](pitch_set)
            # pitch_attr = Encodings.to_tensor(pitch_set, node_attr_type, max_vector_size)
            # pitch_attr = self.ENCODINGS[node_attr_type](pitch_set)

            if node_attr_type == NodeAttributeType.Tensor:
                pitch_attr = torch.FloatTensor(list(pitch_set))
                new_x = torch.zeros(max_vector_size)
                new_x[:pitch_attr.shape[0]] = pitch_attr
            elif node_attr_type == NodeAttributeType.OneHotTensor:
                labels_tensor = torch.FloatTensor(list(pitch_set))
                new_x = torch.nn.functional.one_hot(labels_tensor, num_classes=127)
            else:
                raise ValueError("Unknown encoder type")

            if last_node_hash is None:
                midi_graph.add_node(new_node_hash, attr=new_x, label=new_node_hash, node_hash=new_node_hash)
                # print(new_node_hash)
                last_node_hash = new_node_hash
                last_node_name = pitch_names
            else:
                # if node already connected update weight
                if midi_graph.has_edge(new_node_hash, last_node_hash):
                    midi_graph[new_node_hash][last_node_hash]['weight'] += 1
                    # if self.nodes_connected(midi_graph, new_node_hash, last_node_hash):
                    #     uv = midi_graph.get_edge_data(last_node_hash, new_node_hash)
                    #     if uv is not None:
                    #         midi_graph[new_node_hash][last_node_hash]['weight'] += 1
                    #
                    #     vu = midi_graph.get_edge_data(new_node_hash, last_node_hash)
                    #     if vu is not None:
                    #         midi_graph[last_node_hash][new_node_hash]['weight'] += 1
                    #
                    #     if is_debug:
                    #         print(f"---- Already connected {pitch_names} {last_node_name}")
                    #         print(f"---- uv {uv} last_hash {last_node_hash}")
                    #         print(f"---- vu {vu}")

                    # midi_graph[pitch_set_hash][last_node]['weight'] += 1
                    # midi_graph[new_node_hash][last_node_hash]['weight'] += 1
                    # midi_graph[last_node_hash][new_node_hash]['weight'] += 1
                    if is_debug:
                        print(f"---- Already connected {pitch_names} {last_node_name}")
                else:
                    if is_debug:
                        print(f"---- Adding an edge from {pitch_names} {last_node_name}")
                    midi_graph.add_node(new_node_hash, attr=new_x, label=pitch_names, node_hash=new_node_hash)
                    # midi_graph.add_edge(pitch_set_hash, last_node_hash, weight=1.0)
                    midi_graph.add_edge(last_node_hash, new_node_hash, weight=1.0)
                    # midi_graph.add_weighted_edges_from((pitch_set_hash, last_node, 1.0))

                last_node_hash = new_node_hash
                last_node_name = pitch_names

        # midi_graph = nx.relabel_nodes(midi_graph, mapping)
        return midi_graph

    def graphs(self):
        """Generator
        :return:
        """

        for g in self._pyg_data:
            # print("Label", g.label)
            label_map = {n: l for n, l in zip(g.node_hash, g.label)}
            g.label = torch.tensor(g.node_hash, dtype=torch.long)
            g.y = torch.tensor(g.node_hash, dtype=torch.long)
            # node_labels = [label_map[n] for n in g.node_hash]
            # g.y = g.x[:, -1].long()
            g.hidden_channels = torch.zeros((g.num_nodes, self.hidden_feature_size))

            num_nodes = g.num_nodes
            hidden_channels = torch.zeros(num_nodes, self.hidden_feature_size)
            g.hidden_channels = hidden_channels

            # g = nx.convert_node_labels_to_integers(g)
            # for i, (_, node_attr) in enumerate(g.nodes(data=True)):
            #     node_attr["label"] = label_dict[i]
            # pyg_data = from_networkx(g)

            yield g

        @classmethod
        def from_file(cls, file_path: str,
                      per_instrument: Optional[bool] = True,
                      hidden_feature_size: int = 64) -> 'MidiGraphBuilder':
            """Constructs a MidiGraphBuilder object from a MIDI file path.
            :param cls:
            :param file_path: A string representing the path to the MIDI file to be processed.
            :param per_instrument: A boolean indicating whether to build a graph for each instrument in the MIDI file.
            :param hidden_feature_size: The size of the hidden feature vector to use for the PyTorch Geometric graph.
            :return: A MidiGraphBuilder object constructed from the specified MIDI file.
            :rtype: MidiGraphBuilder
            """
            midi_seq = MidiReader.read(file_path)
            return cls(midi_seq, per_instrument=per_instrument, hidden_feature_size=hidden_feature_size)

        @classmethod
        def from_midi_sequence(cls,
                               midi_sequence: MidiNoteSequence,
                               per_instrument: Optional[bool] = True,
                               hidden_feature_size: int = 64) -> 'MidiGraphBuilder':
            """
            Constructs a new instance of `MidiGraphBuilder` using a `MidiNoteSequence` object.

            :param cls:
            :param midi_sequence: A `MidiNoteSequence` object containing MIDI data to construct a graph from.
            :param per_instrument: If True, will create a graph for each instrument in the `MidiNoteSequence`.
                                   Otherwise, a single graph will be constructed using all instruments.
            :param hidden_feature_size: The size of the hidden feature vector to use for the graph embeddings.
            :return: A new instance of `MidiGraphBuilder`.
            :rtype: MidiGraphBuilder
            """
            midi_sequences = MidiNoteSequences(midi_seq=midi_sequence)
            return cls(midi_sequences, per_instrument, hidden_feature_size)

