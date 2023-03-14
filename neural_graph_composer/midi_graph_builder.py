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


class NodeAttributeType(Enum):
    """This a type how we encode represent node attribute
    """
    Tensor = auto()
    OneHotTensor = auto()


class MidiGraphBuilder:
    """
    """

    def __init__(self):
        # this a default name that we use for node attributes
        self.node_attr_name = ["attr"]
        # in case we need map hashed value back to midi notes set
        self.hash_to_midi_set = {}

    @staticmethod
    def from_midi_networkx(
            midi_graph: Any,
            group_node_attrs: Optional[Union[List[str], all]] = None,
            group_edge_attrs: Optional[Union[List[str], all]] = None) -> PygData:
        """
        This simular to pyg from networkx but it has some critical changes because original
        semantically does something very different.

        :param midi_graph:
        :param group_node_attrs:
        :param group_edge_attrs:
        :return:
        """
        # default attr we expect
        if group_node_attrs is None:
            group_node_attrs = ["attr"]

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
            print(f"comparing {set(feat_dict.keys())} and {set(node_attrs)}")
            if set(feat_dict.keys()) != set(node_attrs):
                raise ValueError('Not all nodes contain the same attributes')
            for key, value in feat_dict.items():
                print(f"adding data key:{key} and value:{value}")
                data[str(key)].append(value)

        for i, (_, _, feat_dict) in enumerate(midi_graph.edges(data=True)):
            if set(feat_dict.keys()) != set(edge_attrs):
                raise ValueError('Not all edges contain the same attributes')
            for key, value in feat_dict.items():
                key = f'edge_{key}' if key in node_attrs else key
                data[str(key)].append(value)

        # print()
        # for key, value in G.graph.items():
        #     print(f" --- checking key: {key} in node_attrs")
        #     key = f'graph_{key}' if key in node_attrs else key
        #     #
        #     print(f"---- Added node atter {key}")
        #     data[str(key)] = value

        if group_node_attrs is not None:
            for key, value in data.items():
                print(f"key {key} and group_node_attrs {group_node_attrs}")
                if key not in group_node_attrs:
                    continue

                if isinstance(value, (tuple, list)) and isinstance(value[0], Tensor):
                    print("Adding val case two")
                    data[key] = torch.stack(value, dim=0)
                else:
                    try:
                        print(f"current key {key}, values", value)
                        if isinstance(value, torch.Tensor):
                            data[key] = value
                        else:
                            data[key] = torch.tensor(value)
                        print("add")
                        print(data[key])
                    except (ValueError, TypeError):
                        pass

        data['edge_index'] = edge_index.view(2, -1)
        data = Data.from_dict(data)

        if group_node_attrs is not None:
            xs = []
            for key in group_node_attrs:
                x = data[key]
                x = x.view(-1, 1) if x.dim() <= 1 else x
                xs.append(x)
                del data[key]
            data.x = torch.cat(xs, dim=-1)

        # TODO remove this if in downstream we wont need any edge attribute
        # we can nuke this out.
        if group_edge_attrs is all:
            group_edge_attrs = list(edge_attrs)
        if group_edge_attrs is not None:
            xs = []
            for key in group_edge_attrs:
                key = f'edge_{key}' if key in node_attrs else key
                x = data[key]
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

    def construct_graph_from_seq(self,
                                 seq,
                                 default_trim_time: Optional[int] = 3,
                                 max_vector_size: Optional[int] = 5,
                                 velocity_num_buckets: Optional[int] = 8,
                                 node_attr_type: NodeAttributeType = NodeAttributeType.Tensor,
                                 is_debug: Optional[bool] = False):
        """
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
        # sort all pitches played at same time.
        # we truncated float time. Long notes that span different chord
        # will be present in different node, and it ok
        data = {}
        # read seq of notes and form a hash list where
        # key is start time and value a note.
        for n in seq.notes:
            # all drum are skipped, we only care about instrument
            # produce harmony and velocity > 0
            if n.is_drum or n.velocity == 0:
                continue
            time_hash = round(n.start_time, default_trim_time)
            if time_hash not in data:
                data[time_hash] = []
                data[time_hash].append(n)
            else:
                data[time_hash].append(n)

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
            # we add hash for a given pitch_set to dict
            # so we can recover if we need 2
            if new_node_hash not in self.hash_to_midi_set:
                self.hash_to_midi_set[pitch_set] = new_node_hash

            # mapping map hash to pitch name
            mapping[new_node_hash] = pitch_names
            if node_attr_type == NodeAttributeType.Tensor:
                pitch_attr = torch.FloatTensor(list(pitch_set))
                new_x = torch.zeros(max_vector_size)
                new_x[:pitch_attr.shape[0]] = pitch_attr
            elif node_attr_type == NodeAttributeType.OneHotTensor:
                labels_tensor = torch.FloatTensor(list(pitch_set))
                new_x = torch.nn.functional.one_hot(labels_tensor, num_classes=127)
            else:
                raise ValueError("Unknown encoder type")

            # pitch_attr_np = torch.from_numpy(pitch_attr_np)
            # print(f"PITCH TORCH {pitch_attr_np}")
            if last_node_hash is None:
                if is_debug:
                    print(f"Adding node {pitch_set} {pitch_names}")
                midi_graph.add_node(new_node_hash, attr=new_x, label=pitch_names)
                last_node_hash = new_node_hash
                last_node_name = pitch_names
            else:
                # if node already connected update weight
                if self.nodes_connected(midi_graph, new_node_hash, last_node_hash):
                    if is_debug:
                        uv = midi_graph.get_edge_data(last_node_hash, new_node_hash)
                        if uv is not None:
                            midi_graph[new_node_hash][last_node_hash]['weight'] += 1
                        vu = midi_graph.get_edge_data(new_node_hash, last_node_hash)
                        if uv is not None:
                            midi_graph[last_node_hash][new_node_hash]['weight'] += 1
                        print(f"---- Already connected {pitch_names} {last_node_name}")
                        print(f"---- uv {uv} last_hash {last_node_hash}")
                        print(f"---- vu {vu}")

                    # midi_graph[pitch_set_hash][last_node]['weight'] += 1
                    # midi_graph[new_node_hash][last_node_hash]['weight'] += 1
                    # midi_graph[last_node_hash][new_node_hash]['weight'] += 1
                    if is_debug:
                        print(f"---- Already connected {pitch_names} {last_node_name}")
                else:
                    if is_debug:
                        print(f"---- Adding an edge from {pitch_names} {last_node_name}")
                    midi_graph.add_node(new_node_hash, attr=new_x, label=pitch_names)
                    # midi_graph.add_edge(pitch_set_hash, last_node_hash, weight=1.0)
                    midi_graph.add_edge(last_node_hash, new_node_hash, weight=1.0)
                    # midi_graph.add_weighted_edges_from((pitch_set_hash, last_node, 1.0))

                last_node_hash = new_node_hash
                last_node_name = pitch_names

        # midi_graph = nx.relabel_nodes(midi_graph, mapping)
        return midi_graph
