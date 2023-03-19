

# midi_seq = midi_to_tensor("dataset/midi_test01.mid")
# midi_graph = construct_graph_from_seq(midi_seq[0])
# pyg_data = from_midi_networkx(midi_graph, group_node_attrs=["attr"])

from unittest import TestCase

import networkx as nx
import torch
from torch import tensor

from neural_graph_composer.midi.midi_note import MidiNote
from neural_graph_composer.midi.midi_sequence import MidiNoteSequence
from neural_graph_composer.midi_graph_builder import MidiGraphBuilder


class Test(TestCase):
    def test_simple_case(self):
        """

        :return:
        """
        #  pitch A follow pitch B follow pitch B
        a = MidiNote(pitch=21, start_time=0, end_time=0.5)
        b = MidiNote(pitch=23, start_time=1.0, end_time=2.0)
        b2 = MidiNote(pitch=23, start_time=2.0, end_time=3.0)

        expect = [a, b, b2]
        midi_seq = MidiNoteSequence(notes=expect)
        self.assertTrue(midi_seq.total_time == 3.0)
        self.assertTrue(len(midi_seq.notes) == len(expect))
        self.assertListEqual(midi_seq.notes, expect)

        midi_graph_builder = MidiGraphBuilder(midi_seq)

    def test_update_abbb(self):
        """Add a check neigh
        :return:
        """
        # create two note same pitch different in time
        # pitch A < -B  <- B <- B
        #
        #
        # self loop
        a = MidiNote(pitch=21, start_time=0, end_time=0.5)
        b = MidiNote(pitch=23, start_time=1.0, end_time=2.0)
        b2 = MidiNote(pitch=23, start_time=2.0, end_time=3.0)
        b3 = MidiNote(pitch=23, start_time=3.0, end_time=4.0)

        expect = [a, b, b2, b3]
        midi_seq = MidiNoteSequence(notes=expect)
        self.assertTrue(midi_seq.total_time == 4.0)
        self.assertTrue(len(midi_seq.notes) == len(expect))
        self.assertListEqual(midi_seq.notes, expect)

        #
        midi_graph = construct_graph_from_seq(midi_seq)
        self.assertTrue(midi_graph.is_directed)
        self.assertTrue(nx.is_weighted(midi_graph))
        self.assertTrue(len(midi_graph.nodes) == 2)
        self.assertTrue(len(midi_graph.edges) == 2)

        # get edge check weight
        edge = get_edge_connected(midi_graph, u=[a.pitch], v=[b.pitch])
        self.assertTrue(edge['weight'] == 1.0, "weight A to B should 1.0")
        edge = get_edge_connected(midi_graph, u=[b.pitch], v=[a.pitch])
        self.assertIsNone(edge, "B should not have edge to A")
        edge = get_edge_connected(midi_graph, u=[b.pitch], v=[b.pitch])
        self.assertTrue(edge['weight'] == 2.0, "B to B weight should be 2.0")
        edge = get_edge_connected(midi_graph, u=[a.pitch], v=[a.pitch])
        self.assertIsNone(edge, "A should an edge to A")

        expected = tensor([[21., 0., 0., 0., 0.],
                           [23., 0., 0., 0., 0.]])
        pyg_data = from_midi_networkx(midi_graph, group_node_attrs=["attr"])
        self.assertTrue(torch.equal(pyg_data.x, expected))
