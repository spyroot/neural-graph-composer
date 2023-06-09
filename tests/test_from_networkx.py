

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

        builder = MidiGraphBuilder(midi_seq)
        #
        midi_graph = builder.build_sequence(midi_seq, max_vector_size=5)
        self.assertTrue(midi_graph.is_directed)
        self.assertTrue(nx.is_weighted(midi_graph))
        self.assertTrue(len(midi_graph.nodes) == 2)
        self.assertTrue(len(midi_graph.edges) == 2)

        # get edge check weight
        edge = builder.get_edge_connected(midi_graph, u=[a.pitch], v=[b.pitch])
        self.assertTrue(edge['weight'] == 1.0, "weight A to B should 1.0")
        edge = builder.get_edge_connected(midi_graph, u=[b.pitch], v=[a.pitch])
        self.assertIsNone(edge, "B should not have edge to A")
        edge = builder.get_edge_connected(midi_graph, u=[b.pitch], v=[b.pitch])
        self.assertTrue(edge['weight'] == 2.0, "B to B weight should be 2.0")
        edge = builder.get_edge_connected(midi_graph, u=[a.pitch], v=[a.pitch])
        self.assertIsNone(edge, "A should an edge to A")

        expected = tensor([[21., 0., 0., 0., 0.],
                           [23., 0., 0., 0., 0.]])

        pyg_data = builder.from_midi_networkx(midi_graph, group_node_attrs=["attr"])
        print(pyg_data.x)

        self.assertTrue(torch.equal(pyg_data.x, expected))

    def test_build_sequence_tolerance(self):
        notes = [
            MidiNote(pitch=60, start_time=0.5, end_time=1.0),
            MidiNote(pitch=61, start_time=1.0, end_time=1.5),
            MidiNote(pitch=62, start_time=1.4, end_time=1.8),
            MidiNote(pitch=63, start_time=2.0, end_time=2.5),
            MidiNote(pitch=64, start_time=2.5, end_time=3.0),
            MidiNote(pitch=65, start_time=2.9, end_time=3.2),
            MidiNote(pitch=66, start_time=3.5, end_time=4.0),
        ]
        seq = MidiNoteSequence(notes=notes)
        builder = MidiGraphBuilder()

        # Test with tolerance of 0.5
        graph = builder.build_sequence(seq, tolerance=0.5)
        for node in graph.nodes:
            notes = builder.hash_to_notes[node]
            print(f"Hash: {node}, Notes: {notes}")

        # builder.hash_to_notes
        self.assertEqual(7, len(graph))
        print(graph.nodes)
        print(graph.nodes)

        self.assertCountEqual(graph.nodes, [hash(frozenset({60})), hash(frozenset({61})), hash(frozenset({62})),
                                            hash(frozenset({63})), hash(frozenset({64})), hash(frozenset({65})),
                                            hash(frozenset({66}))])

        # Test with tolerance of 0.1
        graph = builder.build_sequence(seq, tolerance=0.1)
        self.assertEqual(len(graph), 7)
        self.assertCountEqual(graph.nodes, [hash(frozenset({60})), hash(frozenset({61})), hash(frozenset({62})),
                                            hash(frozenset({63})), hash(frozenset({64})), hash(frozenset({65})),
                                            hash(frozenset({66}))])

    def test_build_sequence_tolerance_close_notes(self):
        notes = [
            MidiNote(pitch=60, start_time=0.0, end_time=0.5),
            MidiNote(pitch=61, start_time=0.1, end_time=0.8),
            MidiNote(pitch=62, start_time=0.09, end_time=1.0),
            MidiNote(pitch=62, start_time=0.2, end_time=1.0),

        ]
        seq = MidiNoteSequence(notes=notes)
        builder = MidiGraphBuilder()

        # Test with tolerance of 0.2
        graph = builder.build_sequence(seq, tolerance=0.2)
        self.assertEqual(len(graph), 2)
        self.assertCountEqual(graph.nodes, [hash(frozenset({60, 61, 62})), hash(frozenset({62}))])

    def test_build_sequence_with_weights(self):
        notes = [
            MidiNote(pitch=60, start_time=0.0, end_time=0.5),
            MidiNote(pitch=61, start_time=1.0, end_time=1.5),
            MidiNote(pitch=62, start_time=2.0, end_time=2.5),
            MidiNote(pitch=62, start_time=3.0, end_time=3.5),
            MidiNote(pitch=60, start_time=4.0, end_time=4.5),
            MidiNote(pitch=60, start_time=6.0, end_time=6.5),
            MidiNote(pitch=60, start_time=7.0, end_time=7.5),
            MidiNote(pitch=61, start_time=8.0, end_time=8.5),
            MidiNote(pitch=62, start_time=9.0, end_time=9.5),
            MidiNote(pitch=62, start_time=10.0, end_time=10.5),
        ]

        seq = MidiNoteSequence(notes=notes)
        builder = MidiGraphBuilder()

        graph = builder.build_sequence(seq)
        for node in graph.nodes:
            notes = builder.hash_to_notes[node]
            print(f"Hash: {node}, Notes: {notes}")
            for neighbor in graph.neighbors(node):
                nei = builder.hash_to_notes[neighbor]
                weight = graph.get_edge_data(node, neighbor)['weight']
                print(f"  --> Neighbor: {nei}, Weight: {weight}")



