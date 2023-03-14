from unittest import TestCase

import networkx as nx

from neural_graph_composer.midi.midi_note import MidiNote
from neural_graph_composer.midi.midi_sequence import MidiNoteSequence
from neural_graph_composer.midi_graph_builder import MidiGraphBuilder


class Test(TestCase):
    def test_add_self_loop(self):
        """Add two note and construct graph.
           - Make sure it directed.
           - same note at next time step in time increase add self loop weight.
        :return:
        """
        # create two note same pitch different in time
        # connect and

        # self loop
        a = MidiNote(pitch=51, start_time=0, end_time=0.5)
        b = MidiNote(pitch=51, start_time=1.0, end_time=2.0)
        expect = [a, b]
        midi_seq = MidiNoteSequence(notes=expect)
        self.assertTrue(midi_seq.total_time == 2.0)
        self.assertTrue(len(midi_seq.notes) == len(expect))
        self.assertListEqual(midi_seq.notes, expect)

        #
        midi_graph = MidiGraphBuilder.construct_graph_from_seq(midi_seq)
        self.assertTrue(midi_graph.is_directed)
        self.assertTrue(nx.is_weighted(midi_graph))
        self.assertTrue(len(midi_graph.nodes) == 1)
        self.assertTrue(len(midi_graph.edges) == 1)

        edge = MidiGraphBuilder.get_edge_connected(midi_graph, u=[b.pitch], v=[a.pitch])
        self.assertTrue(edge['weight'] == 1.0)

    def test_update_self_loop_weight(self):
        """Add three note and construct graph.
           -  51 at time step 0
           -  51 at time step 1.0
           -  51 at time step 2.0

           Last edge should update weight

        :return:
        """
        # create two note same pitch different in time
        # connect and

        # self loop
        a = MidiNote(pitch=51, start_time=0, end_time=0.5)
        b = MidiNote(pitch=51, start_time=1.0, end_time=2.0)
        c = MidiNote(pitch=51, start_time=2.0, end_time=3.0)
        expect = [a, b, c]
        midi_seq = MidiNoteSequence(notes=expect)
        self.assertTrue(midi_seq.total_time == 3.0)
        self.assertTrue(len(midi_seq.notes) == len(expect))
        self.assertListEqual(midi_seq.notes, expect)

        #
        midi_graph = MidiGraphBuilder.construct_graph_from_seq(midi_seq)
        self.assertTrue(midi_graph.is_directed)
        self.assertTrue(nx.is_weighted(midi_graph))
        self.assertTrue(len(midi_graph.nodes) == 1)
        self.assertTrue(len(midi_graph.edges) == 1)

        edge = MidiGraphBuilder.get_edge_connected(midi_graph, u=[b.pitch], v=[a.pitch])
        self.assertTrue(edge['weight'] == 2.0)

    def test_link_a_b(self):
        """Add a check neigh
        :return:
        """
        # create two note same pitch different in time
        # connect and

        # self loop
        a = MidiNote(pitch=51, start_time=0, end_time=0.5)
        b = MidiNote(pitch=52, start_time=1.0, end_time=2.0)
        expect = [a, b]
        midi_seq = MidiNoteSequence(notes=expect)
        self.assertTrue(midi_seq.total_time == 2.0)
        self.assertTrue(len(midi_seq.notes) == len(expect))
        self.assertListEqual(midi_seq.notes, expect)

        #
        midi_graph = MidiGraphBuilder.construct_graph_from_seq(midi_seq)
        self.assertTrue(midi_graph.is_directed)
        self.assertTrue(nx.is_weighted(midi_graph))
        self.assertTrue(len(midi_graph.nodes) == 2)
        self.assertTrue(len(midi_graph.edges) == 1)

        # get edge check weight
        edge = MidiGraphBuilder.get_edge_connected(midi_graph, u=[a.pitch], v=[b.pitch])
        print(edge)
        self.assertTrue(edge['weight'] == 1.0)

    def test_update_abb_weight01(self):
        """Add a check neigh
        :return:
        """
        # create two note same pitch different in time
        # connect and
        # pitch A < -B  <- B
        # self loop
        a = MidiNote(pitch=21, start_time=0, end_time=0.5)
        b = MidiNote(pitch=23, start_time=1.0, end_time=2.0)
        b2 = MidiNote(pitch=23, start_time=2.0, end_time=3.0)

        expect = [a, b, b2]
        midi_seq = MidiNoteSequence(notes=expect)
        self.assertTrue(midi_seq.total_time == 3.0)
        self.assertTrue(len(midi_seq.notes) == len(expect))
        self.assertListEqual(midi_seq.notes, expect)

        #
        midi_graph = MidiGraphBuilder.construct_graph_from_seq(midi_seq)
        self.assertTrue(midi_graph.is_directed)
        self.assertTrue(nx.is_weighted(midi_graph))
        self.assertTrue(len(midi_graph.nodes) == 2)
        self.assertTrue(len(midi_graph.edges) == 2)

        # get edge check weight
        edge = MidiGraphBuilder.get_edge_connected(midi_graph, u=[a.pitch], v=[b.pitch])
        self.assertTrue(edge['weight'] == 1.0)
        edge = MidiGraphBuilder.get_edge_connected(midi_graph, u=[b.pitch], v=[b2.pitch])
        self.assertTrue(edge['weight'] == 1.0)

    def test_update_abba(self):
        """Add a check neigh
        :return:
        """
        # create two note same pitch different in time
        # pitch A < -B  <- B <- A
        #
        #
        # self loop
        a = MidiNote(pitch=21, start_time=0, end_time=0.5)
        b = MidiNote(pitch=23, start_time=1.0, end_time=2.0)
        c = MidiNote(pitch=23, start_time=2.0, end_time=3.0)
        d = MidiNote(pitch=21, start_time=3.0, end_time=4.0)

        expect = [a, b, c, d]
        midi_seq = MidiNoteSequence(notes=expect)
        self.assertTrue(midi_seq.total_time == 4.0)
        self.assertTrue(len(midi_seq.notes) == len(expect))
        self.assertListEqual(midi_seq.notes, expect)

        #
        midi_graph = MidiGraphBuilder.construct_graph_from_seq(midi_seq)
        self.assertTrue(midi_graph.is_directed)
        self.assertTrue(nx.is_weighted(midi_graph))
        self.assertTrue(len(midi_graph.nodes) == 2)
        self.assertTrue(len(midi_graph.edges) == 3)

        # get edge check weight
        edge = MidiGraphBuilder.get_edge_connected(midi_graph, u=[a.pitch], v=[b.pitch])
        self.assertTrue(edge['weight'] == 1.0)
        edge = MidiGraphBuilder.get_edge_connected(midi_graph, u=[b.pitch], v=[a.pitch])
        self.assertTrue(edge['weight'] == 1.0)
        edge = MidiGraphBuilder.get_edge_connected(midi_graph, u=[b.pitch], v=[b.pitch])
        self.assertTrue(edge['weight'] == 1.0)
        edge = MidiGraphBuilder.get_edge_connected(midi_graph, u=[a.pitch], v=[a.pitch])
        self.assertIsNone(edge, "A should not have edge to A")

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
        midi_graph = MidiGraphBuilder.construct_graph_from_seq(midi_seq)
        self.assertTrue(midi_graph.is_directed)
        self.assertTrue(nx.is_weighted(midi_graph))
        self.assertTrue(len(midi_graph.nodes) == 2)
        self.assertTrue(len(midi_graph.edges) == 2)

        # get edge check weight
        edge = MidiGraphBuilder.get_edge_connected(midi_graph, u=[a.pitch], v=[b.pitch])
        self.assertTrue(edge['weight'] == 1.0, "weight A to B should 1.0")
        edge = MidiGraphBuilder.get_edge_connected(midi_graph, u=[b.pitch], v=[a.pitch])
        self.assertIsNone(edge, "B should not have edge to A")
        edge = MidiGraphBuilder.get_edge_connected(midi_graph, u=[b.pitch], v=[b.pitch])
        self.assertTrue(edge['weight'] == 2.0, "B to B weight should be 2.0")
        edge = MidiGraphBuilder.get_edge_connected(midi_graph, u=[a.pitch], v=[a.pitch])
        self.assertIsNone(edge, "A should an edge to A")
