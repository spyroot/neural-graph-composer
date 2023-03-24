from copy import copy
from unittest import TestCase

import networkx as nx
import torch

from neural_graph_composer.midi.midi_instruments import MidiInstrumentInfo
from neural_graph_composer.midi.midi_note import MidiNote
from neural_graph_composer.midi.midi_sequence import MidiNoteSequence
from neural_graph_composer.midi.midi_sequences import MidiNoteSequences
from neural_graph_composer.midi_graph_builder import MidiGraphBuilder, NodeAttributeType
import networkx as nx

from tests.test_utils import generate_triads
from typing import List


class Test(TestCase):
    def test_init(self):
        """

        :return:
        """
        midi_seq = MidiNoteSequence.from_file('example.mid')
        builder = MidiGraphBuilder(midi_seq, __is_instrument_graph=False)
        pyg_data = builder.pyg_data

        # Assert that we have only one graph
        self.assertEqual(len(pyg_data), 1)

        # Assert that the graph contains the correct number of nodes
        self.assertEqual(pyg_data[0].x.shape[0], len(midi_seq))

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
        midi_sequences = MidiNoteSequences(midi_seq=midi_seq)
        self.assertTrue(midi_seq.total_time == 2.0)
        self.assertTrue(midi_sequences[0].total_time == 2.0)
        self.assertTrue(len(midi_seq.notes) == len(expect))
        self.assertListEqual(midi_seq.notes, expect)

        midi_graph_builder = MidiGraphBuilder.from_midi_sequences(midi_sequences)
        self.assertTrue(midi_graph_builder.midi_sequences[0] == midi_seq)

        midi_graph_builder.build()
        self.assertTrue(len(midi_graph_builder.sub_graphs) == 1)

        midi_graph = midi_graph_builder.sub_graphs[0]
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
        midi_sequences = MidiNoteSequences(midi_seq=midi_seq)
        self.assertTrue(midi_seq.total_time == 3.0)
        self.assertTrue(len(midi_seq.notes) == len(expect))
        self.assertListEqual(midi_seq.notes, expect)

        #
        midi_graph_builder = MidiGraphBuilder.from_midi_sequences(midi_sequences)
        self.assertTrue(midi_graph_builder.midi_sequences[0] == midi_seq)
        midi_graph_builder.build()
        self.assertTrue(len(midi_graph_builder.sub_graphs) == 1)

        midi_graph = midi_graph_builder.sub_graphs[0]
        self.assertTrue(midi_graph.is_directed)
        self.assertTrue(nx.is_weighted(midi_graph))
        self.assertTrue(len(midi_graph.nodes) == 1)
        self.assertTrue(len(midi_graph.edges) == 1)
        self.assertTrue(len(midi_graph_builder._notes_to_hash) == 1)

        edge = MidiGraphBuilder.get_edge_connected(
            midi_graph, u=[b.pitch], v=[a.pitch]
        )

        self.assertTrue(
            edge['weight'] == 2.0, msg=f"expected edge weight {2.0} actual {edge['weight']}"
        )

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

        graph_builder = MidiGraphBuilder(MidiNoteSequences())
        #
        midi_graph = graph_builder.build_sequence(midi_seq)
        self.assertTrue(midi_graph.is_directed)
        self.assertTrue(nx.is_weighted(midi_graph))
        self.assertTrue(len(midi_graph.nodes) == 2)
        self.assertTrue(len(midi_graph.edges) == 1)

        # get edge check weight
        edge = MidiGraphBuilder.get_edge_connected(midi_graph, u=[a.pitch], v=[b.pitch])
        print(edge)
        self.assertTrue(edge['weight'] == 1.0)

    def test_update_abb_weight01(self):
        """Create note A and note B and then follow-up Note B.
        Note a point to b and b point to itself.
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

        graph_builder = MidiGraphBuilder(MidiNoteSequences())
        midi_graph = graph_builder.build_sequence(midi_seq)
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
        # Create two note same pitch different in time
        # pitch A < -B  <- B <- A
        #
        # self loop
        a = MidiNote(pitch=21, start_time=0, end_time=0.5)
        b = MidiNote(pitch=23, start_time=1.0, end_time=2.0)
        c = MidiNote(pitch=23, start_time=2.0, end_time=3.0)
        d = MidiNote(pitch=21, start_time=3.0, end_time=4.0)

        # a
        expect = [a, b, c, d]
        midi_seq = MidiNoteSequence(notes=expect)
        self.assertTrue(midi_seq.total_time == 4.0)
        self.assertTrue(len(midi_seq.notes) == len(expect))
        self.assertListEqual(midi_seq.notes, expect)

        graph_builder = MidiGraphBuilder(MidiNoteSequences())
        midi_graph = graph_builder.build_sequence(midi_seq)

        self.assertTrue(midi_graph.is_directed)
        self.assertTrue(nx.is_weighted(midi_graph))
        self.assertTrue(len(midi_graph.nodes) == 2)
        self.assertTrue(len(midi_graph.edges) == 2, msg=f" expected 2 edges got {len(midi_graph.edges)}")

        # get edge check weight
        edge = MidiGraphBuilder.get_edge_connected(midi_graph, u=[a.pitch], v=[b.pitch])
        self.assertTrue(edge['weight'] == 2.0)
        edge = MidiGraphBuilder.get_edge_connected(midi_graph, u=[b.pitch], v=[a.pitch])
        # self.assertTrue(edge['weight'] == 1.0)
        edge = MidiGraphBuilder.get_edge_connected(midi_graph, u=[b.pitch], v=[b.pitch])
        self.assertTrue(edge['weight'] == 1.0)
        edge = MidiGraphBuilder.get_edge_connected(midi_graph, u=[a.pitch], v=[a.pitch])
        # print(edge)
        # self.assertIsNone(edge, "A should not have edge to A")
        print(nx.to_numpy_array(midi_graph))

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
        graph_builder = MidiGraphBuilder(MidiNoteSequences())
        midi_graph = graph_builder.build_sequence(midi_seq)
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

    def test_base_triad(self):
        """ 5 triad spaces to networkx
        :return:
        """
        triads = generate_triads(num_triads=5,
                                 octave=0,
                                 start_pitch=12,
                                 note_shift=2,
                                 duration=1.0,
                                 scale=[0, 2, 4, 5, 7, 9, 11])

        all_triads = [note for triad in triads for note in triad]
        midi_seq = MidiNoteSequence(notes=all_triads)
        self.assertEqual(len(all_triads), len(midi_seq))

        graph_builder = MidiGraphBuilder(MidiNoteSequences())
        midi_graph = graph_builder.build_sequence(midi_seq)
        # we expect 5 by 5, since all triad spaced
        np_adj = nx.to_numpy_array(midi_graph)
        self.assertTrue(np_adj.shape[0], len(all_triads))
        self.assertTrue(np_adj.shape[1], len(all_triads))

    def test_base_triad_to_pyg(self):
        """ 5 triad spaces to pyg
        :return:
        """
        triads = generate_triads(num_triads=5,
                                 octave=0,
                                 start_pitch=12,
                                 note_shift=2,
                                 duration=1.0,
                                 scale=[0, 2, 4, 5, 7, 9, 11])

        all_triads = [note for triad in triads for note in triad]
        midi_seq = MidiNoteSequence(notes=all_triads)
        graph_builder = MidiGraphBuilder(MidiNoteSequences())
        midi_graph = graph_builder.build_sequence(midi_seq)
        # we expect 5 by 5, since all triad spaced
        np_adj = nx.to_numpy_array(midi_graph)
        self.assertTrue(np_adj.shape[0], len(all_triads))
        self.assertTrue(np_adj.shape[1], len(all_triads))

        data = graph_builder.from_midi_networkx(midi_graph)
        data_x = data.x
        self.assertIsInstance(data_x, torch.Tensor)
        # 3 different triad 5 times
        print(data)
        self.assertEqual(5, data.x.shape[0])
        self.assertEqual(3, data.x.shape[1])
        self.assertTrue(data_x.dtype, torch.float32)

        # (2, num_edges)  vs (num_edges, num_feature)
        self.assertTrue(data.edge_index.shape[1], data.edge_attr.shape[0])

    def test_base_triad_two_instrument(self):
        """ Build a two midi seq.  each midi seq is separate instrument
            We pass all to graph builder and each instrument should be in own graph.
        :return:
        """
        num_instruments = 2
        num_triads = 5

        instruments = [
            MidiInstrumentInfo(instrument=instrument_id, name=f"test_{instrument_id}", is_drum=False)
            for instrument_id in range(0, num_instruments)
        ]

        for i, instrument in enumerate(instruments):
            self.assertEqual(i, instrument.instrument_num)

        midi_seq_list = []
        for j, instrument in enumerate(instruments):
            triads = generate_triads(num_triads=num_triads,
                                     octave=0,
                                     start_pitch=12,
                                     note_shift=2,
                                     duration=1.0,
                                     scale=[0, 2, 4, 5, 7, 9, 11],
                                     instrument_id=instrument.instrument_num)

            all_triads = [note for triad in triads for note in triad]
            midi_seq = MidiNoteSequence(notes=all_triads, instrument=copy(instrument))
            self.assertEqual(len(all_triads), len(midi_seq))
            midi_seq_list.append(midi_seq)
            self.assertEqual(instrument.instrument_num, midi_seq.instrument.instrument_num)

        self.assertEqual(num_instruments, len(midi_seq_list),
                         msg=f"we expected {num_instruments} seq")

        for i, seq in enumerate(midi_seq_list):
            self.assertEqual(i, seq.instrument.instrument_num)

        # create sequences and each midi instrument sequence
        midi_sequences = MidiNoteSequences(midi_seq=midi_seq_list)
        self.assertEqual(0, midi_seq_list[0].instrument.instrument_num)
        self.assertEqual(1, midi_seq_list[1].instrument.instrument_num)
        self.assertEqual(num_instruments, midi_sequences.num_instruments(),
                         msg=f"we expected {num_instruments} in midi "
                             f"note MidiNoteSequences")

        # we generate num_triads * 3 notes
        for i in range(0, num_instruments):
            self.assertEqual(num_triads * 3, len(midi_sequences[i].notes),
                             msg=f"we expected {num_triads * 3} notes in each seq ")

        # we create graph each instrument (by default should have onw graph)
        graph_builder = MidiGraphBuilder(midi_sequences)
        midi_graph = graph_builder.build()
        self.assertEqual(num_instruments, len(graph_builder.sub_graphs))
        # we expect 2 graph
        for i in range(0, num_instruments):
            np_adj = nx.to_numpy_array(graph_builder.sub_graphs[i])
            # we expect each sub-graph be a size of num triads.
            self.assertTrue(np_adj.shape[0], num_triads)
            self.assertTrue(np_adj.shape[1], num_triads)

    def test_base_triad_two_instruments(self):
        """ Same test know we convert to tensor TODO
        :return:
        """
        num_instruments = 2
        num_triads = 5

        instruments = [
            MidiInstrumentInfo(instrument=instrument_id, name=f"test_{instrument_id}", is_drum=False)
            for instrument_id in range(0, num_instruments)
        ]

        for i, instrument in enumerate(instruments):
            self.assertEqual(i, instrument.instrument_num)

        midi_seq_list = []
        for j, instrument in enumerate(instruments):
            triads = generate_triads(num_triads=num_triads,
                                     octave=0,
                                     start_pitch=12,
                                     note_shift=2,
                                     duration=1.0,
                                     scale=[0, 2, 4, 5, 7, 9, 11],
                                     instrument_id=instrument.instrument_num)

            all_triads = [note for triad in triads for note in triad]
            midi_seq = MidiNoteSequence(notes=all_triads, instrument=copy(instrument))
            self.assertEqual(len(all_triads), len(midi_seq))
            midi_seq_list.append(midi_seq)
            self.assertEqual(instrument.instrument_num, midi_seq.instrument.instrument_num)

        self.assertEqual(num_instruments, len(midi_seq_list),
                         msg=f"we expected {num_instruments} seq")

        for i, seq in enumerate(midi_seq_list):
            self.assertEqual(i, seq.instrument.instrument_num)

        # create sequences and each midi instrument sequence
        midi_sequences = MidiNoteSequences(midi_seq=midi_seq_list)
        self.assertEqual(0, midi_seq_list[0].instrument.instrument_num)
        self.assertEqual(1, midi_seq_list[1].instrument.instrument_num)
        self.assertEqual(num_instruments, midi_sequences.num_instruments(),
                         msg=f"we expected {num_instruments} in midi "
                             f"note MidiNoteSequences")

        # we generate num_triads * 3 notes
        for i in range(0, num_instruments):
            self.assertEqual(num_triads * 3, len(midi_sequences[i].notes),
                             msg=f"we expected {num_triads * 3} notes in each seq ")

        # we create graph each instrument (by default should have onw graph)
        graph_builder = MidiGraphBuilder(midi_sequences)
        midi_graph = graph_builder.build()
        self.assertEqual(num_instruments, len(graph_builder.sub_graphs))
        # we expect 2 graph
        for i in range(0, num_instruments):
            np_adj = nx.to_numpy_array(graph_builder.sub_graphs[i])
            # we expect each sub-graph be a size of num triads.
            self.assertTrue(np_adj.shape[0], num_triads)
            self.assertTrue(np_adj.shape[1], num_triads)

    def test_tensor(self):
        """
        :return:
        """
        pitch_set = {60, 62, 64}
        velocity_set = {1, 2, 3}
        print(pitch_set)
        print(velocity_set)
        x = MidiGraphBuilder.create_tensor(NodeAttributeType.Tensor,
                                           pitch_set, velocity_set,
                                           feature_vec_size=4)
        expected_x = torch.tensor([[64., 60., 62, 0],
                                   [1., 2., 3., 0.]])
        self.assertTrue(torch.all(torch.eq(x, expected_x)))

    def test_tensor2(self):
        """
        :return:
        """
        pitch_set = {60, 62, 64}
        velocity_set = {1, 2, 3}
        print(pitch_set)
        print(velocity_set)
        x = MidiGraphBuilder.create_tensor(NodeAttributeType.Tensor,
                                           pitch_set,
                                           feature_vec_size=4)
        expected_x = torch.tensor([64., 60., 62, 0])
        self.assertTrue(torch.all(torch.eq(x, expected_x)))

    def test_create_tensor_one_hot(self):
        """

        :return:
        """
        pitch_set = {60, 62, 64}
        velocity_set = {1, 3, 2}
        x = MidiGraphBuilder.create_tensor(NodeAttributeType.OneHotTensor,
                                           pitch_set, velocity_set,
                                           feature_vec_size=4,
                                           num_classes=127,
                                           velocity_num_buckets=8)
        expect = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 1.],
                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 2.],
                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 3.]])
