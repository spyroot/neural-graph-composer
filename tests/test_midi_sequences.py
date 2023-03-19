import itertools
from unittest import TestCase

import numpy as np

from neural_graph_composer.midi.midi_key_signature import KeySignatureType
from neural_graph_composer.midi.midi_note import MidiNote
from neural_graph_composer.midi.midi_sequence import MidiNoteSequence
from neural_graph_composer.midi.midi_sequences import MidiNoteSequences
from neural_graph_composer.midi.midi_spec import DEFAULT_QPM, DEFAULT_PPQ
from neural_graph_composer.midi.midi_time_signature import MidiTimeSignature, MidiTempoSignature
from neural_graph_composer.midi.midi_seq import MidiSeq
from neural_graph_composer.midi.midi_instruments import MidiInstrumentInfo
import random


class Test(TestCase):

    def test_init_default(self):
        """Test initialization of a `MidiNoteSequences` object with the default values."""
        midi_seqs = MidiNoteSequences()
        self.assertIsNone(midi_seqs.midi_seqs)
        self.assertEqual(len(midi_seqs.time_signatures), 1)

        # default time
        self.assertEqual(midi_seqs.time_signatures[0].numerator, 4)
        self.assertEqual(midi_seqs.time_signatures[0].denominator, 4)
        self.assertEqual(len(midi_seqs.key_signatures), 1)

        # default key
        self.assertEqual(midi_seqs.key_signatures[0].midi_time, 0)
        self.assertEqual(midi_seqs.key_signatures[0].mode, KeySignatureType.MAJOR)
        self.assertEqual(len(midi_seqs.tempo_signatures), 1)
        # default tempo
        self.assertEqual(midi_seqs.tempo_signatures[0].qpm, 120)
        self.assertEqual(midi_seqs.resolution, DEFAULT_PPQ)

        # default total time
        self.assertEqual(midi_seqs.total_time, 0.0)
        self.assertEqual(midi_seqs.filename, "")
        self.assertIsNone(midi_seqs.source_info)
        self.assertEqual(midi_seqs.is_debug, True)

    def test_init_from_midi_seq(self):
        """Test initialization of a `MidiNoteSequences`
        object from a `MidiNoteSequence` object
        :return:
        """
        midi_data = MidiNoteSequence(
            instrument=MidiInstrumentInfo(instrument=1, is_drum=False, name='Piano'))
        midi_data.notes.append(MidiNote(60, 0, 1, velocity=100, instrument=1))

        midi_seqs = MidiNoteSequences(midi_seq=midi_data)
        self.assertIsNotNone(midi_seqs.midi_seqs)
        self.assertEqual(len(midi_seqs.midi_seqs), 1)

        self.assertEqual(midi_seqs.midi_seqs[1], midi_data)
        self.assertEqual(len(midi_seqs.time_signatures), 1)
        self.assertEqual(len(midi_seqs.key_signatures), 1)
        self.assertEqual(len(midi_seqs.tempo_signatures), 1)

    def test_init_with_invalid_tempo_signature(self):
        """Test initialization of a `MidiNoteSequences` object with an invalid tempo signature."""
        with self.assertRaises(AssertionError):
            tempo_signature = MidiTempoSignature(midi_time=0.0, qpm=-10)
        with self.assertRaises(AssertionError):
            tempo_signature = MidiTempoSignature(midi_time=0.0, qpm=0)

    def test_init_with_invalid_time_signature(self):
        """Test initialization of a `MidiNoteSequences` object with an invalid time signature."""
        with self.assertRaises(ValueError):
            time_signature = MidiTimeSignature(numerator=3, denominator=0)

    def test_init_with_duplicate_instruments(self):
        """Test initialization of a `MidiNoteSequences` object
        with a list of `MidiNoteSequence` objects where some objects
        have the same instrument number.
        """
        midi_data1 = MidiNoteSequence(instrument=MidiInstrumentInfo(instrument=1, is_drum=False, name='Piano'))
        midi_data1.notes.append(MidiNote(60, 0, 1, velocity=100, instrument=1))

        midi_data2 = MidiNoteSequence(instrument=MidiInstrumentInfo(instrument=2, is_drum=False, name='Guitar'))
        midi_data2.notes.append(MidiNote(62, 0, 1, velocity=100, instrument=2))

        midi_data3 = MidiNoteSequence(instrument=MidiInstrumentInfo(instrument=1, is_drum=False, name='Piano'))
        midi_data3.notes.append(MidiNote(64, 0, 1, velocity=100, instrument=1))

        midi_seqs = MidiNoteSequences(midi_seq=[midi_data1, midi_data2, midi_data3])
        self.assertIsNotNone(midi_seqs.midi_seqs)
        self.assertEqual(len(midi_seqs.midi_seqs), 2)

        self.assertIn(1, midi_seqs.midi_seqs)
        self.assertIn(2, midi_seqs.midi_seqs)

        self.assertEqual(len(midi_seqs.midi_seqs[1]), 2)
        self.assertEqual(len(midi_seqs.midi_seqs[2]), 1)

        self.assertEqual(midi_seqs.midi_seqs[1].notes[0].pitch, 60)
        self.assertEqual(midi_seqs.midi_seqs[1].notes[1].pitch, 64)
        self.assertEqual(midi_seqs.midi_seqs[2].notes[0].pitch, 62)

    def test_init_with_duplicate_instruments_permuted(self):
        """Test initialization of a `MidiNoteSequences` object with a list of `MidiNoteSequence` objects
        where some objects have the same instrument number. The list is permuted and tested n times.
        """
        midi_data1 = MidiNoteSequence(instrument=MidiInstrumentInfo(instrument=1, is_drum=False, name='Piano'))
        midi_data1.notes.append(MidiNote(60, 0, 1, velocity=100, instrument=1))
        midi_data2 = MidiNoteSequence(instrument=MidiInstrumentInfo(instrument=2, is_drum=False, name='Guitar'))
        midi_data2.notes.append(MidiNote(62, 0, 1, velocity=100, instrument=2))
        midi_data3 = MidiNoteSequence(instrument=MidiInstrumentInfo(instrument=3, is_drum=False, name='Piano'))
        midi_data3.notes.append(MidiNote(64, 0, 1, velocity=100, instrument=3))

        # this case we need merge or throw error
        # midi_data4 = MidiNoteSequence(instrument=MidiInstrumentInfo(instrument=1, is_drum=False, name='Piano'))
        # midi_data4.notes.append(MidiNote(64, 0, 1, velocity=100, instrument=1))

        midi_data_list = [midi_data2, midi_data1, midi_data3]

        for perm in itertools.permutations(midi_data_list):
            midi_seqs = MidiNoteSequences(midi_seq=list(perm))
            self.assertIsNotNone(midi_seqs.midi_seqs)
            self.assertEqual(len(midi_seqs.midi_seqs), 3)
            print(midi_seqs)
            self.assertEqual(midi_seqs[1].notes[0].pitch, 60)
            self.assertEqual(midi_seqs[2].notes[0].pitch, 62)
            self.assertEqual(midi_seqs[3].notes[0].pitch, 64)

    def test_create_with_duplicate_instruments_permuted(self):
        """Test initialization of a `MidiNoteSequences` object with a list of `MidiNoteSequence` objects
        where some objects have the same instrument number. The list is permuted and tested n times.
        """
        midi_data1 = MidiNoteSequence(instrument=MidiInstrumentInfo(instrument=1, is_drum=False, name='Piano'))
        midi_data1.notes.append(MidiNote(60, 0, 1, velocity=100, instrument=1))
        midi_data2 = MidiNoteSequence(instrument=MidiInstrumentInfo(instrument=2, is_drum=False, name='Guitar'))
        midi_data2.notes.append(MidiNote(62, 0, 1, velocity=100, instrument=2))
        midi_data3 = MidiNoteSequence(instrument=MidiInstrumentInfo(instrument=3, is_drum=False, name='Piano'))
        midi_data3.notes.append(MidiNote(64, 0, 1, velocity=100, instrument=3))

        # this case we need merge or throw error
        # midi_data4 = MidiNoteSequence(instrument=MidiInstrumentInfo(instrument=1, is_drum=False, name='Piano'))
        # midi_data4.notes.append(MidiNote(64, 0, 1, velocity=100, instrument=1))

        midi_data_list = [midi_data2, midi_data1, midi_data3]

        for perm in itertools.permutations(midi_data_list):
            midi_seqs = MidiNoteSequences(midi_seq=list(perm))
            self.assertIsNotNone(midi_seqs.midi_seqs)
            self.assertEqual(len(midi_seqs.midi_seqs), 3)
            print(midi_seqs)
            self.assertEqual(midi_seqs[1].notes[0].pitch, 60)
            self.assertEqual(midi_seqs[2].notes[0].pitch, 62)
            self.assertEqual(midi_seqs[3].notes[0].pitch, 64)

    def test_init_from_midi_seq_list(self):
        """Test initialization of a `MidiNoteSequences` object
        from a list of `MidiNoteSequence` objects."""
        midi_data1 = MidiNoteSequence(instrument=MidiInstrumentInfo(instrument=1, is_drum=False, name='Piano'))
        midi_data1.notes.append(MidiNote(60, 0, 1, velocity=100, instrument=1))
        midi_data2 = MidiNoteSequence(instrument=MidiInstrumentInfo(instrument=2, is_drum=False, name='Guitar'))
        midi_data2.notes.append(MidiNote(62, 0, 1, velocity=100, instrument=2))

        midi_seqs = MidiNoteSequences(midi_seq=[midi_data1, midi_data2])

        print(midi_seqs)

        self.assertIsNotNone(midi_seqs.midi_seqs)
        self.assertEqual(len(midi_seqs.midi_seqs), 2)
        self.assertIn(1, midi_seqs.midi_seqs)
        self.assertIn(2, midi_seqs.midi_seqs)
        self.assertEqual(midi_seqs.midi_seqs[1], midi_data1)
        self.assertEqual(midi_seqs.midi_seqs[2], midi_data2)

    def test_init_with_shuffled_midi_data_list(self):
        """Test initialization of a `MidiNoteSequences`
        object with a shuffled list of `MidiNoteSequence` objects.

        track from instrument number.
        """
        midi_data1 = MidiNoteSequence(instrument=MidiInstrumentInfo(instrument=1, is_drum=False, name='Piano'))
        midi_data1.notes.append(MidiNote(60, 0, 1, velocity=100, instrument=1))
        midi_data2 = MidiNoteSequence(instrument=MidiInstrumentInfo(instrument=2, is_drum=False, name='Guitar'))
        midi_data2.notes.append(MidiNote(62, 0, 1, velocity=100, instrument=2))
        midi_data3 = MidiNoteSequence(instrument=MidiInstrumentInfo(instrument=3, is_drum=False, name='Bass'))
        midi_data3.notes.append(MidiNote(64, 0, 1, velocity=100, instrument=3))
        midi_data4 = MidiNoteSequence(instrument=MidiInstrumentInfo(instrument=4, is_drum=False, name='Bass'))
        midi_data4.notes.append(MidiNote(64, 0, 1, velocity=100, instrument=3))

        midi_data_list = [midi_data2, midi_data1, midi_data3, midi_data4]

        # Shuffle the list randomly
        random.shuffle(midi_data_list)
        midi_seqs = MidiNoteSequences(midi_seq=midi_data_list)

        self.assertIsNotNone(midi_seqs.midi_seqs)
        self.assertEqual(len(midi_seqs.midi_seqs), 4)

        self.assertEqual(midi_seqs.midi_seqs[3].instrument.instrument_num, 3)
        self.assertEqual(midi_seqs.midi_seqs[2].instrument.instrument_num, 2)
        self.assertEqual(midi_seqs.midi_seqs[1].instrument.instrument_num, 1)

    def test_create_instrument_order(self):
        """
        :return:
        """
        midi_seqs = MidiNoteSequences()
        idx1 = midi_seqs.create_track(3, 10, "Piano", False)
        idx2 = midi_seqs.create_track(1, 11, "Guitar", False)
        idx3 = midi_seqs.create_track(2, 12, "Bass", False)

        print(midi_seqs.midi_seqs)
        print(midi_seqs._track_to_idx)
        self.assertEqual(idx1, 3)
        self.assertEqual(idx2, 1)
        self.assertEqual(idx3, 2)
        # self.assertEqual(midi_seqs.create_instrument(1, "Another Piano", False), 0)
        # self.assertEqual(midi_seqs.create_instrument(4, "Drums", True), 3)

    def test_add_time_signatures(self):
        """Test that `add_time_signatures` correctly inserts new `MidiTimeSignature`
        instances into the `time_signatures` list of a `MidiNoteSequences` instance
        in different orders and check order of  `midi_time.
        The order must sort.
        :return:
        """
        # sequential order
        midi_seq = MidiNoteSequences()
        midi_seq.add_time_signatures(MidiTimeSignature(denominator=4, numerator=4, midi_time=0))
        midi_seq.add_time_signatures(MidiTimeSignature(denominator=4, numerator=3, midi_time=100))
        midi_seq.add_time_signatures(MidiTimeSignature(denominator=8, numerator=6, midi_time=200))
        midi_seq.add_time_signatures(MidiTimeSignature(denominator=4, numerator=5, midi_time=300))
        self.assertTrue(MidiNoteSequences.is_sorted(midi_seq.time_signatures, key=lambda x: x.midi_time))
        self.assertEqual(4, len(midi_seq.time_signatures))
        self.assertEqual(300, midi_seq.time_signatures[-1].midi_time)

        # reverse order
        midi_seq = MidiNoteSequences()
        midi_seq.add_time_signatures(MidiTimeSignature(denominator=4, numerator=5, midi_time=300))
        midi_seq.add_time_signatures(MidiTimeSignature(denominator=4, numerator=3, midi_time=200))
        midi_seq.add_time_signatures(MidiTimeSignature(denominator=8, numerator=6, midi_time=100))
        midi_seq.add_time_signatures(MidiTimeSignature(denominator=4, numerator=4, midi_time=0))

        self.assertTrue(MidiNoteSequences.is_sorted(midi_seq.time_signatures, key=lambda x: x.midi_time))
        self.assertEqual(4, len(midi_seq.time_signatures))
        self.assertEqual(0, midi_seq.time_signatures[0].midi_time, 0)
        self.assertEqual(300, midi_seq.time_signatures[-1].midi_time)

        # random order
        midi_seq = MidiNoteSequences()
        midi_seq.add_time_signatures(MidiTimeSignature(denominator=4, numerator=4, midi_time=300))
        midi_seq.add_time_signatures(MidiTimeSignature(denominator=4, numerator=3, midi_time=200))
        midi_seq.add_time_signatures(MidiTimeSignature(denominator=8, numerator=6, midi_time=100))
        midi_seq.add_time_signatures(MidiTimeSignature(denominator=4, numerator=5, midi_time=0))
        self.assertTrue(MidiNoteSequences.is_sorted(midi_seq.time_signatures, key=lambda x: x.midi_time))
        self.assertEqual(4, len(midi_seq.time_signatures))
        self.assertEqual(0, midi_seq.time_signatures[0].midi_time)
        self.assertEqual(300, midi_seq.time_signatures[-1].midi_time)

    def test_add_time_signatures_close_to_existing(self):
        """
        Correct order
        [
         MidiTempoSignature(seq=1, numerator=4, denominator=4, time=0),
         MidiTempoSignature(seq=2, numerator=3, denominator=4, time=1.5),
         MidiTempoSignature(seq=3, numerator=6, denominator=8, time=3.0),
         MidiTempoSignature(seq=4, numerator=6, denominator=8, time=3.0),
         MidiTempoSignature(seq=5, numerator=6, denominator=8, time=3.0),
         MidiTempoSignature(seq=6, numerator=6, denominator=8, time=4.0)
        ]
        :return:
        """
        midi_seq = MidiNoteSequences()
        time_sig1 = MidiTimeSignature(numerator=4, denominator=4, midi_time=0)
        time_sig2 = MidiTimeSignature(numerator=3, denominator=4, midi_time=1.5)
        time_sig3 = MidiTimeSignature(numerator=6, denominator=8, midi_time=3.0)

        midi_seq.add_time_signatures(time_sig1)
        midi_seq.add_time_signatures(time_sig2)
        midi_seq.add_time_signatures(time_sig3)
        time_sig_last01 = MidiTimeSignature(numerator=6, denominator=8, midi_time=3.0)
        time_sig_last02 = MidiTimeSignature(numerator=6, denominator=8, midi_time=3.0)
        last_diff_time_step = MidiTimeSignature(numerator=6, denominator=8, midi_time=4.0)

        midi_seq.add_time_signatures(time_sig_last01)
        midi_seq.add_time_signatures(time_sig_last02)
        midi_seq.add_time_signatures(last_diff_time_step)

        assert midi_seq.time_signatures[0] == time_sig1
        assert len(midi_seq.time_signatures) == 6
        assert midi_seq.time_signatures[0] == time_sig1
        assert midi_seq.time_signatures[1] == time_sig2
        assert midi_seq.time_signatures[2] == time_sig3
        assert midi_seq.time_signatures[3] == time_sig_last01
        assert midi_seq.time_signatures[4] == time_sig_last02

    def test_add_tempo_signatures(self):
        """Test that `add_tempo_signatures` correctly inserts new `MidiTimeSignature`
        instances into the `tempo` list of a `MidiNoteSequences` instance
        in different orders and check order of  `midi_time`.
        The order must in sorted order.
        :return:
        """
        # sequential order
        midi_seq = MidiNoteSequences()
        midi_seq.add_tempo_signature(MidiTempoSignature(qpm=140, midi_time=0))
        midi_seq.add_tempo_signature(MidiTempoSignature(qpm=140, midi_time=100))
        midi_seq.add_tempo_signature(MidiTempoSignature(qpm=140, midi_time=200))
        midi_seq.add_tempo_signature(MidiTempoSignature(qpm=145, midi_time=300))
        self.assertTrue(MidiNoteSequences.is_sorted(midi_seq.tempo_signatures, key=lambda x: x.midi_time))
        self.assertEqual(4, len(midi_seq.tempo_signatures))
        # last
        self.assertEqual(300, midi_seq.tempo_signatures[-1].midi_time)

        # reverse order
        midi_seq = MidiNoteSequences()
        midi_seq.add_tempo_signature(MidiTempoSignature(qpm=4, midi_time=300))
        midi_seq.add_tempo_signature(MidiTempoSignature(qpm=4, midi_time=200))
        midi_seq.add_tempo_signature(MidiTempoSignature(qpm=8, midi_time=100))
        midi_seq.add_tempo_signature(MidiTempoSignature(qpm=4, midi_time=0))

        self.assertTrue(MidiNoteSequences.is_sorted(midi_seq.tempo_signatures, key=lambda x: x.midi_time))
        self.assertEqual(4, len(midi_seq.tempo_signatures))
        self.assertEqual(0, midi_seq.tempo_signatures[0].midi_time, 0)
        self.assertEqual(300, midi_seq.tempo_signatures[-1].midi_time)

        # random order
        midi_seq = MidiNoteSequences()
        midi_seq.add_tempo_signature(MidiTempoSignature(qpm=4, midi_time=300))
        midi_seq.add_tempo_signature(MidiTempoSignature(qpm=4, midi_time=200))
        midi_seq.add_tempo_signature(MidiTempoSignature(qpm=8, midi_time=100))
        midi_seq.add_tempo_signature(MidiTempoSignature(qpm=4, midi_time=0))
        self.assertTrue(MidiNoteSequences.is_sorted(midi_seq.tempo_signatures, key=lambda x: x.midi_time))
        self.assertEqual(4, len(midi_seq.tempo_signatures))
        self.assertEqual(0, midi_seq.tempo_signatures[0].midi_time)
        self.assertEqual(300, midi_seq.tempo_signatures[-1].midi_time)

    def test_add_time_signatures_0(self):
        """Test that initially state we have default value in time_signatures
        as soon as we get a midi time signature with midi_time 0.0 we replace initial
        value.
        :return:
        """
        # sequential order
        midi_seq = MidiNoteSequences()
        # we must have default
        self.assertEqual(1, len(midi_seq.time_signatures))
        # we check for implicit numerator and denominator
        self.assertEqual(4, midi_seq.time_signatures[0].numerator)
        self.assertEqual(4, midi_seq.time_signatures[0].denominator)

        midi_seq.add_time_signatures(MidiTimeSignature(numerator=3, denominator=4, midi_time=0.0))
        self.assertTrue(MidiNoteSequences.is_sorted(midi_seq.tempo_signatures, key=lambda x: x.midi_time))
        self.assertEqual(1, len(midi_seq.time_signatures))
        self.assertEqual(3, midi_seq.time_signatures[-1].numerator)
        self.assertEqual(4, midi_seq.time_signatures[-1].denominator)

    def test_add_time_signatures_out_order(self):
        """Test that initially state we have default value in time_signatures
        as soon we first insert at time step 0.5, and then we receive 0.0.
        i.e. out of the order.  At the end the 0.0 should replace a initial one.
        value.
        :return:
        """
        # sequential order
        midi_seq = MidiNoteSequences()
        # we must have default
        self.assertEqual(1, len(midi_seq.time_signatures))
        # we check for implicit numerator and denominator
        self.assertEqual(4, midi_seq.time_signatures[0].numerator)
        self.assertEqual(4, midi_seq.time_signatures[0].denominator)

        # this should be added
        midi_seq.add_time_signatures(MidiTimeSignature(numerator=3, denominator=2, midi_time=0.5))
        self.assertTrue(MidiNoteSequences.is_sorted(midi_seq.tempo_signatures, key=lambda x: x.midi_time))

        # value should be 3 and 4
        self.assertEqual(2, len(midi_seq.time_signatures))
        self.assertEqual(3, midi_seq.time_signatures[-1].numerator)
        self.assertEqual(2, midi_seq.time_signatures[-1].denominator)

        # the top should be 4/4
        self.assertEqual(4, midi_seq.time_signatures[0].numerator)
        self.assertEqual(4, midi_seq.time_signatures[0].denominator)
        self.assertEqual(0.0, midi_seq.time_signatures[0].midi_time)

        # we get time signature for 0.25, and it should be 4/4
        denom, numerator = midi_seq.time_signature(0.25)
        self.assertEqual(4, denom, f"expected denominator {4}, got {denom} ts 0.0 is 0.0")
        self.assertEqual(4, numerator, f"expected numerator {4}, got {numerator} ts 0.0 is 0.0")

        # we add new time signature out of the order for 0.0
        midi_seq.add_time_signatures(MidiTimeSignature(numerator=8, denominator=8, midi_time=0.0))
        # let should be the same
        self.assertEqual(2, len(midi_seq.time_signatures))
        # top one should 8/8
        self.assertEqual(0.0, midi_seq.time_signatures[0].midi_time)
        self.assertEqual(8, midi_seq.time_signatures[0].numerator)
        self.assertEqual(8, midi_seq.time_signatures[0].denominator)

        # last the same
        self.assertEqual(3, midi_seq.time_signatures[-1].numerator)
        self.assertEqual(2, midi_seq.time_signatures[-1].denominator)

        # cache invalidate and we get update
        denom, numerator = midi_seq.time_signature(0.25)
        self.assertEqual(8, denom, f"expected denominator {8}, got {denom} ts 0.0 is 0.0")
        self.assertEqual(8, numerator, f"expected numerator {8}, got {numerator} ts 0.0 is 0.0")

    def test_add_tempo_signatures_out_order(self):
        """Test that initially state we have default value in tempo
        as soon we first insert at time step 0.5, and then we receive 0.0.
        i.e. out of the order.  At the end the 0.0 should replace initial one.
        value.

        This test simulate tempo change,
        :return:
        """
        # sequential order
        midi_seq = MidiNoteSequences()
        # we must have default
        self.assertEqual(1, len(midi_seq.tempo_signatures))
        # we check for implicit numerator and denominator
        self.assertEqual(DEFAULT_QPM, midi_seq.tempo_signatures[0].qpm)
        self.assertEqual(DEFAULT_PPQ, midi_seq.tempo_signatures[0].resolution)

        # this should be added so qpm at 0.5 is 90
        midi_seq.add_tempo_signature(MidiTempoSignature(qpm=90, midi_time=0.5))
        self.assertTrue(MidiNoteSequences.is_sorted(midi_seq.tempo_signatures, key=lambda x: x.midi_time))

        # value should be 3 and 4
        self.assertEqual(2, len(midi_seq.tempo_signatures))
        # last tempo added 0.5
        self.assertEqual(90, midi_seq.tempo_signatures[-1].qpm)

        # we use implicitly assumed QPM tempo
        self.assertEqual(DEFAULT_QPM, midi_seq.tempo_signatures[0].qpm)
        self.assertEqual(0.0, midi_seq.tempo_signatures[0].midi_time)

        # we get time signature for 0.25, and it should be 4/4
        qpm, _ = midi_seq.tempo_signature(0.25)
        self.assertEqual(DEFAULT_QPM, qpm, f"expected qpm {DEFAULT_QPM}, got {qpm} ts 0.0 is 0.0")

        # add new tempo signature 80 at 0.0
        midi_seq.add_tempo_signature(MidiTempoSignature(qpm=80, midi_time=0.0))
        self.assertEqual(80, midi_seq.tempo_signatures[0].qpm)
        self.assertEqual(0.0, midi_seq.tempo_signatures[0].midi_time)
        # assert len so we replace 0.0
        self.assertEqual(2, len(midi_seq.tempo_signatures))

        # old one should be still there but cache must invalidated
        self.assertEqual(90, midi_seq.tempo_signatures[-1].qpm)
        qpm, _ = midi_seq.tempo_signature(0.25)
        self.assertEqual(80, qpm, f"expected qpm {80}, got {qpm} ts 0.0 is 0.0")

    def test_add_tempo_signatures_0(self):
        """Test that initially state we have default value in tempo_signature
        as soon as we get a midi tempo signature with midi_time 0.0 we replace initial
        value.
        :return:
        """
        # sequential order
        midi_seq = MidiNoteSequences()
        self.assertEqual(1, len(midi_seq.tempo_signatures))
        # we check for implicit tempo
        self.assertEqual(120, midi_seq.tempo_signatures[0].qpm)
        self.assertEqual(120, midi_seq.tempo_signatures[0].qpm)

        midi_seq.add_tempo_signature(MidiTempoSignature(qpm=130, midi_time=0.0))
        self.assertTrue(MidiNoteSequences.is_sorted(midi_seq.tempo_signatures, key=lambda x: x.midi_time))
        self.assertEqual(1, len(midi_seq.tempo_signatures))
        self.assertEqual(130, midi_seq.tempo_signatures[-1].qpm)

    def test_add_time_signatures02(self):
        """Test that `add_time_signatures` correctly inserts new `MidiTimeSignature`
        instances into the `time_signatures` list of a `MidiNoteSequences` instance
        in order of increasing `midi_time`.
        :return:
        """
        # sequential order
        midi_seq = MidiNoteSequences()
        midi_seq.add_time_signatures(MidiTimeSignature(midi_time=4, numerator=4, denominator=4))
        midi_seq.add_time_signatures(MidiTimeSignature(midi_time=100, numerator=3, denominator=4))
        midi_seq.add_time_signatures(MidiTimeSignature(midi_time=200, numerator=6, denominator=8))
        midi_seq.add_time_signatures(MidiTimeSignature(midi_time=300, numerator=5, denominator=4))
        self.assertTrue(midi_seq.is_sorted(midi_seq.time_signatures, key=lambda x: x.midi_time))
        self.assertEqual(5, len(midi_seq.time_signatures))
        self.assertEqual(300, midi_seq.time_signatures[-1].midi_time)

    def test_calculate_min_step(self):
        """Tests the `calculate_min_step` method of the `MidiNoteSequence`
        class with two notes that have a gap of 1.5."""
        seq = MidiNoteSequence()
        seq.add_note(MidiNote(pitch=60, velocity=100, start_time=0.0, end_time=1.0))
        seq.add_note(MidiNote(pitch=62, velocity=100, start_time=1.5, end_time=2.0))
        self.assertTrue(seq[0].start_time == 0.0 and seq[0].end_time == 1.0)
        self.assertTrue(seq[1].start_time == 1.5 and seq[1].end_time == 2.0)
        self.assertTrue(seq.calculate_min_step() == 1.5, "min step expected 1.5")

    def test_calculate_min_step02(self):
        """Tests the `calculate_min_step` method of the `MidiNoteSequence`
        class with two notes that have a gap of 1.5."""
        seq = MidiNoteSequence()
        seq.add_note(MidiNote(pitch=60, velocity=100, start_time=0.0, end_time=1.0))
        seq.add_note(MidiNote(pitch=62, velocity=100, start_time=1.5, end_time=2.0))
        self.assertTrue(seq[0].start_time == 0.0 and seq[0].end_time == 1.0)
        self.assertTrue(seq[1].start_time == 1.5 and seq[1].end_time == 2.0)
        self.assertTrue(seq.calculate_min_step() == 1.5, "min step expected 1.5")

    @staticmethod
    def generate_midi_sequence(
            num_notes: int = 10,
            min_pitch: int = 60,
            max_pitch: int = 72,
            velocity: int = 64,
            duration: float = 0.5,
            random_start_end_time: bool = False,
            resolution: int = 220,
            is_drum: bool = False,
            instrument_number: int = 0,
            instrument_name: str = 'Acoustic Grand Piano'
    ) -> MidiNoteSequence:
        """
        Generate note for unit test
        :param num_notes:
        :param min_pitch:
        :param max_pitch:
        :param velocity:
        :param duration:
        :param random_start_end_time:
        :param resolution:
        :param is_drum:
        :param instrument_number:
        :param instrument_name:
        :return:
        """
        notes = []
        for i in range(num_notes):
            pitch = random.randint(min_pitch, max_pitch)
            start_time = random.uniform(0, duration) if random_start_end_time else (i * duration)
            end_time = random.uniform(start_time, duration) if random_start_end_time else ((i + 1) * duration)
            note = MidiNote(pitch=pitch, start_time=start_time, end_time=end_time, velocity=velocity, is_drum=is_drum)
            notes.append(note)

        instrument_info = MidiInstrumentInfo(instrument_number, instrument_name)
        midi_seq = MidiNoteSequence(notes=notes, instrument=instrument_info, resolution=resolution)
        return midi_seq

    def test_create_with_list(self):
        """Test that a MidiNoteSequences object
        can be created with a list of MidiNoteSequence objects.
        :return:
        """
        midi_seq1 = self.generate_midi_sequence(num_notes=10, random_start_end_time=True, instrument_number=0)
        midi_seq2 = self.generate_midi_sequence(num_notes=10, random_start_end_time=True, instrument_number=1)
        self.midi2 = MidiNoteSequence()
        # a list of MidiNoteSequence objects
        self.seq1 = [midi_seq1, midi_seq2]

        midi_seq = MidiNoteSequences(midi_seq=self.seq1)
        # that the MidiNoteSequences object contains both MidiNoteSequence objects
        self.assertEqual(len(midi_seq.midi_seqs), 2)
        self.assertIn(midi_seq1, midi_seq.midi_seqs.values())
        self.assertIn(midi_seq2, midi_seq.midi_seqs.values())

    def test_init_single_seq(self):
        """Test creating a `MidiNoteSequences`
        object with a single `MidiNoteSequence` object.
        :return:
        """
        seq = MidiNoteSequence()
        mns = MidiNoteSequences(midi_seq=seq)
        self.assertIsInstance(mns.midi_seqs, dict)
        self.assertEqual(len(mns.midi_seqs), 1)
        self.assertTrue(0 in mns.midi_seqs)
        self.assertEqual(mns.midi_seqs[0], seq)

    def test_instruments(self):
        """
        :return:
        """
        midi_seq1 = self.generate_midi_sequence(num_notes=10, random_start_end_time=True, instrument_number=0)
        midi_seq2 = self.generate_midi_sequence(num_notes=10, random_start_end_time=True, instrument_number=1)
        self.midi2 = MidiNoteSequence()
        # a list of MidiNoteSequence objects
        self.seq1 = [midi_seq1, midi_seq2]

        midi_seq = MidiNoteSequences(midi_seq=self.seq1)
        # that the MidiNoteSequences object contains both MidiNoteSequence objects
        self.assertEqual(len(midi_seq.midi_seqs), 2)
        self.assertIn(midi_seq1, midi_seq.midi_seqs.values())
        self.assertIn(midi_seq2, midi_seq.midi_seqs.values())

        self.assertEqual(len(midi_seq.instruments), 2)
        self.assertEqual(midi_seq.instruments[0].instrument_num, 0)
        self.assertEqual(midi_seq.instruments[1].instrument_num, 1)

    def test_ordered_dict(self):
        """Test that MidiNoteSequences object preserves
        the order of MidiNoteSequence objects according to their instrument
        number when a list of MidiNoteSequence objects is passed to its constructor.
        :return:
        """
        midi_seq1 = self.generate_midi_sequence(num_notes=10, random_start_end_time=True, instrument_number=0)
        midi_seq2 = self.generate_midi_sequence(num_notes=10, random_start_end_time=True, instrument_number=1)
        midi_seq3 = self.generate_midi_sequence(num_notes=10, random_start_end_time=True, instrument_number=2)
        midi_seq = [midi_seq2, midi_seq1, midi_seq3]
        mns = MidiNoteSequences(midi_seq=midi_seq)
        self.assertEqual(list(mns.midi_seqs.values()), [midi_seq1, midi_seq2, midi_seq3])

    def test_sorted_order_instruments(self):
        """
        :return:
        """
        midi_seq1 = self.generate_midi_sequence(num_notes=10, random_start_end_time=True, instrument_number=0)
        midi_seq2 = self.generate_midi_sequence(num_notes=10, random_start_end_time=True, instrument_number=1)
        midi_seq3 = self.generate_midi_sequence(num_notes=10, random_start_end_time=True, instrument_number=2)
        self.seq1 = [midi_seq2, midi_seq1, midi_seq3]  # intentionally out of order
        midi_seq = MidiNoteSequences(midi_seq=self.seq1)

        # Check that the MidiNoteSequences object contains all three MidiNoteSequence objects
        self.assertEqual(len(midi_seq.midi_seqs), 3)
        self.assertIn(midi_seq1, midi_seq.midi_seqs.values())
        self.assertIn(midi_seq2, midi_seq.midi_seqs.values())
        self.assertIn(midi_seq3, midi_seq.midi_seqs.values())

        # Check that the instruments attribute is sorted by instrument number
        self.assertEqual(len(midi_seq.instruments), 3)
        self.assertEqual(midi_seq.instruments[0].instrument_num, 0)
        self.assertEqual(midi_seq.instruments[1].instrument_num, 1)
        self.assertEqual(midi_seq.instruments[2].instrument_num, 2)

    def test_init(self):
        """Test the `__init__` method of the `MidiNoteSequence` class."""
        midi_seq = MidiNoteSequence()
        self.assertIsInstance(midi_seq, MidiNoteSequence)
        self.assertIsInstance(midi_seq.notes, list)
        self.assertEqual(len(midi_seq.notes), 0)
        self.assertIsNotNone(midi_seq.instrument)

    def test_index_access_on_empty(self):
        """
        :return:
        """
        seq = MidiNoteSequences()
        self.assertEqual(0, len(seq))
        x = seq[10]
        self.assertEqual(1, len(seq))
        self.assertIsInstance(x, MidiNoteSequence)

        y = seq[10]
        self.assertEqual(1, len(seq))
        self.assertIs(x, y)
        self.assertIsInstance(y, MidiNoteSequence)
        self.assertEqual(10, y.instrument.instrument_num)

        # modify
        seq[10].total_time = 2
        self.assertEqual(seq[10].total_time, 2)

        # get again
        y = seq[10]
        self.assertEqual(1, len(seq))
        self.assertIs(x, y)
        self.assertIsInstance(y, MidiNoteSequence)
        self.assertEqual(10, y.instrument.instrument_num)
        self.assertEqual(seq[10].total_time, 2)

    def test_truncate(self):

        # Create a MidiNoteSequence with two tracks
        seq = MidiNoteSequences()
        track_1 = seq.create_track(0, 0, "Piano", False)
        track_2 = seq.create_track(1, 40, "Violin", False)

        print(type(seq[0]))

        #
        # seq.midi_seqs[track_2] = [
        #     MidiNote(72, 0, 1, velocity=100),
        #     MidiNote(74, 2, 3, velocity=100),
        #     MidiNote(76, 4, 5, velocity=100),
        #     MidiNote(77, 6, 7, velocity=100),
        # ]
        #
        # self.assertEqual(4, len(seq[track_1]))
        # self.assertEqual(4, len(seq[track_2]))
        #
        # print(seq.midi_seqs)

        # # Truncate the sequence from the start to 3 seconds
        # last_event_time = seq.truncate(from_start=3.0)
        #
        # # Assert that the last event time is correct
        # self.assertEqual(last_event_time, 2.5)

        # #
        # # # Assert that the notes after the truncation are removed
        # expected_notes_1 = [MidiNote(60, 0, 1, velocity=100)]
        # expected_notes_2 = [
        #     MidiNote(72, 0, 1, velocity=100),
        #     MidiNote(74, 2, 3, velocity=100),
        # ]
        # assert seq.get_track(track_1).get_notes() == expected_notes_1
        # assert seq.get_track(track_2).get_notes() == expected_notes_2
        #
        # # Truncate the sequence from the start to 4 seconds
        # last_event_time = seq.truncate(from_start=4.0)
        #
        # # Assert that the last event time is correct
        # assert last_event_time == 3.5

        # Assert that all notes are removed after the truncation
        # expected_notes_1 = [MidiNote(60, 0, 1, velocity=100)]
        # expected_notes_2 = [
        #     MidiNote(72, 0, 1, velocity=100),
        # ]
        # assert seq.get_track(track_1).get_notes() == expected_notes_1
        # assert seq.get_track(track_2).get_notes() == expected_notes_2



