import math
from unittest import TestCase

from neural_graph_composer.midi.midi_note import MidiNote
from neural_graph_composer.midi.midi_sequence import MidiNoteSequence
from neural_graph_composer.midi.midi_instruments import MidiInstrumentInfo
from tests.test_utils import generate_notes
import copy


class Test(TestCase):
    # def test_midi_compute_slice(self):
    #     channel_one = [60, 62, 64, 65, 67, 69, 71, 72]
    #     channel_two = [61, 63, 65, 66, 68, 70, 72, 73]
    #     midi_seqs = MidiReader.midi_to_tensor("../neural_graph_composer/dataset/unit_test/test-c-major-scale.mid")
    #     midi_seq = midi_seqs[0]
    #     self.assertIsInstance(midi_seq, MidiNoteSequence)
    #
    #     seq_sorted_by_time = sorted(
    #         list(midi_seq._notes),
    #         key=lambda note: note.start_time)
    #
    #     window_in_seconds = [1.0, 1.0]
    #     segments = MidiReader.midi_seq.split(window_in_seconds)
    #     print(len(segments))
    #     # print(midi_seqs[0].notes[0].is_drum)

    def test_calculate_min_step(self):
        """Tests the `calculate_min_step` method of the `MidiNoteSequence`
        class with two notes that have a gap of 1.5."""
        seq = MidiNoteSequence()
        seq.add_note(MidiNote(pitch=60, velocity=100, start_time=0.0, end_time=1.0))
        seq.add_note(MidiNote(pitch=62, velocity=100, start_time=1.5, end_time=2.0))
        self.assertTrue(seq[0].start_time == 0.0 and seq[0].end_time == 1.0)
        self.assertTrue(seq[1].start_time == 1.5 and seq[1].end_time == 2.0)
        self.assertTrue(seq.calculate_min_step() == 1.5, "min step expected 1.5")

    def test_calculate_min_step_zero_distance(self):
        """Tests the `calculate_min_step` method of the `MidiNoteSequence`
        class with two notes that have a gap of 0.0.
        :return:
        """
        seq = MidiNoteSequence()
        seq.add_note(MidiNote(pitch=60, velocity=100, start_time=0.0, end_time=1.0))
        seq.add_note(MidiNote(pitch=62, velocity=100, start_time=1.0, end_time=2.0))
        self.assertTrue(seq[0].start_time == 0.0 and seq[0].end_time == 1.0)
        self.assertTrue(seq[1].start_time == 1.0 and seq[1].end_time == 2.0)
        print(seq.calculate_min_step())
        self.assertTrue(seq.calculate_min_step() == 0.0, "min step expected 0.0")

    def test_calculate_min_step_single_note(self):
        """Tests the `calculate_min_step` method
        of the `MidiNoteSequence` class with a single note."
        :return:
        """
        seq = MidiNoteSequence()
        seq.add_note(MidiNote(pitch=60, velocity=100, start_time=0.0, end_time=1.0))
        self.assertTrue(seq.calculate_min_step() == 0.0, "min step expected 0.0")

    def test_calculate_min_step_empty_sequence(self):
        """test min step on empty seq
        :return:
        """
        seq = MidiNoteSequence()
        self.assertEqual(seq.calculate_min_step(), 0.0, "min step expected 0.0")

    def test_update_total_time02(self):
        """Add single note in midi seq
        :return:
        """
        # create two note same pitch different in time
        # connect and
        a = MidiNote(pitch=51, start_time=0, end_time=0.5, velocity=127)
        expect = [a]
        midi_seq = MidiNoteSequence(notes=expect)
        self.assertTrue(midi_seq.total_time == 0.5)
        self.assertTrue(len(midi_seq.notes) == len(expect))
        self.assertListEqual(midi_seq.notes, expect)

    def test_update_total_time03(self):
        """Add two note and make sure total time updated.
        :return:
        """
        # create two note same pitch different in time
        # connect and
        a = MidiNote(pitch=51, start_time=0, end_time=0.5)
        b = MidiNote(pitch=51, start_time=0.5, end_time=1.0)
        expect = [a, b]
        midi_seq = MidiNoteSequence(notes=expect)
        self.assertTrue(midi_seq.total_time == 1.0)
        self.assertTrue(len(midi_seq.notes) == len(expect))
        self.assertListEqual(midi_seq.notes, expect)

    def test_midi_seq_set_instrument01(self):
        """set instrument during construction
        :return:
        """
        instrument = MidiInstrumentInfo(0, "Generic", is_drum=False)
        midi_seq = MidiNoteSequence(instrument=instrument)
        self.assertTrue(midi_seq.instrument == instrument)

    def test_midi_seq_set_instrument02(self):
        """set none drum instrument set pitch
        during construction and add single note
        it should in drum event vs normal notes
        :return:
        """
        # create none drum instrument
        instrument = MidiInstrumentInfo(0, "Generic", is_drum=False)
        a = MidiNote(pitch=51, start_time=0, end_time=0.5)
        expect = [a]
        midi_seq = MidiNoteSequence(notes=expect, instrument=instrument)
        self.assertTrue(midi_seq.instrument == instrument)
        self.assertListEqual(midi_seq.notes, expect)
        self.assertTrue(len(midi_seq._drum_events) == 0)

    def test_midi_seq_set_instrument03(self):
        """set instrument drum during construction and add single note
        it should in drum event vs normal notes
        :return:
        """
        # create none drum instrument
        instrument = MidiInstrumentInfo(0, "Generic", is_drum=True)
        a = MidiNote(pitch=51, start_time=0, end_time=0.5)
        expect = [a]
        midi_seq = MidiNoteSequence(notes=expect, instrument=instrument)
        self.assertTrue(midi_seq.instrument.is_drum)
        self.assertTrue(midi_seq.instrument == instrument)
        self.assertListEqual(midi_seq._drum_events, expect)
        self.assertListEqual(midi_seq.notes, expect)

    def test_midi_seq_add_note(self):
        """add note and check that it added
        :return:
        """
        # create none drum instrument
        a = MidiNote(pitch=51, start_time=0, end_time=0.5, is_drum=False)
        midi_seq = MidiNoteSequence()
        midi_seq.add_note(a)
        expect = [a]
        self.assertListEqual(midi_seq._notes, expect)

    def test_midi_seq_add_drum_note(self):
        """add drum instrument drum during construction and add single note
        it should in drum event vs normal notes
        :return:
        """
        # create none drum instrument
        a = MidiNote(pitch=51, start_time=0, end_time=0.5, is_drum=True)
        midi_seq = MidiNoteSequence()
        midi_seq.add_note(a)
        expect = [a]
        self.assertListEqual(midi_seq._drum_events, expect)

    def test_add_drum_note_update_time(self):
        """Add drum note and check time updated if total time <
        :return:
        """
        # create none drum instrument
        a = MidiNote(pitch=51, start_time=0, end_time=0.5, is_drum=True)
        midi_seq = MidiNoteSequence()
        midi_seq.add_note(a)
        expect = [a]
        self.assertListEqual(midi_seq._drum_events, expect)
        self.assertTrue(midi_seq.total_time == 0.5)

    def test_add_note_update_time(self):
        """Add note and check time updated if total time <
        :return:
        """
        # create none drum instrument
        a = MidiNote(pitch=51, start_time=0, end_time=0.5, is_drum=False)
        midi_seq = MidiNoteSequence()
        midi_seq.add_note(a)
        expect = [a]
        self.assertListEqual(midi_seq._notes, expect)
        self.assertTrue(midi_seq.total_time == 0.5)

    def test_update_total_time04(self):
        """Add multiple notes in midi seq with time > 0
        :return:
        """
        # create two note same pitch different in time
        # connect and
        a = MidiNote(pitch=51, start_time=0.0, end_time=0.5)
        b = MidiNote(pitch=51, start_time=0.5, end_time=1.0)
        expect = [a, b]
        midi_seq = MidiNoteSequence(notes=expect)
        self.assertTrue(midi_seq.total_time == 1.0)
        self.assertListEqual(midi_seq.notes, expect)

    def test_quantize(self):
        """
        :return:
        """
        seq = MidiNoteSequence()
        seq.add_note(MidiNote(pitch=60, velocity=100, start_time=0.2, end_time=1.4))
        seq.add_note(MidiNote(pitch=62, velocity=100, start_time=1.5, end_time=2.3))
        self.assertEqual(seq[0].start_time, 0.2)
        self.assertEqual(seq[0].end_time, 1.4)
        self.assertEqual(seq[1].start_time, 1.5)
        self.assertEqual(seq[1].end_time, 2.3)
        seq.quantize(2)
        self.assertEqual(seq[0].start_time, 0)
        self.assertEqual(seq[0].end_time, 1.5)
        self.assertEqual(seq[1].start_time, 1.5)
        self.assertEqual(seq[1].end_time, 2.5)

    def test_shift_times(self):
        # create a midi note sequence
        notes = list(generate_notes(10, 4, 60))
        last_time = max([n.end_time for n in notes])
        notes_copy = copy.deepcopy(notes)

        # print(notes)
        seq = MidiNoteSequence(notes)
        self.assertTrue(last_time, seq.total_time)

        # # Shift the notes by 2.0
        seq.shift_times(2.0)

        # # Assert that all notes have been shifted by 2.0
        for i, n in enumerate(notes_copy):
            self.assertAlmostEqual(n.start_time + 2.0, seq.notes[i].start_time,
                                   msg=f"expect new time {n.start_time + 2.0} got {seq.notes[i].start_time}")
            self.assertAlmostEqual(n.end_time + 2.0, seq.notes[i].end_time,
                                   msg=f"expect new time {n.end_time + 2.0} got {seq.notes[i].end_time}")

    def test_from_notes(self):
        # Generate a list of MidiNote objects
        notes = list(generate_notes(10, 4, 60))
        # Create a MidiNoteSequence from the notes list
        seq = MidiNoteSequence.from_notes(notes)

        # Check that all the notes were added to the sequence
        self.assertEqual(len(seq.notes), len(notes))
        # Check that the start and end times of the sequence are correct
        self.assertEqual(seq.notes[0].start_time, 0.0)
        self.assertEqual(seq.notes[-1].end_time, seq.total_time)

    def test_slice_notes(self):
        # create a test sequence with some notes
        seq = MidiNoteSequence.from_notes([
            MidiNote(60, 0.0, 1.0, velocity=80),
            MidiNote(62, 1.0, 2.0, velocity=90),
            MidiNote(64, 2.0, 3.0, velocity=100),
            MidiNote(65, 3.0, 4.0, velocity=110),
            MidiNote(67, 4.0, 5.0, velocity=120),
        ])

        sliced_seq = seq.slice(1.0, 3.0)
        #
        # # expected sliced sequence
        expected_seq = MidiNoteSequence.from_notes([
            MidiNote(62, 1.0, 2.0, velocity=90),
            MidiNote(64, 2.0, 3.0, velocity=100),
        ])

        # check that the sliced sequence matches the expected sequence
        self.assertEqual(sliced_seq.notes, expected_seq.notes)
        self.assertAlmostEqual(expected_seq.total_time, 3.0)

    def test_midi_note_sequence_comparison(self):
        """
        :return:
        """
        midi_data1 = MidiNoteSequence(instrument=MidiInstrumentInfo(instrument=1, is_drum=False, name='Piano'))
        midi_data2 = MidiNoteSequence(instrument=MidiInstrumentInfo(instrument=2, is_drum=False, name='Guitar'))
        midi_data3 = MidiNoteSequence(instrument=MidiInstrumentInfo(instrument=3, is_drum=False, name='Violin'))
        midi_data4 = MidiNoteSequence(instrument=MidiInstrumentInfo(instrument=4, is_drum=False, name='Bass'))

        self.assertLess(midi_data1, midi_data2)
        self.assertLess(midi_data2, midi_data3)
        self.assertLess(midi_data3, midi_data4)
        self.assertGreater(midi_data4, midi_data3)
        self.assertGreater(midi_data3, midi_data2)
        self.assertGreater(midi_data2, midi_data1)

    def test_insert(self):
        # Create a MidiNoteSequence object with some notes
        notes = [
            MidiNote(pitch=60, velocity=100, start_time=0, end_time=1),
            MidiNote(pitch=64, velocity=100, start_time=0, end_time=2),
        ]
        seq = MidiNoteSequence(notes=notes, instrument=1)

        # Insert a new note
        new_note = MidiNote(pitch=67, velocity=100, start_time=1.5, end_time=2)
        seq.insert(new_note)

        # Check that the note was inserted at the correct position
        self.assertEqual(len(seq.notes), 3)
        self.assertEqual(seq.notes[0], MidiNote(pitch=60, velocity=100, start_time=0, end_time=1))
        self.assertEqual(seq.notes[1], MidiNote(pitch=64, velocity=100, start_time=0, end_time=2))
        self.assertEqual(seq.notes[2], MidiNote(pitch=67, velocity=100, start_time=1.5, end_time=2))

    def test_insert02(self):
        # Create a MidiNoteSequence object with some notes
        notes = [
            MidiNote(pitch=60, velocity=100, start_time=0, end_time=1),
            MidiNote(pitch=64, velocity=100, start_time=0, end_time=2),
            MidiNote(pitch=67, velocity=100, start_time=1, end_time=3)
        ]

        seq = MidiNoteSequence(notes=notes, instrument=1)
        self.assertEqual(len(seq.notes), 3)

        # Insert a note at the beginning of the sequence
        new_note1 = MidiNote(pitch=62, velocity=100, start_time=0, end_time=1)
        seq.insert(new_note1)
        self.assertEqual(len(seq.notes), 4)
        self.assertEqual(seq.notes[0], new_note1)

        # Insert a note in the middle of the sequence
        new_note2 = MidiNote(pitch=66, velocity=100, start_time=0.5, end_time=1.5)
        seq.insert(new_note2)
        self.assertEqual(len(seq.notes), 5)
        self.assertEqual(seq.notes[2], new_note2)

        # Insert a note at the end of the sequence
        new_note3 = MidiNote(pitch=69, velocity=100, start_time=3, end_time=4)
        seq.insert(new_note3)
        self.assertEqual(len(seq.notes), 6)
        self.assertEqual(seq.notes[5], new_note3)

        # Insert a note that starts before the first note
        new_note4 = MidiNote(pitch=71, velocity=100, start_time=-1, end_time=0.5)
        seq.insert(new_note4)
        self.assertEqual(len(seq.notes), 7)
        self.assertEqual(seq.notes[0], new_note4)

        # Insert a note that starts after the last note
        new_note5 = MidiNote(pitch=72, velocity=100, start_time=4, end_time=5)
        seq.insert(new_note5)
        self.assertEqual(len(seq.notes), 8)
        self.assertEqual(seq.notes[7], new_note5)

        # Insert a note that has the same start time as an existing note
        new_note6 = MidiNote(pitch=74, velocity=100, start_time=1, end_time=2)
        seq.insert(new_note6)
        self.assertEqual(len(seq.notes), 9)
        self.assertIn(new_note6, seq.notes)

    def test_truncate(self):
        # Create a MidiNoteSequence object with some notes
        notes = [
            MidiNote(pitch=60, velocity=100, start_time=0.0, end_time=1.0),
            MidiNote(pitch=64, velocity=100, start_time=0.0, end_time=2.0),
            MidiNote(pitch=67, velocity=100, start_time=1.0, end_time=3.0)
        ]

        seq = MidiNoteSequence(notes=notes, instrument=1)
        self.assertEqual(len(notes), 3)

        # truncate the sequence to end at time 2
        truncated_seq = seq.truncate(2.0)
        print(truncated_seq)
        self.assertEqual(len(truncated_seq.notes), 2)
        print(truncated_seq.notes)

        self.assertEqual(truncated_seq.notes[0], MidiNote(pitch=60, velocity=100, start_time=0.0, end_time=1.0))
        self.assertEqual(truncated_seq.notes[1], MidiNote(pitch=64, velocity=100, start_time=0.0, end_time=2.0))

        # Check that the original sequence was not modified
        self.assertEqual(len(seq.notes), 3)
        self.assertEqual(truncated_seq.notes[0], MidiNote(pitch=60, velocity=100, start_time=0, end_time=1))
        self.assertEqual(seq.notes[1], MidiNote(pitch=64, velocity=100, start_time=0.0, end_time=2.0))
        self.assertEqual(seq.notes[2], MidiNote(pitch=67, velocity=100, start_time=1.0, end_time=3.0))

    def test_cut_in_place(self):
        # Create a MidiNoteSequence object with some notes
        notes = [
            MidiNote(pitch=60, velocity=100, start_time=0, end_time=1),
            MidiNote(pitch=64, velocity=100, start_time=0, end_time=2),
            MidiNote(pitch=67, velocity=100, start_time=1, end_time=3)
        ]

        seq = MidiNoteSequence(notes=notes, instrument=1)
        self.assertEqual(len(seq.notes), 3)

        # Cut the sequence in place
        seq.cut_in_place(2)

        # Check that the sequence was truncated correctly
        self.assertEqual(len(seq.notes), 2)
        self.assertEqual(seq.notes[0], MidiNote(pitch=60, velocity=100, start_time=0, end_time=1))
        self.assertEqual(seq.notes[1], MidiNote(pitch=64, velocity=100, start_time=0, end_time=2))

    def test_cut(self):
        # Create a MidiNoteSequence object with some notes
        notes = [
            MidiNote(pitch=60, velocity=100, start_time=0, end_time=1),
            MidiNote(pitch=64, velocity=100, start_time=0, end_time=2),
            MidiNote(pitch=67, velocity=100, start_time=1, end_time=3)
        ]

        seq = MidiNoteSequence(notes=notes, instrument=1)
        self.assertEqual(len(seq.notes), 3)

        # Cut the sequence
        truncated_seq = seq.cut(2)

        # Check that the original sequence was not modified
        self.assertEqual(len(seq.notes), 3)
        self.assertEqual(seq.notes[0], MidiNote(pitch=60, velocity=100, start_time=0, end_time=1))
        self.assertEqual(seq.notes[1], MidiNote(pitch=64, velocity=100, start_time=0, end_time=2))
        self.assertEqual(seq.notes[2], MidiNote(pitch=67, velocity=100, start_time=1, end_time=3))

        # Check that the truncated sequence is correct
        self.assertEqual(len(truncated_seq.notes), 2)
        self.assertEqual(truncated_seq.notes[0], MidiNote(pitch=60, velocity=100, start_time=0, end_time=1))
        self.assertEqual(truncated_seq.notes[1], MidiNote(pitch=64, velocity=100, start_time=0, end_time=2))
        self.assertEqual(truncated_seq.total_time, 2.0)

    def test_merge(self):
        notes1 = [
            MidiNote(pitch=60, velocity=100, start_time=0, end_time=1, instrument=1),
            MidiNote(pitch=64, velocity=100, start_time=0, end_time=2, instrument=1),
            MidiNote(pitch=67, velocity=100, start_time=1, end_time=3, instrument=1)
        ]

        notes2 = [
            MidiNote(pitch=72, velocity=100, start_time=0, end_time=1.5, instrument=1),
            MidiNote(pitch=76, velocity=100, start_time=0, end_time=3, instrument=1),
            MidiNote(pitch=79, velocity=100, start_time=1, end_time=4, instrument=1)
        ]

        seq1 = MidiNoteSequence(notes=notes1, instrument=1)
        seq2 = MidiNoteSequence(notes=notes2, instrument=1)

        seq1.merge(seq2)

        expected_notes = [
            MidiNote(pitch=60, velocity=100, start_time=0, end_time=1, instrument=1),
            MidiNote(pitch=64, velocity=100, start_time=0, end_time=2, instrument=1),
            MidiNote(pitch=72, velocity=100, start_time=0, end_time=1.5, instrument=1),
            MidiNote(pitch=67, velocity=100, start_time=1, end_time=3, instrument=1),
            MidiNote(pitch=76, velocity=100, start_time=0, end_time=3, instrument=1),
            MidiNote(pitch=79, velocity=100, start_time=1, end_time=4, instrument=1)
        ]

        self.assertEqual(len(seq1.notes), 6)
        self.assertListEqual(seq1.notes, expected_notes)
        self.assertEqual(seq1.total_time, 4)
        self.assertEqual(seq1.instrument, 1)








