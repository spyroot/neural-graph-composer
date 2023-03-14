from unittest import TestCase

from neural_graph_composer.midi.midi_note import MidiNote
from neural_graph_composer.midi.midi_sequence import MidiNoteSequence
from neural_graph_composer.midi.midi_instruments import MidiInstrumentInfo


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
