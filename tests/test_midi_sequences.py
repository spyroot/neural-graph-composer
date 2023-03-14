from unittest import TestCase

from neural_graph_composer.midi.midi_note import MidiNote
from neural_graph_composer.midi.midi_sequence import MidiNoteSequence
from neural_graph_composer.midi.midi_sequences import MidiNoteSequences
from neural_graph_composer.midi.midi_spec import DEFAULT_QPM, DEFAULT_PPQ
from neural_graph_composer.midi.midi_time_signature import MidiTimeSignature, MidiTempoSignature


class Test(TestCase):

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

    # def test_add_time_signatures(self):
    #     """Test that `add_time_signatures` correctly inserts new `MidiTimeSignature`
    #     instances into the `time_signatures` list of a `MidiNoteSequences` instance
    #     in order of increasing `midi_time`.
    #     :return:
    #     """
    #     # sequential order
    #     midi_seq = MidiNoteSequences()
    #     midi_seq.add_time_signatures(MidiTimeSignature(4, 4, 0))
    #     midi_seq.add_time_signatures(MidiTimeSignature(4, 3, 100))
    #     midi_seq.add_time_signatures(MidiTimeSignature(8, 6, 200))
    #     midi_seq.add_time_signatures(MidiTimeSignature(4, 5, 300))
    #     self.assertTrue(midi_seq.is_sorted(key=lambda x: x.midi_time))
    #     self.assertEqual(4, len(midi_seq.time_signatures))
    #     self.assertEqual(300, midi_seq.time_signatures[-1].midi_time)
    #
    #     midi_seq = MidiNoteSequences()
    #     midi_seq.add_time_signatures(MidiTimeSignature(4, 4, 300))
    #     midi_seq.add_time_signatures(MidiTimeSignature(4, 3, 200))
    #     midi_seq.add_time_signatures(MidiTimeSignature(8, 6, 100))
    #     midi_seq.add_time_signatures(MidiTimeSignature(4, 5, 0))
    #     self.assertTrue(midi_seq.is_sorted(key=lambda x: x.midi_time))
    #     self.assertEqual(4, len(midi_seq.time_signatures))
    #     self.assertEqual(0, midi_seq.time_signatures[0].midi_time, 0)
    #     self.assertEqual(300, midi_seq.time_signatures[-1].midi_time)
    #
    #     # random order
    #     midi_seq = MidiNoteSequences()
    #     midi_seq.add_time_signatures(MidiTimeSignature(4, 4, 300))
    #     midi_seq.add_time_signatures(MidiTimeSignature(4, 3, 200))
    #     midi_seq.add_time_signatures(MidiTimeSignature(8, 6, 100))
    #     midi_seq.add_time_signatures(MidiTimeSignature(4, 5, 0))
    #     self.assertTrue(midi_seq.is_sorted(key=lambda x: x.midi_time))
    #     self.assertEqual(4, len(midi_seq.time_signatures))
    #     self.assertEqual(0, midi_seq.time_signatures[0].midi_time)
    #     self.assertEqual(300, midi_seq.time_signatures[-1].midi_time)

    # self.assertEqual(midi_seq.time_signatures[0].numerator, 4)
    # self.assertEqual(midi_seq.time_signatures[0].denominator, 3)
    # self.assertEqual(midi_seq.time_signatures[1].numerator, 4)
    # self.assertEqual(midi_seq.time_signatures[1].denominator, 4)
    # self.assertEqual(midi_seq.time_signatures[2].numerator, 4)
    # self.assertEqual(midi_seq.time_signatures[2].denominator, 5)
    # self.assertEqual(midi_seq.time_signatures[3].numerator, 8)
    # self.assertEqual(midi_seq.time_signatures[3].denominator, 6)

    def test_calculate_min_step(self):
        """Tests the `calculate_min_step` method of the `MidiNoteSequence`
        class with two notes that have a gap of 1.5."""
        seq = MidiNoteSequence()
        seq.add_note(MidiNote(pitch=60, velocity=100, start_time=0.0, end_time=1.0))
        seq.add_note(MidiNote(pitch=62, velocity=100, start_time=1.5, end_time=2.0))
        self.assertTrue(seq[0].start_time == 0.0 and seq[0].end_time == 1.0)
        self.assertTrue(seq[1].start_time == 1.5 and seq[1].end_time == 2.0)
        self.assertTrue(seq.calculate_min_step() == 1.5, "min step expected 1.5")
