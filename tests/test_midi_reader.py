import os
from unittest import TestCase
from neural_graph_composer.midi_reader import MidiReader


class Test(TestCase):
    def test_midi_to_tensor_type_0(self):
        """Test the conversion of a MIDI file with two tracks
        in format 0 to a MidiNoteSequences object.
        :return:
        """
        channel_one = [60, 62, 64, 65, 67, 69, 71, 72]
        channel_two = [61, 63, 65, 66, 68, 70, 72, 73]
        midi_seqs = MidiReader.midi_to_tensor("../neural_graph_composer/dataset/unit_test/test-2-tracks-type-0.mid")
        self.assertTrue(midi_seqs.num_instruments(), 2)
        self.assertEqual(len(midi_seqs[0].notes), len(channel_one))
        self.assertListEqual(midi_seqs[0].as_note_seq(), channel_one)
        self.assertListEqual(midi_seqs[1].as_note_seq(), channel_two)

    def test_midi_to_tensor_type_1(self):
        """
        :return:
        """
        channel_one = [60, 62, 64, 65, 67, 69, 71, 72]
        channel_two = [61, 63, 65, 66, 68, 70, 72, 73]
        midi_seqs = MidiReader.midi_to_tensor("../neural_graph_composer/dataset/unit_test/test-2-tracks-type-1.mid")
        self.assertTrue(midi_seqs.num_instruments(), 2)
        self.assertEqual(len(midi_seqs[0].notes), len(channel_one))
        self.assertListEqual(midi_seqs[0].as_note_seq(), channel_one)
        self.assertListEqual(midi_seqs[1].as_note_seq(), channel_two)

    def test_midi_to_tensor_type_2(self):
        """
        :return:
        """
        channel_one = [60, 62, 64, 65, 67, 69, 71, 72]
        channel_two = [61, 63, 65, 66, 68, 70, 72, 73]
        midi_seqs = MidiReader.midi_to_tensor("../neural_graph_composer/dataset/unit_test/test-2-tracks-type-2.mid")
        self.assertTrue(midi_seqs.num_instruments(), 2)
        self.assertEqual(len(midi_seqs[0].notes), len(channel_one))
        self.assertListEqual(midi_seqs[0].as_note_seq(), channel_one)
        self.assertListEqual(midi_seqs[1].as_note_seq(), channel_two)

    def test_is_quantized(self):
        """Un quantize should return False
        :return:
        """
        midi_seqs = MidiReader.midi_to_tensor("../neural_graph_composer/dataset/unit_test/test-all-gm-percussion.mid")
        self.assertFalse(midi_seqs[0].is_quantized() == False)

    def test_midi_as_note_seq(self):
        """Drums on midi channel 10 should be all drums
        :return:
        """
        midi_seqs = MidiReader.midi_to_tensor("../neural_graph_composer/dataset/unit_test/test-all-gm-percussion.mid")
        self.assertTrue(len(midi_seqs[1].as_note_seq()) == 0)

    def test_midi_to_tensor_percussion(self):
        """Drums on midi channel 10, should be all drums
        :return:
        """
        channel_one = [i for i in range(27, 88)] + [i for i in range(27, 88)] + [i for i in range(27, 88)]
        channel_one = sorted(channel_one)
        midi_seqs = MidiReader.midi_to_tensor("../neural_graph_composer/dataset/unit_test/test-all-gm-percussion.mid")
        self.assertIsNotNone(midi_seqs[0].instrument)
        self.assertTrue(midi_seqs[0].instrument.is_drum == True)
        self.assertListEqual(midi_seqs[0].as_note_seq(), channel_one)

    def test_midi_to_tensor_key_test(self):
        channel_one = [i for i in range(27, 88)]
        channel_two = [61, 63, 65, 66, 68, 70, 72, 73]
        midi_seqs = MidiReader.midi_to_tensor("../neural_graph_composer/dataset/unit_test/test-c-major-scale.mid")
        print(midi_seqs[0].notes)
        # print(midi_seqs[0].notes[0].is_drum)
