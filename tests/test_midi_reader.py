"""Test suite for midi reader
Author Mus
mbayramo@stanford.edu
spyroot@gmail.com
"""

import difflib
import os
from unittest import TestCase
from unittest.mock import Mock

from neural_graph_composer.midi.midi_instruments import MidiInstrumentInfo
from neural_graph_composer.midi.midi_note import MidiNote
from neural_graph_composer.midi.midi_sequence import MidiNoteSequence
from neural_graph_composer.midi.midi_sequences import MidiNoteSequences
from neural_graph_composer.midi_reader import MidiReader
import mido
import pretty_midi
import tempfile
import unittest

from tests.test_utils import mido_to_pretty_midi


class Test(TestCase):
    def test_midi_to_tensor_type_0(self):
        """Test the conversion of a MIDI file with two tracks
        in format 0 to a MidiNoteSequences object.
        :return:
        """
        channel_one = [60, 62, 64, 65, 67, 69, 71, 72]
        channel_two = [61, 63, 65, 66, 68, 70, 72, 73]
        midi_seqs = MidiReader.read("../neural_graph_composer/dataset/unit_test/test-2-tracks-type-0.mid")
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
        midi_seqs = MidiReader.read("../neural_graph_composer/dataset/unit_test/test-2-tracks-type-1.mid")
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
        midi_seqs = MidiReader.read("../neural_graph_composer/dataset/unit_test/test-2-tracks-type-2.mid")
        self.assertTrue(midi_seqs.num_instruments(), 2)
        self.assertEqual(len(midi_seqs[0].notes), len(channel_one))
        self.assertListEqual(midi_seqs[0].as_note_seq(), channel_one)
        self.assertListEqual(midi_seqs[1].as_note_seq(), channel_two)

    def test_is_quantized(self):
        """Un quantize should return False
        :return:
        """
        midi_seqs = MidiReader.read("../neural_graph_composer/dataset/unit_test/test-all-gm-percussion.mid")
        self.assertEqual(midi_seqs[0].is_quantized(), False)

    def test_midi_as_note_seq(self):
        """Drums on midi channel 10 should be all drums
        :return:
        """
        midi_seqs = MidiReader.read("../neural_graph_composer/dataset/unit_test/test-all-gm-percussion.mid")
        self.assertTrue(len(midi_seqs[1].as_note_seq()) == 0)

    def test_midi_to_tensor_percussion(self):
        """Drums on midi channel 10, should be all drums
        :return:
        """
        channel_one = [i for i in range(27, 88)] + [i for i in range(27, 88)] + [i for i in range(27, 88)]
        channel_one = sorted(channel_one)
        midi_seqs = MidiReader.read("../neural_graph_composer/dataset/unit_test/test-all-gm-percussion.mid")
        self.assertIsNotNone(midi_seqs[0].instrument)
        self.assertTrue(midi_seqs[0].instrument.is_drum == True)
        self.assertListEqual(midi_seqs[0].as_note_seq(), channel_one)

    def test_midi_to_tensor_key_test(self):
        channel_one = [i for i in range(27, 88)]
        channel_two = [61, 63, 65, 66, 68, 70, 72, 73]
        midi_seqs = MidiReader.read("../neural_graph_composer/dataset/unit_test/test-c-major-scale.mid")
        print(midi_seqs[0].notes)
        # print(midi_seqs[0].notes[0].is_drum)

    def assertBinaryFilesEqual(self, file1, file2, encoding='latin1'):
        # with open(file1, 'r', encoding=encoding) as f1, open(file2, 'r', encoding=encoding) as f2:
        with open(file1, 'rb') as f1, open(file2, 'rb') as f2:
            content1 = f1.read()
            content2 = f2.read()
        size1 = os.path.getsize(file1)
        size2 = os.path.getsize(file2)
        self.assertEqual(size1, size2, msg=f"File sizes differ: {size1} != {size2}")
        self.assertEqual(content1, content2, msg=self.get_binary_diff(content1, content2))

    def get_binary_diff(self, content1, content2):
        """
        :param content1:
        :param content2:
        :return:
        """
        d = difflib.Differ()
        cmd_result = d.compare(content1, content2)
        delta = list(d.compare(content1, content2))

        r = sum(1 for d in delta if d[0] in ['-', '+', '?'])
        print("Result", r)
        # print(dir(cmd_result))
        # print(f"diff len {len(diff)}")
        # return '\n'.join(diff)

    def test_write_to_file(self):
        """Basic write test
        :return:
        """
        midi_seqs = MidiReader.read("../neural_graph_composer/dataset/unit_test/test-c-major-scale.mid")
        MidiReader.write(midi_seqs, '../neural_graph_composer/dataset/unit_test/test-c-major-scale_back.mid')

    def test_conversion(self):
        """
        :return:
        """
        # Create a simple MIDI file using mido
        mido_file = mido.MidiFile()
        track = mido.MidiTrack()
        mido_file.tracks.append(track)
        track.append(mido.Message('program_change', program=12, time=0))
        track.append(mido.Message('note_on', note=64, velocity=64, time=32))
        track.append(mido.Message('note_off', note=64, velocity=64, time=64))

        # convert and assert
        pretty_midi_file = mido_to_pretty_midi(mido_file)
        self.assertEqual(len(pretty_midi_file.instruments), 1)
        self.assertEqual(pretty_midi_file.instruments[0].program, 12)
        self.assertEqual(len(pretty_midi_file.instruments[0].notes), 1)
        self.assertEqual(pretty_midi_file.instruments[0].notes[0].pitch, 64)

    def test_from_file(self):
        """Test the `from_file` method of the `MidiReader` class.
        This test creates a new MIDI file with a single note, loads it using the `from_file`
        method of a mock MIDI reader,
        and checks that the resulting `MidiNoteSequence` object has the expected properties.
        Finally, it cleans up the test file.
        :return:
        """
        midi_data = MidiNoteSequence()
        midi_data.notes.append(MidiNote(60, 0, 1, velocity=100))
        test_file = "test.mid"
        midi_data.to_midi_file(test_file)

        # test loading the MIDI file with the mock reader
        midi_seq = MidiReader.from_file(test_file)
        self.assertIsInstance(midi_seq, MidiNoteSequences)
        self.assertEqual(len(midi_seq[0].notes), 1)
        self.assertEqual(midi_seq[0].notes[0].pitch, 60)
        self.assertEqual(midi_seq[0].notes[0].start_time, 0)
        self.assertEqual(midi_seq[0].notes[0].end_time, 1)
        self.assertEqual(midi_seq[0].notes[0].velocity, 100)

        # clean up the test file
        os.remove(test_file)

    def test_from_file_different_instrument(self):
        """Test the `from_file` method of the `MidiReader` class using
        a MIDI file with a note played on a different instrument.

        This test creates a new MIDI file with a single note
        played on a different instrument, loads it using the `from_file`
        method of a mock MIDI reader, and checks that the
        resulting `MidiNoteSequence` object has the expected properties.
        :return:
        """
        midi_data = MidiNoteSequence(
            instrument=MidiInstrumentInfo(instrument=1, is_drum=False, name='Piano'))
        midi_data.notes.append(MidiNote(60, 0, 1, velocity=100, instrument=1))
        test_file = "test_instrument.mid"
        midi_data.to_midi_file(test_file)

        # test loading the MIDI file with the mock reader
        midi_seq = MidiReader.from_file(test_file)
        self.assertIsInstance(midi_seq, MidiNoteSequences)
        self.assertEqual(len(midi_seq[0].notes), 1)
        self.assertEqual(midi_seq[0].notes[0].pitch, 60)
        self.assertEqual(midi_seq[0].notes[0].start_time, 0)
        self.assertEqual(midi_seq[0].notes[0].end_time, 1)
        self.assertEqual(midi_seq[0].notes[0].velocity, 100)
        self.assertEqual(midi_seq[0].notes[0].instrument, 1)

        os.remove(test_file)

    def test_from_file_multiple_instruments(self):
        """Test the `from_file` method of the `MidiReader`
        class using a MIDI file with notes on different instruments.

        This test creates a new MIDI file with two notes played
        on different instruments, loads it using the `from_file`
        method of a mock MIDI reader, and checks that the resulting
        `MidiNoteSequence` object has the expected properties.

        :return:
        """
        midi_data1 = MidiNoteSequence(instrument=MidiInstrumentInfo(instrument=1, is_drum=False, name='Piano'))
        midi_data1.notes.append(MidiNote(60, 0, 1, velocity=100, instrument=1))

        midi_data2 = MidiNoteSequence(instrument=MidiInstrumentInfo(instrument=2, is_drum=False, name='Guitar'))
        midi_data2.notes.append(MidiNote(64, 0.5, 1.5, velocity=80, instrument=2))

        # combine the two MidiNoteSequences into one
        new_midi_seq = MidiNoteSequences(midi_seq=[midi_data1, midi_data2])
        self.assertEqual(len(new_midi_seq), 2)
        self.assertEqual(new_midi_seq[0].instrument.instrument_num, 1)
        self.assertEqual(new_midi_seq[1].instrument.instrument_num, 2)

        #
        # test_file = "test.mid"
        # MidiReader.write(new_midi_seq, test_file)
        #
        # # test loading the MIDI file with the mock reader
        # midi_seq = MidiReader.from_file(test_file)
        # self.assertIsInstance(midi_seq, MidiNoteSequences)
        # self.assertEqual(len(midi_seq[0].notes), 2)
        # self.assertEqual(midi_seq[0].notes[0].pitch, 60)
        # self.assertEqual(midi_seq[0].notes[0].start_time, 0)
        # self.assertEqual(midi_seq[0].notes[0].end_time, 1)
        # self.assertEqual(midi_seq[0].notes[0].velocity, 100)
        # self.assertEqual(midi_seq[0].notes[0].instrument, 1)
        # self.assertEqual(midi_seq[0].notes[1].pitch, 64)
        # self.assertEqual(midi_seq[0].notes[1].start_time, 0.5)
        # self.assertEqual(midi_seq[0].notes[1].end_time, 1.5)
        # self.assertEqual(midi_seq[0].notes[1].velocity, 80)
        # self.assertEqual(midi_seq[0].notes[1].instrument, 2)
        #
        # os.remove(test_file)
