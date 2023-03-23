import unittest
import tempfile
import os

from neural_graph_composer.midi.midi_sequence import MidiNoteSequence
from neural_graph_composer.midi_graph_builder import MidiGraphBuilder


class TestMidiGraphBuilder(unittest.TestCase):

    # def setUp(self):
    #     self.midi_file = tempfile.NamedTemporaryFile(suffix=".mid")
    #     self.midi_seq = MidiNoteSequence.from_time_unit(100, [])
    #     self.midi_seq.instruments = []
    #     self.midi_seq.ticks_per_quarter_note = 480
    #
    # def tearDown(self):
    #     self.midi_file.close()
    #
    # def test_from_file(self):
    #     builder = MidiGraphBuilder.from_file(self.midi_file.name)
    #     self.assertIsInstance(builder, MidiGraphBuilder)
    #
    # def test_from_midi_sequence(self):
    #     builder = MidiGraphBuilder.from_midi_sequence(self.midi_seq)
    #     self.assertIsInstance(builder, MidiGraphBuilder)

    class TestMidiReader(unittest.TestCase):
        def test_from_file(self):
            # Create a mock MidiBaseReader
            mock_reader.read.return_value = MidiNoteSequence()

            # Create a test MIDI file
            midi_data = MidiNoteSequence()
            midi_data.notes.append(MidiNote(60, 0, 1, 100))
            test_file = "test.mid"
            midi_data.to_midi_file(test_file)

            # Test loading the MIDI file with the mock reader
            midi_seq = MidiReader.from_file(test_file, mock_reader)
            self.assertIsInstance(midi_seq, MidiNoteSequence)

            # Clean up the test file
            os.remove(test_file)
