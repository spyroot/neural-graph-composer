"""Midi instrument unit test.
Author Mus spyroot@gmail.com
"""
from unittest import TestCase
from neural_graph_composer.midi.midi_instruments import MidiInstrumentInfo
import json


class TestMidiInstrumentInfo(TestCase):
    def test_instrument_creation(self):
        # Test if the MidiInstrumentInfo class can be initialized correctly.
        midi_info = MidiInstrumentInfo(1, "Guitar", False)
        self.assertEqual(midi_info.instrument_num, 1)
        self.assertEqual(midi_info.name, "Guitar")
        self.assertFalse(midi_info.is_drum)

    def test_is_drum(self):
        # Test default is_drum value
        instrument_info = MidiInstrumentInfo(instrument=0)
        self.assertFalse(instrument_info.is_drum)

        # Test setting is_drum to True
        instrument_info = MidiInstrumentInfo(instrument=0, is_drum=True)
        self.assertTrue(instrument_info.is_drum)

        # Test setting is_drum to False
        instrument_info = MidiInstrumentInfo(instrument=0, is_drum=False)
        self.assertFalse(instrument_info.is_drum)

    def test_instrument_number_boundaries(self):
        # Test valid instrument number
        instrument_info = MidiInstrumentInfo(instrument=0)
        self.assertEqual(instrument_info.instrument_num, 0)

        # Test lower boundary
        instrument_info = MidiInstrumentInfo(instrument=0)
        self.assertEqual(instrument_info.instrument_num, 0)

        # Test upper boundary
        instrument_info = MidiInstrumentInfo(instrument=255)
        self.assertEqual(instrument_info.instrument_num, 255)

        # Test out-of-bounds instrument number
        with self.assertRaises(ValueError):
            instrument_info = MidiInstrumentInfo(instrument=-1)

    def test_instrument_name(self):
        # Test default instrument name
        instrument_info = MidiInstrumentInfo(instrument=0)
        self.assertEqual(instrument_info.name, "Unknown")

        # Test non-empty instrument name
        instrument_info = MidiInstrumentInfo(instrument=0, name="Acoustic Grand Piano")
        self.assertEqual(instrument_info.name, "Acoustic Grand Piano")

    def test_eq(self):
        # Test equality of two identical instrument infos
        instrument_info1 = MidiInstrumentInfo(instrument=0, name="Acoustic Grand Piano", is_drum=False)
        instrument_info2 = MidiInstrumentInfo(instrument=0, name="Acoustic Grand Piano", is_drum=False)
        self.assertEqual(instrument_info1, instrument_info2)

        # Test equality of two different instrument infos
        instrument_info1 = MidiInstrumentInfo(instrument=0, name="Acoustic Grand Piano", is_drum=False)
        instrument_info2 = MidiInstrumentInfo(instrument=1, name="Bright Acoustic Piano", is_drum=False)
        self.assertNotEqual(instrument_info1, instrument_info2)

        # Test equality of an instrument info with a non-instrument info object
        instrument_info1 = MidiInstrumentInfo(instrument=0, name="Acoustic Grand Piano", is_drum=False)
        self.assertNotEqual(instrument_info1, 1)

        # Test equality of an instrument info with None
        instrument_info1 = MidiInstrumentInfo(instrument=0, name="Acoustic Grand Piano", is_drum=False)
        self.assertNotEqual(instrument_info1, None)

    def test_hash(self):
        # Test hash of two identical instrument infos
        instrument_info1 = MidiInstrumentInfo(instrument=0, name="Acoustic Grand Piano", is_drum=False)
        instrument_info2 = MidiInstrumentInfo(instrument=0, name="Acoustic Grand Piano", is_drum=False)
        self.assertEqual(hash(instrument_info1), hash(instrument_info2))

        # Test hash of two different instrument infos
        instrument_info1 = MidiInstrumentInfo(instrument=0, name="Acoustic Grand Piano", is_drum=False)
        instrument_info2 = MidiInstrumentInfo(instrument=1, name="Bright Acoustic Piano", is_drum=False)
        self.assertNotEqual(hash(instrument_info1), hash(instrument_info2))

    def test_lt(self):
        instrument1 = MidiInstrumentInfo(20, "Trumpet")
        instrument2 = MidiInstrumentInfo(40, "Piano")
        instrument3 = MidiInstrumentInfo(60, "Organ")
        self.assertLess(instrument1, instrument2)
        self.assertLess(instrument2, instrument3)
        self.assertLess(instrument1, instrument3)

    def test_ne(self):
        instrument1 = MidiInstrumentInfo(20, "Trumpet")
        instrument2 = MidiInstrumentInfo(20, "Trumpet")
        instrument3 = MidiInstrumentInfo(40, "Piano")
        self.assertFalse(instrument1 != instrument2)
        self.assertTrue(instrument1 != instrument3)

    def test_to_json(self):
        instrument_info = MidiInstrumentInfo(1, "Piano", False)
        json_str = json.dumps(instrument_info.to_dict())
        expected_json_str = '{"instrument_num": 1, "name": "Piano", "is_drum": false}'
        self.assertEqual(json_str, expected_json_str)

    def test_from_json(self):
        json_str = '{"instrument_num": 1, "name": "Piano", "is_drum": false}'
        expected_instrument_info = MidiInstrumentInfo(1, "Piano", False)
        instrument_info = MidiInstrumentInfo.from_dict(json.loads(json_str))
        self.assertEqual(instrument_info, expected_instrument_info)
