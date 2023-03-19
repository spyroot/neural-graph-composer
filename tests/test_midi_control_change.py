"""Midi instrument unit test.
Author Mus spyroot@gmail.com
"""
import unittest
from unittest import TestCase
from neural_graph_composer.midi.midi_control_change import MidiControlChange
import json


class TestMidiControlChange(unittest.TestCase):

    def test_cc_value(self):
        cc = MidiControlChange(cc_number=1, cc_value=100)
        self.assertEqual(cc.cc_value, 100)

        cc = MidiControlChange(cc_number=1, cc_value=300)
        self.assertEqual(cc.cc_value, 255)

        cc = MidiControlChange(cc_number=1, cc_value=-10)
        self.assertEqual(cc.cc_value, 0)

    def test_quantize(self):
        cc = MidiControlChange(cc_number=1, cc_value=100, cc_time=0.5)
        cc.quantize(sps=4, amount=0.5)
        self.assertEqual(cc.quantized_start_step, 2)
        self.assertEqual(cc.quantized_end_step, 2)

    def test_lt(self):
        midi_cc1 = MidiControlChange(7, 100, 0.5, 1, 2, True, 4)
        midi_cc2 = MidiControlChange(7, 100, 0.5, 1, 2, True, 4)
        midi_cc1.seq = 0
        midi_cc2.seq = 1
        self.assertEqual(midi_cc1 < midi_cc2, True)
        midi_cc1.seq = 1
        midi_cc2.seq = 0
        self.assertEqual(midi_cc1 < midi_cc2, False)

    def test_eq(self):
        midi_cc1 = MidiControlChange(7, 100, 0.5, 1, 2, True, 4)
        midi_cc2 = MidiControlChange(7, 100, 0.5, 1, 2, True, 4)
        midi_cc3 = MidiControlChange(6, 100, 0.5, 1, 2, True, 4)
        self.assertEqual(midi_cc1 == midi_cc2, True)
        self.assertEqual(midi_cc1 == midi_cc3, False)

    def test_quantized(self):
        midi_cc = MidiControlChange(7, 100, 0.5, 1, 2, True, 4)
        midi_cc.quantize(4, 0.5)
        self.assertEqual(midi_cc.quantized_start_step, 1)
        self.assertEqual(midi_cc.quantized_end_step, 1)

    def test_is_quantized(self):
        midi_cc = MidiControlChange(7, 100, 0.5, 1, 2, True, 4, 1, 3)
        self.assertEqual(midi_cc.is_quantized(), True)
        midi_cc.quantized_start_step = -1
        self.assertEqual(midi_cc.is_quantized(), False)
        midi_cc.quantized_start_step = 1
        midi_cc.quantized_end_step = -1
        self.assertEqual(midi_cc.is_quantized(), False)

    def test_midi_cc_name_to_cc_zero(self):
        """Check blank cc
        :return:
        """
        cc_values = MidiControlChange.MIDI_CC_NAME_TO_CC
        assert cc_values[0].cc_number == 0
        assert cc_values[0].description == "Bank"

    def test_midi_cc_name_to_cc_number(self):
        """Check all key correspond to correct cc.
        :return:
        """
        cc_values = MidiControlChange.MIDI_CC_NAME_TO_CC
        for i, cc in cc_values.items():
            self.assertEqual(i, cc.cc_number)



