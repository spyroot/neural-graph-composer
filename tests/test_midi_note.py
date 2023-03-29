"""
Test suite for midi note
Author Mus
mbayramo@stanford.edu
spyroot@gmail.com
"""
import math
from unittest import TestCase
import neural_graph_composer
from neural_graph_composer.midi.midi_note import MidiNote


class Test(TestCase):

    def test_quantize_to_nearest_step(self):
        sps_list = [2, 4, 8, 16, 32]
        expected = [1, 2, 4, 8, 16]
        amount = 0.5
        for i, sps in enumerate(sps_list):
            note = MidiNote(pitch=60, start_time=0.5, end_time=1.0, velocity=100)
            step_number = note.quantize_to_nearest_step(note.start_time, sps_list[i], amount)
            print(f"comparing {expected[i]} {step_number}")
            self.assertEqual(expected[i], step_number, f"expected {expected[i]} step number")

        sps_list = [2, 4, 8, 16, 32]
        expected = [1, 2, 4, 8, 16]
        amount = 0.25
        for i, sps in enumerate(sps_list):
            note = MidiNote(pitch=60, start_time=0.5, end_time=1.0, velocity=100)
            step_number = note.quantize_to_nearest_step(note.start_time, sps_list[i], amount)
            print(f"comparing {expected[i]} {step_number}")
            self.assertEqual(expected[i], step_number, f"expected {str(expected[i])} "
                                                       f"for amount {str(amount)} and sps {sps_list[i]}")

        sps_list = [2, 4, 8, 16, 32]
        expected = [1, 2, 4, 8, 16]
        amount = 0.75
        for i, sps in enumerate(sps_list):
            note = MidiNote(pitch=60, start_time=0.5, end_time=1.0, velocity=100)
            step_number = note.quantize_to_nearest_step(note.start_time, sps_list[i], amount)
            print(f"comparing {expected[i]} {step_number}")
            self.assertEqual(expected[i], step_number, f"expected {str(expected[i])} "
                                                       f"for amount {str(amount)} and sps {sps_list[i]}")

        # Test with edge cases
        #
        note = MidiNote(pitch=60, start_time=0.0, end_time=10.0, velocity=100)
        step_number = note.quantize_to_nearest_step(note.start_time, sps=4)
        self.assertEqual(0, step_number, "expected 0 step number for start time of zero-length note")

        with self.assertRaises(ValueError):
            note = MidiNote(pitch=60, start_time=0.0, end_time=0.5, velocity=100)
            step_number = note.quantize_to_nearest_step(note.start_time, sps=0)

        with self.assertRaises(ValueError):
            note = MidiNote(pitch=60, start_time=0.0, end_time=0.5, velocity=100)
            step_number = note.quantize_to_nearest_step(note.start_time, sps=-4)

    def test_compute_step_duration_in_seconds_quantized_start_0_end_2(self):
        """Test compute_step_duration_in_seconds with quantized_start_step = 0
        :return:
        """
        note = MidiNote(pitch=60, start_time=0.0, end_time=0.5, velocity=100,
                        quantized_start_step=0, quantized_end_step=2)

        sps_values = [2, 4, 8, 16, 32]
        expected_durations = [1.0, 0.5, 0.25, 0.125, 0.0625]

        for sps, expected_duration in zip(sps_values, expected_durations):
            duration = note.compute_step_duration_in_seconds(sps)
            self.assertAlmostEqual(expected_duration, duration, places=8,
                                   msg=f"expected {expected_duration} sec for {note} and {sps} sps")

    def test_compute_step_duration_in_seconds_quantized_start_0(self):
        """
        Test compute_step_duration_in_seconds for
        note with quantized_start_step=0 and quantized_end_step=1
        """
        note = MidiNote(pitch=60, start_time=0.0, end_time=0.5, velocity=255,
                        quantized_start_step=0, quantized_end_step=1)

        sps_values = [2, 4, 8, 16, 32]
        for sps in sps_values:
            expected_duration = 1.0 / sps
            duration = note.compute_step_duration_in_seconds(sps)
            self.assertAlmostEqual(expected_duration, duration, places=8,
                                   msg=f"expected {expected_duration} sec for {note} and {sps} sps")

    def test_compute_step_duration_in_seconds01(self):
        """Test compute_step_duration_in_seconds for a range of sps values
        and step gap 1
        """
        note = MidiNote(pitch=60, start_time=0.5, end_time=1.0, velocity=100,
                        quantized_start_step=3, quantized_end_step=4)
        sps_values = [2, 4, 8, 16, 32]
        expected_durations = [0.5, 0.25, 0.125, 0.0625, 0.03125]

        for sps, expected_duration in zip(sps_values, expected_durations):
            duration = note.compute_step_duration_in_seconds(sps)
            self.assertAlmostEqual(expected_duration, duration, places=8,
                                   msg=f"expected {expected_duration} sec for {note} and {sps} sps")

    def test_compute_step_duration_in_seconds02(self):
        """Test compute_step_duration_in_seconds for a range of sps values
        and gap 2
        :return:
        """
        note = MidiNote(pitch=60, start_time=0.5, end_time=1.0, velocity=100,
                        quantized_start_step=2, quantized_end_step=5)

        sps_values = [2, 4, 8, 16, 32]
        expected_durations = [1.5, 0.75, 0.375, 0.1875, 0.09375]

        for sps, expected_duration in zip(sps_values, expected_durations):
            duration = note.compute_step_duration_in_seconds(sps)
            self.assertAlmostEqual(expected_duration, duration, places=8,
                                   msg=f"expected {expected_duration} sec for {note} and {sps} sps")

    def test_step_duration_in_samples(self):
        """Test test_step_duration_in_samples
        :return:
        """
        note = MidiNote(pitch=60, start_time=0.5, end_time=1.0, velocity=100)
        note.quantize_in_place(sps=4, amount=0.5)
        self.assertEqual(0.5, note.start_time)
        self.assertEqual(1.25, note.end_time)
        self.assertEqual(4, note.step_duration_in_samples())

        note2 = MidiNote(pitch=60, start_time=0.5, end_time=1.0, velocity=100)
        self.assertEqual(0.5, note2.start_time)
        self.assertEqual(1.0, note2.end_time)
        note2.quantize_in_place(sps=16, amount=0.5)
        self.assertEqual(10, note2.step_duration_in_samples())

    def test_steps_to_seconds(self):
        """
        Test that the `steps_to_seconds` method in the `MidiNote` class returns the expected values.
        Tests are performed for various step sizes and sample rates. In addition, a separate test is performed
        for a step size of 5. The expected values for each combination of step size and sample rate are stored
        in the `expected` list. The `expected_step5` list contains the expected values for a step size of 5.

        :raise AssertionError: If the calculated value from `steps_to_seconds` method differs from the expected value.
        :return:
        """
        sps_list = [2, 4, 8, 16, 32]
        steps_list = [1, 2, 4, 8, 16, 32]
        expected = [[0.5, 1.0, 2.0, 4.0, 8.0, 16],
                    [0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
                    [0.125, 0.25, 0.5, 1.0, 2.0, 4.0],
                    [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0],
                    [0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0]]

        for i, sps in enumerate(sps_list):
            for j, steps in enumerate(steps_list):
                seconds = MidiNote.steps_to_seconds(steps, sps)
                self.assertAlmostEqual(expected[i][j], seconds, places=8,
                                       msg=f"expected {expected[i][j]} sec for {steps} step(s) and {sps} sps")

        expected_step5 = [2.5, 1.25, 0.625, 0.3125, 0.15625]
        # Test for step=5
        for i, sps in enumerate(sps_list):
            seconds = MidiNote.steps_to_seconds(5, sps)
            self.assertAlmostEqual(expected_step5[i], seconds, places=8,
                                   msg=f"expected {expected_step5[i]} sec for 5 step(s) and {sps} sps")

        expected_step9 = [4.5, 2.25, 1.125, 0.5625, 0.28125]
        for i, sps in enumerate(sps_list):
            seconds = MidiNote.steps_to_seconds(9, sps)
            self.assertAlmostEqual(expected_step9[i], seconds, places=8,
                                   msg=f"expected {expected_step9[i]} sec for 9 step(s) and {sps} sps")

    def test_quantize_in_place_adjusts_end_time(self):
        """
        Test that quantize_in_place adjusts the end time of a note to the quantized end step.
        :return:
        """
        # Test with sps=4
        note = MidiNote(pitch=60, start_time=0.5, end_time=1.0, velocity=100)
        note.quantize_in_place(sps=4, amount=0.5)
        expected_end_time = 4.0 / 4.0 + 0.5
        self.assertAlmostEqual(note.end_time, expected_end_time, places=6)

        # Test with sps=8
        note = MidiNote(pitch=60, start_time=0.5, end_time=1.0, velocity=100)
        note.quantize_in_place(sps=8, amount=0.5, min_step=None)
        expected_end_time = 8.0 / 8.0
        self.assertAlmostEqual(note.end_time, expected_end_time, places=6)

        # Test with sps=16
        note = MidiNote(pitch=60, start_time=0.5, end_time=1.0, velocity=100)
        note.quantize_in_place(sps=16, amount=0.5, min_step=None)
        expected_end_time = 16.0 / 16.0
        self.assertAlmostEqual(note.end_time, expected_end_time, places=6)

        # Test with sps=32
        note = MidiNote(pitch=60, start_time=0.5, end_time=1.0, velocity=100)
        note.quantize_in_place(sps=32, amount=0.5, min_step=None)
        expected_end_time = 32.0 / 32.0
        self.assertAlmostEqual(note.end_time, expected_end_time, places=6)

    def test_quantize_in_place(self):
        """
        :return:
        """
        note = MidiNote(pitch=60, start_time=0.5, end_time=1.0, velocity=100)
        note.quantize_in_place(sps=4, amount=0.5, min_step=None)
        self.assertEqual(2, note.quantized_start_step)
        self.assertEqual(5, note.quantized_end_step)
        self.assertTrue(note.is_quantized())

    def test_quantize_amount0(self):
        """Test with amount = 0"""
        note = MidiNote(pitch=60, start_time=0.5, end_time=1.0, velocity=100)
        with self.assertRaises(ValueError):
            note.quantize_in_place(sps=4, amount=0, min_step=None)

    def test_beat_duration(self):
        """

        :return:
        """
        note1 = MidiNote(start_time=0.0, end_time=2.0, pitch=60, velocity=64)
        self.assertAlmostEqual(note1.beat_duration(), 0.5, places=9)

        note2 = MidiNote(start_time=0.0, end_time=1.0, pitch=60, velocity=64, numerator=3, denominator=4)
        self.assertEqual(3, note2.numerator)
        self.assertAlmostEqual(4, note2.denominator)
        self.assertAlmostEqual(note2.beat_duration(tempo=80), 0.5625, places=9)

        note3 = MidiNote(start_time=0.0, end_time=1.0, pitch=60, velocity=64, numerator=1, denominator=2)
        self.assertEqual(1, note3.numerator)
        self.assertAlmostEqual(2, note3.denominator)
        self.assertAlmostEqual(note3.beat_duration(tempo=120), 0.25, places=9)

        note4 = MidiNote(start_time=1.0, end_time=5.0, pitch=60, velocity=64, numerator=6, denominator=8)
        self.assertEqual(6, note4.numerator)
        self.assertAlmostEqual(8, note4.denominator)
        self.assertAlmostEqual(note4.beat_duration(tempo=100), 0.45, places=9)

    def test_bpm(self):
        """Test the calculation of beats per minute (BPM) for a MidiNote object.
        This test function creates several MidiNote objects with different start times,
        end times, and numbers of beats per measure, and verifies
        that the BPM calculation is correct for each note.
        :return:
        """
        note = MidiNote(start_time=0.0, end_time=4.0, pitch=60, velocity=64)
        self.assertTrue(math.isclose(note.bpm(), 120.0, rel_tol=1e-9), f"bpm {note.bpm()} expected {60.0}")
        note = MidiNote(start_time=0.0, end_time=2.0, pitch=60, velocity=64)
        self.assertTrue(math.isclose(note.bpm(), 120.0, rel_tol=1e-9), f"bpm {note.bpm()} expected {120.0}")
        note = MidiNote(start_time=0.0, end_time=1.0, pitch=60, velocity=64)
        self.assertTrue(math.isclose(note.bpm(), 120.0, rel_tol=1e-9), f"bpm {note.bpm()} expected {240}")
        note = MidiNote(start_time=0.0, end_time=1.0, pitch=60, velocity=64, numerator=3, denominator=4)
        self.assertTrue(math.isclose(note.bpm(), 160.0, rel_tol=1e-9), f"bpm {note.bpm()} expected {240}")
        note = MidiNote(start_time=0.0, end_time=1.0, pitch=60, velocity=64, numerator=5, denominator=4)
        self.assertTrue(math.isclose(note.bpm(), 96, rel_tol=1e-9), f"bpm {note.bpm()} expected {375.0}")
        #
        # 0.0 (start time of note)
        # 0.4375 (start time of note + duration of one beat)
        # 0.875 (start time of note + 2 * duration of one beat)
        # 1.3125 (start time of note + 3 * duration of one beat)
        # Now we can calculate the time between each of these beats, in seconds:
        #
        # 0.4375 seconds (time of beat 2 - time of beat 1)
        # 0.4375 seconds (time of beat 3 - time of beat 2)
        # 0.4375 seconds (time of beat 4 - time of beat 3)
        # The average time between beats is (0.4375 + 0.4375 + 0.4375) / 3 = 0.4375 seconds.
        #
        # BPM = 60 / 0.4375 = 137.14
        note = MidiNote(start_time=0.0, end_time=1.75, pitch=60, velocity=64)
        self.assertTrue(math.isclose(note.bpm(), 120, rel_tol=1e-4))

    def test_note_to_freq(self):
        """
        freq = 440 * 2^((44 - 69) / 12)
             = 440 * 2^(-25/12)
            = 103.826 Hz (approx.)
        :return:
        """
        # Test A4, which should be 440 Hz
        self.assertTrue(math.isclose(MidiNote.note_to_freq(69), 440, rel_tol=1e-9))

        # Test C4, which should be 261.63 Hz
        self.assertTrue(math.isclose(MidiNote.note_to_freq(60), 261.63, rel_tol=1e-3))

        # Test E5, which should be 659.25 Hz
        self.assertTrue(math.isclose(MidiNote.note_to_freq(76), 659.25, rel_tol=1e-5))

        # Test G#3, which should be 103.826 Hz
        self.assertTrue(math.isclose(MidiNote.note_to_freq(44), 103.82, rel_tol=1e-2))

    #
    # def test_quantize_in_place(self):
    #     # Create a note with start time at 0.31 seconds and end time at 0.6 seconds
    #     note = MidiNote(pitch=60, start_time=0.31, end_time=0.6)
    #     sps = 4  # Steps per second
    #
    #     # Check the original duration of the note
    #     self.assertTrue(math.isclose(note.duration(), 0.29, rel_tol=1e-9))
    #
    #     # Quantize the note with a quantization amount of 0.5
    #     note.quantize_in_place(sps, amount=0.5)
    #
    #     # The start time and end time should be quantized to the nearest step#
    #     self.assertTrue(note.quantized_start_step == 1,
    #                     msg="Start time of the note is not quantized to the nearest step after quantization")
    #     self.assertTrue(note.quantized_end_step == 2,
    #                     msg="End time of the note is not quantized to the nearest step after quantization")
    #
    #     # The duration of the note should not change after quantization
    #     print(f"note.duration {note.duration()}")
    #     self.assertAlmostEqual(note.start_time, 0.25, places=6)
    #
    #     self.assertTrue(math.isclose(note.duration(), 0.29, rel_tol=1e-9))
    #
    #     # Create a note that is already quantized
    #     quantized_note = MidiNote(pitch=60, start_time=0.5, end_time=0.75,
    #                               quantized_start_step=2, quantized_end_step=3)
    #
    #     # Quantizing the already quantized note should not change its quantized start and end steps
    #     quantized_note.quantize_in_place(sps, amount=0.5)
    #     self.assertTrue(quantized_note.quantized_start_step == 2)
    #     self.assertTrue(quantized_note.quantized_end_step == 3)
    #
    #     # The duration of the note should not change after quantization
    #     self.assertTrue(quantized_note.duration() == 0.25)

    # def test_quantize_in_place_1_16(self):
    #     # Create a note with start and end times that are not quantized to 1/16th notes
    #     note = MidiNote(pitch=60, start_time=0.5, end_time=1.2)
    #     # Quantize the note to 1/16th notes
    #     note.quantize_in_place(sps=16, amount=0.5)
    #     # Check that the note was quantized correctly
    #     assert note.quantized_start_step == 8
    #     assert note.quantized_end_step == 19
    #     assert note.start_time == 0.5
    #     assert note.end_time == 1.1875

    def test_quantize_basic_amount_05(self):
        """
        :return:
        """
        note = MidiNote(pitch=60, velocity=100, start_time=1, end_time=1.5)
        note.quantize_in_place(1)
        self.assertEqual(1, note.quantized_start_step)
        self.assertEqual(2, note.quantized_end_step)
        self.assertEqual(1.0, note.start_time)
        self.assertEqual(2.0, note.end_time)

        note = MidiNote(pitch=60, velocity=100, start_time=1, end_time=1.5)
        note.quantize_in_place(2)
        self.assertEqual(2, note.quantized_start_step)
        self.assertEqual(4, note.quantized_end_step)
        self.assertEqual(1.0, note.start_time)
        self.assertEqual(2.0, note.end_time)

        note = MidiNote(pitch=60, velocity=100, start_time=1, end_time=1.5)
        note.quantize_in_place(4)
        self.assertEqual(4, note.quantized_start_step)
        self.assertEqual(7, note.quantized_end_step)
        self.assertEqual(1.0, note.start_time)
        self.assertEqual(1.75, note.end_time)

        note = MidiNote(pitch=60, velocity=100, start_time=1, end_time=1.5)
        note.quantize_in_place(8)
        self.assertEqual(8, note.quantized_start_step)
        self.assertEqual(13, note.quantized_end_step)
        self.assertEqual(1.0, note.start_time)
        self.assertAlmostEquals(1.62, note.end_time, delta=0.01)

        note = MidiNote(pitch=60, velocity=100, start_time=1, end_time=1.5)
        note.quantize_in_place(16)
        self.assertEqual(note.quantized_start_step, 16)
        self.assertEqual(note.quantized_end_step, 25)
        self.assertEqual(1.0, note.start_time)
        self.assertAlmostEquals(1.56, note.end_time, delta=0.01)

        note = MidiNote(pitch=60, velocity=100, start_time=1, end_time=1.5)
        note.quantize_in_place(32)
        self.assertEqual(32, note.quantized_start_step)
        self.assertEqual(49, note.quantized_end_step)
        self.assertEqual(1.0, note.start_time)
        self.assertAlmostEquals(1.53, note.end_time, delta=0.01)

        note = MidiNote(pitch=60, velocity=100, start_time=1, end_time=1.5)
        note.quantize_in_place(64)
        self.assertEqual(64, note.quantized_start_step)
        self.assertEqual(97, note.quantized_end_step)
        self.assertEqual(1.0, note.start_time)
        self.assertAlmostEquals(1.52, note.end_time, delta=0.01)

    def test_quantize_basic_amount_10(self):
        """Same test amount 1.0
        :return:
        """
        note = MidiNote(pitch=60, velocity=100, start_time=1, end_time=1.5)
        note.quantize_in_place(2, amount=1.0)
        self.assertEqual(2, note.quantized_start_step)
        self.assertEqual(3, note.quantized_end_step)
        self.assertEqual(1.0, note.start_time)
        self.assertEqual(1.5, note.end_time)

        note = MidiNote(pitch=60, velocity=100, start_time=1, end_time=1.5)
        note.quantize_in_place(4, amount=1.0)
        self.assertEqual(4, note.quantized_start_step)
        self.assertEqual(6, note.quantized_end_step)
        self.assertEqual(1.0, note.start_time)
        self.assertEqual(1.5, note.end_time)

        note = MidiNote(pitch=60, velocity=100, start_time=1, end_time=1.5)
        note.quantize_in_place(8, amount=1.0)
        self.assertEqual(8, note.quantized_start_step)
        self.assertEqual(12, note.quantized_end_step)
        self.assertEqual(1.0, note.start_time)
        self.assertAlmostEquals(1.5, note.end_time, delta=0.01)

        note = MidiNote(pitch=60, velocity=100, start_time=1, end_time=1.5)
        note.quantize_in_place(16, amount=1.0)
        self.assertEqual(16, note.quantized_start_step)
        self.assertEqual(24, note.quantized_end_step)
        self.assertEqual(1.0, note.start_time)
        self.assertAlmostEquals(1.5, note.end_time, delta=0.01)

        note = MidiNote(pitch=60, velocity=100, start_time=1, end_time=1.5)
        note.quantize_in_place(32, amount=1.0)
        self.assertEqual(32, note.quantized_start_step)
        self.assertEqual(48, note.quantized_end_step)
        self.assertEqual(1.0, note.start_time)
        self.assertAlmostEquals(1.5, note.end_time, delta=0.01)

        note = MidiNote(pitch=60, velocity=100, start_time=1, end_time=1.5)
        note.quantize_in_place(64, amount=1.0)
        self.assertEqual(64, note.quantized_start_step)
        self.assertEqual(96, note.quantized_end_step)
        self.assertEqual(1.0, note.start_time)
        self.assertAlmostEquals(1.50, note.end_time, delta=0.01)

    def test_quantize_shifted(self):
        """ Test shifted note with amount 0.5
        :return:
        """
        # Start step = note start time * step per second = 0.2 * 2 = 0.4
        # Stop step = note stop time * step per second = 1.4 * 2 = 2.8
        note = MidiNote(pitch=60, velocity=100, start_time=0.2, end_time=1.4)
        note.quantize_in_place(2)
        self.assertAlmostEquals(0, note.start_time, delta=0.01)
        self.assertEqual(2.0, note.end_time)
        self.assertEqual(0, note.quantized_start_step)
        self.assertEqual(4, note.quantized_end_step)

        note = MidiNote(pitch=60, velocity=100, start_time=0.2, end_time=1.4)
        note.quantize_in_place(4)
        self.assertAlmostEquals(0.25, note.start_time, delta=0.01)
        self.assertEqual(1.75, note.end_time)
        self.assertEqual(1, note.quantized_start_step)
        self.assertEqual(7, note.quantized_end_step)

        note = MidiNote(pitch=60, velocity=100, start_time=0.2, end_time=1.4)
        note.quantize_in_place(8)
        self.assertAlmostEquals(0.25, note.start_time, delta=0.01)
        self.assertAlmostEquals(1.5, note.end_time, delta=0.1)
        self.assertEqual(2, note.quantized_start_step)
        self.assertEqual(12, note.quantized_end_step)

        #
        note = MidiNote(pitch=60, velocity=100, start_time=0.2, end_time=1.4)
        note.quantize_in_place(16)
        self.assertAlmostEquals(0.25, note.start_time, delta=0.1)
        self.assertAlmostEquals(1.5, note.end_time, delta=0.1)
        self.assertEqual(3, note.quantized_start_step)
        self.assertEqual(23, note.quantized_end_step)

        note = MidiNote(pitch=60, velocity=100, start_time=0.2, end_time=1.4)
        note.quantize_in_place(32)
        self.assertAlmostEquals(0.25, note.start_time, delta=0.1)
        self.assertAlmostEquals(1.5, note.end_time, delta=0.1)
        self.assertEqual(6, note.quantized_start_step)
        self.assertEqual(46, note.quantized_end_step)

    def test_quantize_shifted_amount10(self):
        """ Test shifted note with amount 0.5
        :return:
        """
        # Start step = note start time * step per second = 0.2 * 2 = 0.4
        # Stop step = note stop time * step per second = 1.4 * 2 = 2.8
        note = MidiNote(pitch=60, velocity=100, start_time=0.2, end_time=1.4)
        note.quantize_in_place(2, amount=1.0)
        self.assertAlmostEquals(0, note.start_time, delta=0.01)
        self.assertEqual(1.5, note.end_time)
        self.assertEqual(0, note.quantized_start_step)
        self.assertEqual(3, note.quantized_end_step)
        #
        # note = MidiNote(pitch=60, velocity=100, start_time=0.2, end_time=1.4)
        # note.quantize_in_place(4, amount=1.0)
        # self.assertAlmostEquals(0, note.start_time, delta=0.01)
        # self.assertEqual(1.5, note.end_time)
        # self.assertEqual(1, note.quantized_start_step)
        # self.assertEqual(6, note.quantized_end_step)
        #
        #
        # note = MidiNote(pitch=60, velocity=100, start_time=0.2, end_time=1.4)
        # note.quantize_in_place(8, amount=1.0)
        # self.assertAlmostEquals(0.25, note.start_time, delta=0.01)
        # self.assertAlmostEquals(1.5, note.end_time, delta=0.1)
        # self.assertEqual(2, note.quantized_start_step)
        # self.assertEqual(12, note.quantized_end_step)
        # #
        # # # #
        note = MidiNote(pitch=60, velocity=100, start_time=0.2, end_time=1.4)
        note.quantize_in_place(16, amount=1.0)
        self.assertAlmostEquals(0.1875, note.start_time, delta=0.1)
        self.assertAlmostEquals(1.375, note.end_time, delta=0.1)
        self.assertEqual(3, note.quantized_start_step)
        self.assertEqual(23, note.quantized_end_step)
        #
        note = MidiNote(pitch=60, velocity=100, start_time=0.2, end_time=1.4)
        note.quantize_in_place(32, amount=1.0)
        self.assertAlmostEquals(0.1875, note.start_time, delta=0.1)
        self.assertAlmostEquals(1.375, note.end_time, delta=0.1)
        self.assertEqual(6, note.quantized_start_step)
        self.assertEqual(45, note.quantized_end_step)

    def test_note_comparison01(self):
        # create some notes for testing
        note1 = MidiNote(pitch=60, start_time=0.0, end_time=1.0)
        note2 = MidiNote(pitch=60, start_time=0.0, end_time=2.0)
        note3 = MidiNote(pitch=64, start_time=1.0, end_time=2.0)
        note4 = MidiNote(pitch=60, start_time=1.0, end_time=2.0)

        # test equality
        self.assertEqual(note1, note1)
        self.assertEqual(note1, MidiNote(pitch=60, start_time=0.0, end_time=1.0))
        self.assertNotEqual(note1, note2)
        self.assertNotEqual(note1, note3)

        # test less than
        self.assertLess(note1, note2)
        self.assertLess(note1, note3)
        self.assertLess(note1, note4)
        self.assertLess(note2, note3)

        self.assertFalse(note3 < note4)
        self.assertFalse(note4 < note3)

    def test_note_comparison02(self):
        note1 = MidiNote(pitch=60, start_time=1.0, end_time=2.0)
        note2 = MidiNote(pitch=64, start_time=1.0, end_time=2.0)
        note3 = MidiNote(pitch=64, start_time=1.0, end_time=2.0)
        note4 = MidiNote(pitch=64, start_time=1.0, end_time=3.0)
        note5 = MidiNote(pitch=64, start_time=2.0, end_time=3.0)

        self.assertTrue(note1 <= note2)
        self.assertTrue(note2 <= note3)
        self.assertTrue(note2 <= note1)
        #
        self.assertTrue(note3 <= note2)
        self.assertTrue(note3 <= note4)
        self.assertFalse(note4 <= note3)
        self.assertTrue(note4 <= note5)
        self.assertFalse(note5 <= note4)
        self.assertFalse(note4 > note5)
        self.assertFalse(note4 >= note5)

    def test_non_overlapping_notes(self):
        note1 = MidiNote(pitch=60, start_time=0, end_time=1)
        note2 = MidiNote(pitch=62, start_time=1.5, end_time=2.5)
        self.assertFalse(note1.overlaps(note2))

    def test_notes_with_same_start_and_end_time(self):
        note1 = MidiNote(pitch=60, start_time=0, end_time=1)
        note3 = MidiNote(pitch=64, start_time=1.0001, end_time=1.0002)
        self.assertFalse(note1.overlaps(note3))

    def test_notes_with_same_start_time_but_different_end_time(self):
        note1 = MidiNote(pitch=60, start_time=0, end_time=1)
        note4 = MidiNote(pitch=67, start_time=0.5, end_time=1.5)
        self.assertTrue(note1.overlaps(note4))

    def test_notes_with_different_start_time_but_same_end_time(self):
        note1 = MidiNote(pitch=60, start_time=0, end_time=1)
        note5 = MidiNote(pitch=69, start_time=0.5, end_time=1)
        self.assertTrue(note1.overlaps(note5))

    def test_fully_overlapping_notes(self):
        note1 = MidiNote(pitch=60, start_time=0, end_time=1)
        note6 = MidiNote(pitch=72, start_time=0.5, end_time=1.5)
        self.assertTrue(note1.overlaps(note6))

    def test_partially_overlapping_notes(self):
        note1 = MidiNote(pitch=60, start_time=0, end_time=1)
        note7 = MidiNote(pitch=74, start_time=0.5, end_time=1.5)
        self.assertTrue(note1.overlaps(note7))

    def test_non_overlapping_notes2(self):
        note1 = MidiNote(pitch=60, start_time=0, end_time=1)
        note2 = MidiNote(pitch=62, start_time=1.5, end_time=2.5)
        self.assertFalse(note1.overlaps(note2))

    def test_notes_with_same_pitch_but_no_overlap(self):
        note1 = MidiNote(pitch=60, start_time=0, end_time=1)
        note8 = MidiNote(pitch=60, start_time=1.5, end_time=2.5)
        self.assertFalse(note1.overlaps(note8))

    def test_notes_with_same_start_and_end_time_but_different_pitch(self):
        note1 = MidiNote(pitch=60, start_time=0, end_time=1)
        note9 = MidiNote(pitch=61, start_time=1, end_time=1.001)
        self.assertFalse(note1.overlaps(note9))

    def test_notes_with_same_pitch_and_same_start_time_but_different_end_time(self):
        note1 = MidiNote(pitch=60, start_time=0.1, end_time=1.001)
        note10 = MidiNote(pitch=60, start_time=0.1, end_time=2.001)
        self.assertTrue(note1.overlaps(note10))

    def test_overlap_precision(self):
        # Test notes with a small gap between them
        note1 = MidiNote(pitch=60, start_time=0, end_time=1)
        note2 = MidiNote(pitch=62, start_time=1.0001, end_time=2.0001)
        self.assertFalse(note1.overlaps(note2))

        note3 = MidiNote(pitch=60, start_time=0, end_time=1)
        note4 = MidiNote(pitch=62, start_time=1.00001, end_time=2.00001)
        self.assertFalse(note3.overlaps(note4))

        note5 = MidiNote(pitch=60, start_time=0, end_time=0.0001)
        note6 = MidiNote(pitch=62, start_time=0.00005, end_time=0.00015)
        self.assertTrue(note5.overlaps(note6))

        note7 = MidiNote(pitch=60, start_time=0, end_time=1)
        note8 = MidiNote(pitch=62, start_time=0.0000001, end_time=0.5)
        self.assertTrue(note7.overlaps(note8))

    def test_note_lt(self):
        note0 = MidiNote(pitch=60, start_time=0.01, end_time=1.001)
        note1 = MidiNote(pitch=60, start_time=0, end_time=1.001)
        note2 = MidiNote(pitch=64, start_time=0, end_time=2)
        note3 = MidiNote(pitch=67, start_time=1, end_time=3)
        note4 = MidiNote(pitch=72, start_time=0.5, end_time=1.5)
        note5 = MidiNote(pitch=60, start_time=0, end_time=0.5)
        note6 = MidiNote(pitch=60, start_time=0.001, end_time=1.001)

        self.assertTrue(note1 < note2)
        self.assertTrue(note2 < note3)
        self.assertTrue(note1 < note3)
        self.assertTrue(note1 < note4)
        self.assertFalse(note2 < note1)
        self.assertFalse(note3 < note2)
        self.assertFalse(note3 < note1)
        self.assertFalse(note4 < note1)
        self.assertTrue(note5 < note1)
        self.assertTrue(note6 < note0)

    def test_note_le(self):
        note1 = MidiNote(pitch=60, start_time=0, end_time=1)
        note2 = MidiNote(pitch=64, start_time=0, end_time=2)
        note3 = MidiNote(pitch=67, start_time=1, end_time=3)
        note4 = MidiNote(pitch=72, start_time=0.5, end_time=1.5)

        self.assertTrue(note1 <= note2)
        self.assertTrue(note2 <= note3)
        self.assertTrue(note1 <= note3)
        self.assertTrue(note1 <= note4)
        self.assertFalse(note2 <= note1)
        self.assertFalse(note3 <= note2)
        self.assertFalse(note3 <= note1)
        self.assertFalse(note4 <= note1)
        self.assertTrue(note1 <= note1)

        note5 = MidiNote(pitch=60, start_time=0, end_time=1 + 1e-7)
        note6 = MidiNote(pitch=64, start_time=0, end_time=2 - 1e-7)
        note7 = MidiNote(pitch=67, start_time=1 - 1e-7, end_time=3 + 1e-7)
        note8 = MidiNote(pitch=72, start_time=0.5 + 1e-7, end_time=1.5 - 1e-7)

        self.assertTrue(note1 <= note5)
        self.assertTrue(note2 <= note6)
        self.assertTrue(note3 <= note7)
        self.assertTrue(note1 <= note8)
        self.assertFalse(note2 <= note1)


