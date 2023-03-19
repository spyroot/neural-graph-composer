import decimal
import math
import unittest
import random

import numpy as np
from decimal import Decimal

from neural_graph_composer.midi.midi_utils import frames_from_times, quantize_to_nearest_step, \
    quantize_to_nearest_step_alt, quantize_note_to_step


class TestFramesFromTimes(unittest.TestCase):

    def test_positive_times(self):
        """
        Test the function with valid input parameters, and verify that the output is as expected.
        """
        start_time = 1.0
        end_time = 2.0
        fps = 30
        min_frame_occupancy = 0.0
        expected_start_frame = 30
        expected_end_frame = 60
        actual_start_frame, actual_end_frame = frames_from_times(start_time, end_time, fps, min_frame_occupancy)
        self.assertEqual(actual_start_frame, expected_start_frame)
        self.assertEqual(actual_end_frame, expected_end_frame)

    def test_negative_times(self):
        """
        Test the function with negative start_time, and verify that the function raises a ValueError exception.
        :return:
        """
        start_time = -1.0
        end_time = 2.0
        fps = 30
        min_frame_occupancy = 0.0
        with self.assertRaises(ValueError):
            frames_from_times(start_time, end_time, fps, min_frame_occupancy)

    def test_end_before_start(self):
        """
        Test the function with end_time less than start_time,
         and verify that the function raises a ValueError exception.
        :return:
        """
        start_time = 2.0
        end_time = 1.0
        fps = 30
        min_frame_occupancy = 0.0
        with self.assertRaises(ValueError):
            frames_from_times(start_time, end_time, fps, min_frame_occupancy)

    def test_min_frame_occupancy(self):
        """
         Test the function with a non-zero min_frame_occupancy, and verify that the output is as expected.
        :return:
        """
        start_time = 1.0
        end_time = 2.0
        fps = 30
        min_frame_occupancy = 0.5
        expected_start_frame = 31
        expected_end_frame = 59
        actual_start_frame, actual_end_frame = frames_from_times(start_time, end_time, fps, min_frame_occupancy)
        self.assertEqual(actual_start_frame, expected_start_frame)
        self.assertEqual(actual_end_frame, expected_end_frame)

    def test_end_time_rounding(self):
        """
        Test the function with end_time that rounds to the next frame, and verify that the output is as expected.
        """
        start_time = 1.0
        end_time = 1.5
        fps = 30
        min_frame_occupancy = 0.0
        expected_start_frame = 30
        expected_end_frame = 45
        actual_start_frame, actual_end_frame = frames_from_times(start_time, end_time, fps, min_frame_occupancy)
        self.assertEqual(actual_start_frame, expected_start_frame)
        self.assertEqual(actual_end_frame, expected_end_frame)

    def test_min_frame_occupancy_rounding(self):
        """Tests the function with min_frame_occupancy that rounds up to
        the next frame, and verify that the output is as expected.
        i.e. This test case verifies that when the min_frame_occupancy
        is set to a value that rounds up to the next frame,
        the function returns the expected start_frame and end_frame values.
        """
        start_time = 1.0
        end_time = 3.0
        fps = 30
        min_frame_occupancy = 0.7
        expected_start_frame = 31
        expected_end_frame = 89
        actual_start_frame, actual_end_frame = frames_from_times(start_time, end_time, fps, min_frame_occupancy)
        self.assertEqual(actual_start_frame, expected_start_frame)
        self.assertEqual(actual_end_frame, expected_end_frame)

    def test_quantization(self):
        expected_outputs = {
            1: 4,
            2: 2,
            4: 1,
            8: 1,
            16: 1,
            32: 1
        }

        final_results = {}

        for step_size, expected_output in expected_outputs.items():
            ticks_per_beat = 480  # default value for MIDI
            ticks_per_step = math.floor(ticks_per_beat * 4 / step_size)
            ticks_per_second = ticks_per_beat * 120 / 60

            results = []

            for i in range(10):
                unquantized_time = i * step_size
                quantized_time = quantize_to_nearest_step(unquantized_time, step_size)

                unquantized_ticks = int(unquantized_time * ticks_per_second)
                quantized_ticks = int(round(unquantized_ticks / ticks_per_step)) * ticks_per_step

                results.append((unquantized_time, quantized_time, unquantized_ticks, quantized_ticks))

            final_results[step_size] = results

        num_passed = 0
        num_failed = 0
        failed_assertions = []

        for step_size, expected_output in expected_outputs.items():
            for unquantized_time, quantized_time, unquantized_ticks, quantized_ticks in final_results[step_size]:
                try:
                    self.assertAlmostEqual(quantized_time, unquantized_time, delta=1e-6,
                                           msg=f"Step size: {step_size}, Unquantized time: {unquantized_time}")
                    self.assertAlmostEqual(quantized_ticks / ticks_per_second, quantized_time, delta=1e-6,
                                           msg=f"Step size: {step_size}, Unquantized time: {unquantized_time}")
                except AssertionError as e:
                    num_failed += 1
                    failed_assertions.append(f"{e} -- Step size: {step_size}, Unquantized time: {unquantized_time}")
                else:
                    num_passed += 1

            actual_output = len(set(quantized_time for _, quantized_time, _, _ in final_results[step_size]))
            if actual_output != expected_output:
                num_failed += 1
                # failed_assertions.append(f"Step size: {step_size}")

        print(f"Number of passed assertions: {num_passed}")
        print(f"Number of failed assertions: {num_failed}\n")

        if failed_assertions:
            print("Failed assertions:")
            for assertion in failed_assertions:
                print(assertion)
        else:
            print("All assertions passed.")

    def test_quantization123123(self):
        expected_outputs = {
            1: 4,
            2: 2,
            4: 1,
            8: 1,
            16: 1,
            32: 1
        }

        final_results = {}

        for step_size, expected_output in expected_outputs.items():
            ticks_per_beat = 480  # default value for MIDI
            if step_size == 1:
                ticks_per_step = math.ceil(ticks_per_beat * 4 / step_size)
            else:
                ticks_per_step = math.floor(ticks_per_beat * 4 / step_size)
            ticks_per_second = ticks_per_beat * 120 / 60

            results = []

            for i in range(100):
                unquantized_time = i * step_size
                quantized_time = quantize_to_nearest_step(unquantized_time, step_size)
                unquantized_ticks = int(unquantized_time * ticks_per_second)
                quantized_ticks = int(round(unquantized_ticks / ticks_per_step)) * ticks_per_step
                results.append((unquantized_time, quantized_time, unquantized_ticks, quantized_ticks))

            final_results[step_size] = results

        num_passed = 0
        num_failed = 0
        failed_assertions = []

        for step_size, expected_output in expected_outputs.items():
            for unquantized_time, quantized_time, unquantized_ticks, quantized_ticks in final_results[step_size]:
                try:
                    self.assertAlmostEqual(quantized_time, unquantized_time, delta=1e-6,
                                           msg=f"Step size: {step_size}, Unquantized time: {unquantized_time}")
                    self.assertAlmostEqual(quantized_ticks / ticks_per_second, quantized_time, delta=1e-6,
                                           msg=f"Step size: {step_size}, Unquantized time: {unquantized_time}")
                except AssertionError as e:
                    num_failed += 1
                    failed_assertions.append(f"{e} -- Step size: {step_size}, Unquantized time: {unquantized_time}")
                else:
                    num_passed += 1

            actual_output = len(set(quantized_time for _, quantized_time, _, _ in final_results[step_size]))
            if actual_output != expected_output:
                num_failed += 1
                failed_assertions.append(
                    f"Step size: {step_size}, Expected: {expected_output}, Actual: {actual_output}")

        print(f"Number of passed assertions: {num_passed}")
        print(f"Number of failed assertions: {num_failed}\n")

        if failed_assertions:
            print("Failed assertions:")
            for assertion in failed_assertions:
                print(assertion)
        else:
            print("All assertions passed.")

    def test_quantize_to_nearest_step1(self):
        num_tests = 100
        num_passed = 0
        num_failed = 0
        for i in range(num_tests):
            step_size = random.choice([1, 2, 4, 8, 16, 32])
            un_quantized_time = random.uniform(0, 100)
            quantized_time = quantize_to_nearest_step(un_quantized_time, step_size)
            expected_quantized_time = round(un_quantized_time / step_size) * step_size
            expected_quantized_time = round(expected_quantized_time, 6)
            quantized_time = round(quantized_time, 6)
            if expected_quantized_time == quantized_time:
                num_passed += 1
            else:
                print(f"Step size: {step_size}, Expected: {expected_quantized_time}, Actual: {quantized_time}")
                num_failed += 1
        print(f"Number of passed assertions: {num_passed}")
        print(f"Number of failed assertions: {num_failed}")

    def test_quantization3(self):
        expected_outputs = {
            1: 4,
            2: 2,
            4: 1,
            8: 1,
            16: 1,
            32: 1
        }

        final_results = {}

        for step_size, expected_output in expected_outputs.items():
            ticks_per_beat = 480  # default value for MIDI
            ticks_per_step = math.floor(ticks_per_beat * 4 / step_size)
            ticks_per_second = ticks_per_beat * 120 / 60

            results = []

            for i in range(10):
                unquantized_time = i * step_size
                quantized_time = quantize_to_nearest_step(unquantized_time, step_size)

                unquantized_ticks = int(unquantized_time * ticks_per_second)
                quantized_ticks = int(round(unquantized_ticks / ticks_per_step)) * ticks_per_step

                results.append((unquantized_time, quantized_time, unquantized_ticks, quantized_ticks))

            final_results[step_size] = results

        num_passed = 0
        num_failed = 0
        failed_assertions = []

        for step_size, expected_output in expected_outputs.items():
            for unquantized_time, quantized_time, unquantized_ticks, quantized_ticks in final_results[step_size]:
                try:
                    self.assertAlmostEqual(quantized_time, unquantized_time, delta=1e-6,
                                           msg=f"Step size: {step_size}, Unquantized time: {unquantized_time}")
                    self.assertAlmostEqual(quantized_ticks / ticks_per_second, quantized_time, delta=1e-6,
                                           msg=f"Step size: {step_size}, Unquantized time: {unquantized_time}")
                except AssertionError as e:
                    num_failed += 1
                    failed_assertions.append(f"{e} -- Step size: {step_size}, Unquantized time: {unquantized_time}")
                    print(
                        f"FAILED: {e} -- Step size: {step_size},"
                        f" Unquantized time: {unquantized_time}, "
                        f"Expected: {unquantized_time}, Actual: {quantized_time}")
                else:
                    num_passed += 1

            actual_output = len(set(quantized_time for _, quantized_time, _, _ in final_results[step_size]))
            if actual_output != expected_output:
                num_failed += 1
                failed_assertions.append(f"Step size: {step_size}")
                print(f"FAILED: Step size: {step_size}, Expected: {expected_output}, Actual: {actual_output}")

        print(f"Number of passed assertions: {num_passed}")
        print(f"Number of failed assertions: {num_failed}\n")

        if failed_assertions:
            print("Failed assertions:")
            for assertion in failed_assertions:
                print(assertion)
        else:
            print("All assertions passed.")

    def test_scalar_input(self):
        expected_outputs = {
            0.0: 4,
            0.1: 4,
            0.2: 4,
            0.3: 3,
            0.4: 3,
            0.5: 3,
            0.6: 2,
            0.7: 2,
            0.8: 2,
            0.9: 1,
            1.0: 1,
        }
        unquantized_times = [0.1, 0.33, 0.6, 0.75, 0.9]
        threshold_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        pass_list = [False] * len(threshold_list)
        original_pass_count = 0
        alt_pass_count = 0

        for unquantized_time in unquantized_times:
            for i, threshold in enumerate(threshold_list):
                steps_per_second = 0.3
                expected_output = expected_outputs[threshold]

                # test original function
                actual_output = quantize_to_nearest_step(unquantized_time, steps_per_second, threshold)
                error = abs(actual_output - expected_output)

                if error < 1e-6:
                    pass_list[i] = True
                    original_pass_count += 1

                print(f"Unquantized Time: {unquantized_time}, Threshold: {threshold}, "
                      f"Expected Output: {expected_output}, Actual Output: {actual_output}, Pass: {pass_list[i]}")

        print(f"Original pass count: {original_pass_count}/{len(threshold_list) * len(unquantized_times)}")

    def test_negative_input(self):
        """Test quantize_to_nearest_step the function with negative input values.
        """
        unquantized_time = -1.7
        steps_per_second = 2
        rounding_threshold = 0.5
        expected_output = -3.0

        actual_output = quantize_to_nearest_step(unquantized_time, steps_per_second, rounding_threshold)
        np.testing.assert_allclose(actual_output, expected_output)

    def test_array_input(self):
        """Test quantize_to_nearest_step the function with array input values.
        """
        unquantized_time = np.array([1.7, 2.2, 3.5])
        steps_per_second = 2
        rounding_threshold = 0.5
        expected_output = np.array([3.0, 4.0, 7.0])

        actual_output = quantize_to_nearest_step(unquantized_time, steps_per_second, rounding_threshold)
        np.testing.assert_allclose(actual_output, expected_output)

    def test_rounding_threshold(self):
        """Test quantize_to_nearest_step the function with different rounding thresholds.
        """
        unquantized_time = 1.7
        steps_per_second = 2
        rounding_threshold = 0.3
        expected_output = 3.0

        actual_output = quantize_to_nearest_step(unquantized_time, steps_per_second, rounding_threshold)
        np.testing.assert_allclose(actual_output, expected_output)

        rounding_threshold = 0.8
        expected_output = 2.0

        actual_output = quantize_to_nearest_step(unquantized_time, steps_per_second, rounding_threshold)
        np.testing.assert_allclose(actual_output, expected_output)

    def test_very_small_rounding_threshold(self):
        """Test 'quantize_to_nearest_step' the function with a very small rounding threshold.
        """
        unquantized_time = 1.7
        steps_per_second = 2
        rounding_threshold = 1e-10
        expected_output = 3.0

        actual_output = quantize_to_nearest_step(unquantized_time, steps_per_second, rounding_threshold)
        np.testing.assert_allclose(actual_output, expected_output)

    def test_large_input_values(self):
        """
        Test the function with very large input values.
        """
        unquantized_time = 1e20
        steps_per_second = 1e10
        rounding_threshold = 0.5
        expected_output = 1e30

        actual_output = quantize_to_nearest_step(unquantized_time, steps_per_second, rounding_threshold)
        np.testing.assert_allclose(actual_output, expected_output)

    def test_zero_steps_per_second(self):
        """
        Test the function with steps_per_second set to zero.
        """
        unquantized_time = 1.7
        steps_per_second = 0
        rounding_threshold = 0.5

        with self.assertRaises(ValueError):
            quantize_to_nearest_step(unquantized_time, steps_per_second, rounding_threshold)

    def test_non_numeric_input(self):
        """
        Test the function with non-numeric input values.
        """
        unquantized_time = "1.7"
        steps_per_second = 2
        rounding_threshold = 0.5

        with self.assertRaises(TypeError):
            quantize_to_nearest_step(unquantized_time, steps_per_second, rounding_threshold)

        unquantized_time = 1.7
        steps_per_second = "2"
        rounding_threshold = 0.5

        with self.assertRaises(TypeError):
            quantize_to_nearest_step(unquantized_time, steps_per_second, rounding_threshold)

        unquantized_time = 1.7
        steps_per_second = 2
        rounding_threshold = "0.5"

        with self.assertRaises(TypeError):
            quantize_to_nearest_step(unquantized_time, steps_per_second, rounding_threshold)

    def test_invalid_rounding_threshold(self):
        """
        Test the function with invalid rounding thresholds.
        """
        unquantized_time = 1.7
        steps_per_second = 2
        rounding_threshold = -0.1

        with self.assertRaises(ValueError):
            quantize_to_nearest_step(unquantized_time, steps_per_second, rounding_threshold)

        rounding_threshold = 1.1

        with self.assertRaises(ValueError):
            quantize_to_nearest_step(unquantized_time, steps_per_second, rounding_threshold)

    def test_non_scalar_input(self):
        """Test the function with non-scalar input values.
        """
        unquantized_time = [1.7, 2.2, 3.5]
        steps_per_second = 2
        rounding_threshold = 0.5

        with self.assertRaises(TypeError):
            quantize_to_nearest_step(unquantized_time, steps_per_second, rounding_threshold)

    def test_quantize_to_nearest_step(self):
        # Test with different step sizes and tempos
        tests = [
            # step_size, tempo, time_signature, input, expected_output
            (1, 120, (4, 4), 0, 0),
            (1, 120, (4, 4), 1, 1),
            (1, 120, (4, 4), 2, 2),
            (1, 120, (4, 4), 1.5, 2),
            (1, 120, (4, 4), 0.5, 0),
            (0.5, 120, (4, 4), 1.2, 1),
            (0.5, 120, (4, 4), 1.8, 2),
            (0.25, 120, (4, 4), 3.2, 3),
            (0.25, 120, (4, 4), 3.8, 4),
            (0.125, 240, (3, 4), 5.5, 5.5),
            (0.125, 240, (3, 4), 5.6, 5.5),
            (0.125, 240, (3, 4), 5.7, 6),
        ]
        for step_size, tempo, time_signature, input_val, expected_output in tests:
            output = quantize_note_to_step(input_val, step_size, tempo, time_signature)
            assert math.isclose(output, expected_output,
                                rel_tol=1e-6), f"Test failed for input: {input_val}, expected: {expected_output}, but got: {output}"
        print("All tests passed!")

    def test_quantize_note_to_step(self):
        note_start, note_end = 0.6, 1.6
        step_size = 0.25
        grid_size = 1.0
        output = quantize_note_to_step(note_start, note_end, step_size, grid_size)
        expected_output = (2.75, 3.0)
        assert output == expected_output, f"Expected {expected_output}, but got {output}"





