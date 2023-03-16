import decimal
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Pitch:
    Pitch = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    Accent = {"#": 1, "": 0, "b": -1, "!": -1, "♯": 1, "♭": -1}
    Sharp = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"]
    Flat = ["C", "D♭", "D", "E♭", "E", "F", "G♭", "G", "A♭", "A", "B♭", "B"]

    Sharp_Offset = [
        (5, "E♯"),
        (0, "B♯"),
        (7, "F𝄪"),
        (2, "C𝄪"),
        (9, "G𝄪"),
        (4, "D𝄪"),
        (11, "A𝄪"),
    ]

    Flats_Offset = [
        (11, "C♭"),
        (4, "F♭"),
        (9, "B𝄫"),
        (2, "E𝄫"),
        (7, "A𝄫"),
        (0, "D𝄫"),
    ]


def steps_per_second_from_bpm(bpm: Optional[float] = 120, resolution: Optional[int] = 480):
    """Calculates steps per second given bpm and resolution.

    The tempo of a MIDI sequence in BPM (beats per minute)
    We can calculate the number of steps per second by dividing the tempo by 60.

    Steps per second = (120 BPM * 480 standard resolution) / 60 = 960

    :param bpm:  default value for midi is 120
    :param resolution:  default resolution 480
    :return:
    """
    return bpm * resolution / 60.0


def from_bpm_spq_to_sps(steps_per_quarter, bpm, four_quarter: Optional[int] = 4):
    """Calculate step per second from step per quarter.

    :param steps_per_quarter:
    :param bpm:
    :param four_quarter:
    :return:
    """
    qpm = bpm / four_quarter
    return steps_per_quarter * qpm / 60.0


def from_spq_to_sps(steps_per_quarter, qpm):
    """Calculates steps per second (sps ) from steps per quarter.

    The tempo of a MIDI sequence in BPM (beats per minute)
    We can calculate the number of steps per second by dividing the tempo by 60.

    Steps per second = (120 BPM * 480 standard resolution) / 60 = 960

    :param steps_per_quarter:
    :param qpm: quarter notes per minute
    :return:
    """
    return steps_per_quarter * qpm / 60.0


def bpm_to_qpm(bpm: Optional[float] = 120.0, beats_per_measure: Optional[int] = 4) -> float:
    """
    Numeric stable version, Convert beats per minute (BPM) to quarter notes per minute (QPM).

    :param bpm: The tempo in beats per minute (default 120 BPM).
    :param beats_per_measure: The number of beats per measure (default 4).
    :return: The tempo in quarter notes per minute.
    """
    qpm = (bpm * beats_per_measure) / 60.0
    return np.round(qpm, decimals=10)


def bpm_to_qpm(bpm: Optional[float] = 120.0, beats_per_measure: Optional[int] = 4) -> float:
    """
    Convert beats per minute (BPM) to quarter notes per minute (QPM).

    :param bpm: The tempo in beats per minute (default 120 BPM).
    :param beats_per_measure: The number of beats per measure (default 4).
    :return: The tempo in quarter notes per minute.
    """
    qpm = (bpm / 60.0) * beats_per_measure
    return qpm


def qpm_to_bpm(qpm: Optional[int] = 120, quarter_note: Optional[int] = 4):
    """ QPM (Quarter Notes per Minute) is a measure of tempo in music,
        BPM (Beats per Minute) , convert QPM to BPM.
    :param qpm:
    :param quarter_note:
    :return:
    """
    return qpm * quarter_note


import numpy as np


def frames_from_times(start_time, end_time, fps, min_frame_occupancy=0.0):
    """
    Convert start and end times to frame indices.

    :param start_time: The start time of the event, in seconds.
    :param end_time: The end time of the event, in seconds.
    :param fps: The frames per second.
    :param min_frame_occupancy: The minimum occupancy threshold for each frame, as a fraction of the total frame duration.
    :return: A tuple of the start and end frame indices.
    """
    if start_time < 0.0 or end_time < 0.0:
        raise ValueError("Start and end times must be non-negative.")

    start_frame = np.floor(start_time * fps).astype(int)
    end_frame = np.ceil(end_time * fps).astype(int)

    if end_frame <= start_frame:
        raise ValueError("End time must be greater than start time.")

    if min_frame_occupancy > 0.0:
        start_occupancy = (start_frame + 1.0 - start_time * fps) / fps
        start_frame += np.ceil(min_frame_occupancy - start_occupancy).astype(int)

        end_occupancy = (end_time * fps - end_frame) / fps
        end_frame -= np.ceil(min_frame_occupancy - end_occupancy).astype(int)
        end_frame = max(start_frame, end_frame)

    return start_frame, end_frame


#
# def quantize_to_nearest_step(unquantized_time: float,
#                              steps_per_second: float,
#                              rounding_threshold: float = 0.5) -> int:
#     """
#     Returns the nearest step based on the provided steps per second.
#
#     Args:
#         unquantized_time: The unquantized time in seconds.
#         steps_per_second: The number of steps per second to use for quantization.
#         rounding_threshold: The rounding threshold to use. If the fractional part of the result is greater than or equal
#                             to the rounding threshold, the result is rounded up to the nearest integer. Otherwise, the
#                             result is rounded down.
#
#     Returns:
#         The nearest step based on the provided steps per second.
#     """
#     if not isinstance(unquantized_time, (int, float)):
#         raise TypeError("unquantized_time must be a numeric type.")
#     if not isinstance(steps_per_second, (int, float)):
#         raise TypeError("steps_per_second must be a numeric type.")
#     if not isinstance(rounding_threshold, (int, float)):
#         raise TypeError("rounding_threshold must be a numeric type.")
#
#     if steps_per_second <= 0:
#         raise ValueError("steps_per_second must be greater than zero.")
#     if rounding_threshold < 0 or rounding_threshold >= 1:
#         raise ValueError("rounding_threshold must be in the range [0, 1).")
#
#     unquantized_steps = unquantized_time * steps_per_second
#     fractional_part = unquantized_steps % 1
#
#     if fractional_part >= rounding_threshold:
#         return int(unquantized_steps + 1)
#     else:
#         return int(unquantized_steps)


def quantize_to_nearest_step_float(unquantized_time: float,
                                   steps_per_second: float,
                                   rounding_threshold: float = 0.5) -> int:
    """
    Returns the nearest step based on the provided steps per second.

    Args:
        unquantized_time: The unquantized time in seconds.
        steps_per_second: The number of steps per second to use for quantization.
        rounding_threshold: The rounding threshold to use. If the fractional part of the result is greater than or equal
                            to the rounding threshold, the result is rounded up to the nearest integer. Otherwise, the
                            result is rounded down.

    Returns:
        The nearest step based on the provided steps per second.
    """
    if not isinstance(unquantized_time, (int, float)):
        raise TypeError("unquantized_time must be a numeric type.")
    if not isinstance(steps_per_second, (int, float)):
        raise TypeError("steps_per_second must be a numeric type.")
    if not isinstance(rounding_threshold, (int, float)):
        raise TypeError("rounding_threshold must be a numeric type.")
    if steps_per_second <= 0:
        raise ValueError("steps_per_second must be greater than zero.")
    if rounding_threshold < 0 or rounding_threshold >= 1:
        raise ValueError("rounding_threshold must be in the range [0, 1).")

    # Multiply unquantized_time and steps_per_second
    # by a large power of 10 to convert them to integers.
    # This ensures that the resulting value is an integer,
    # even if there are rounding errors.
    unquantized_steps = int(unquantized_time * steps_per_second * 10 ** 10)
    threshold = int(rounding_threshold * 10 ** 10)

    if unquantized_steps < 0:
        # Handle negative values separately, since % behaves differently for negative numbers in Python.
        remainder = abs(unquantized_steps) % steps_per_second
        if remainder >= threshold:
            return int(unquantized_steps / steps_per_second) - 1
        else:
            return int(unquantized_steps / steps_per_second)
    else:
        remainder = unquantized_steps % steps_per_second
        if remainder >= threshold:
            return int(unquantized_steps / steps_per_second) + 1
        else:
            return int(unquantized_steps / steps_per_second)


#
# #
# def quantize_to_nearest_step(un_quantized_time, steps_per_second, rounding_threshold=0.5):
#     """
#     Quantizes the input time to the nearest step based on the specified steps per second.
#     """
#     un_quantized_steps = un_quantized_time * steps_per_second
#     quantized_steps = np.round(un_quantized_steps - rounding_threshold)
#     return quantized_steps.astype(int)


# def quantize_to_nearest_step(unquantized_time: np.ndarray,
#                              steps_per_second: float,
#                              rounding_threshold: float = 0.0) -> np.ndarray:
#     """This vectorized version.  Returns the nearest
#     :param unquantized_time: A NumPy array of unquantized times in seconds.
#     :param steps_per_second: The number of steps per second to use for quantization.
#     :param rounding_threshold: The rounding threshold to use.
#                             If the fractional part of the result is greater than or equal
#                             to the rounding threshold, the result
#                             is rounded up to the nearest integer. Otherwise, the result is rounded down.
#     :return: ndarray of the nearest steps based on the provided steps per second.
#     """
#     if not isinstance(steps_per_second, (int, float)):
#         raise TypeError("steps_per_second must be a numeric type.")
#     if not isinstance(rounding_threshold, (int, float)):
#         raise TypeError("rounding_threshold must be a numeric type.")
#
#     if steps_per_second <= 0:
#         raise ValueError("steps_per_second must be greater than zero.")
#     if rounding_threshold < 0 or rounding_threshold >= 1:
#         raise ValueError("rounding_threshold must be in the range [0, 1).")
#
#     # Multiply unquantized_time and steps_per_second by a large power of 10 to convert them to integers.
#     # This ensures that the resulting value is an integer, even if there are rounding errors.
#     unquantized_steps = np.round(unquantized_time * steps_per_second * 10 ** 10)
#     threshold = int(rounding_threshold * 10 ** 10)
#
#     # Handle negative values separately, since % behaves differently for negative numbers in Python.
#     mask = unquantized_steps < 0
#     remainder = np.abs(unquantized_steps) % steps_per_second
#
#     result = np.where(remainder >= threshold, unquantized_steps // steps_per_second - 1,
#                       unquantized_steps // steps_per_second)
#
#     # Convert the result back to a float and apply the mask to handle negative values.
#     result = result.astype(float)
#     result[mask] = -result[mask] / steps_per_second
#
#     return result
#
# def quantize_to_nearest_step(un_quantized_time, steps_per_second):
#     """
#     Quantizes the input time to the nearest step based on the specified steps per second.
#     """
#     un_quantized_steps = un_quantized_time * steps_per_second
#     decimal_component = un_quantized_steps - int(un_quantized_steps)
#     rounding_threshold = 0.5 - decimal_component
#     quantized_steps = int(un_quantized_steps + rounding_threshold)
#     return quantized_steps
# def quantize_to_nearest_step(un_quantized_time, step_size, tempo=120, time_signature=(4, 4)):
#     """
#     Quantizes the input time to the nearest step based on the specified step size (in beats).
#     """
#     ticks_per_beat = 480  # default value for MIDI
#     if step_size == 1:
#         ticks_per_step = math.floor(ticks_per_beat * time_signature[0] / step_size)
#         print(f"Step size {step_size} ticks_per_step {ticks_per_step}")
#     else:
#         ticks_per_step = math.ceil(ticks_per_beat * time_signature[0] / step_size)
#     ticks_per_second = ticks_per_beat * tempo / 60
#     un_quantized_ticks = int(un_quantized_time * ticks_per_second)
#     quantized_ticks = int(round(un_quantized_ticks / ticks_per_step)) * ticks_per_step
#     quantized_time = quantized_ticks / ticks_per_second
#     if step_size == 1:
#         quantized_ticks = round(quantized_ticks / ticks_per_beat) * ticks_per_beat
#         assert quantized_ticks % ticks_per_beat == 0, f"Ticks per step: {ticks_per_step}"
#     return quantized_time
def quantize_to_nearest_step(un_quantized_time, step_size, tempo=120, time_signature=(4, 4)):
    """
    Quantizes the input time to the nearest step based on the specified step size (in beats).
    """
    ticks_per_beat = 480  # default value for MIDI
    if step_size == 1:
        ticks_per_step = math.floor(ticks_per_beat * time_signature[0] / step_size)
        # print(f"Step size {step_size} ticks_per_step {ticks_per_step}")
    else:
        ticks_per_step = math.ceil(ticks_per_beat * time_signature[0] / step_size)
    ticks_per_second = ticks_per_beat * tempo / 60
    un_quantized_ticks = un_quantized_time * ticks_per_second
    quantized_ticks = round(un_quantized_ticks / ticks_per_step) * ticks_per_step
    quantized_time = quantized_ticks / ticks_per_second
    if step_size == 1:
        quantized_ticks = round(quantized_ticks / ticks_per_beat) * ticks_per_beat
        assert quantized_ticks % ticks_per_beat == 0, f"Ticks per step: {ticks_per_step}"
    return quantized_time


# def quantize_to_nearest_step(time, step_size):
#     return round(time / step_size) * step_size
#
# def quantize_to_nearest_step(un_quantized_time, step_size, tempo=120, time_signature=(4, 4)):
#     """
#     Quantizes the input time to the nearest step based on the specified step size (in beats).
#     """
#     ticks_per_beat = 480  # default value for MIDI
#     if step_size == 1:
#         ticks_per_step = math.ceil(ticks_per_beat * time_signature[0] / step_size)
#     else:
#         ticks_per_step = math.floor(ticks_per_beat * time_signature[0] / step_size)
#     ticks_per_second = ticks_per_beat * tempo / 60
#     un_quantized_ticks = int(un_quantized_time * ticks_per_second)
#     quantized_ticks = int(round(un_quantized_ticks / ticks_per_step)) * ticks_per_step
#     quantized_time = quantized_ticks / ticks_per_second
#     return quantized_time

def quantize_to_nearest_step_alt(unquantized_time, steps_per_second, rounding_threshold):
    decimal_unquantized_time = decimal.Decimal(str(unquantized_time))
    decimal_steps_per_second = decimal.Decimal(str(steps_per_second))
    decimal_rounding_threshold = decimal.Decimal(str(rounding_threshold))
    decimal_quantized_steps = decimal_unquantized_time * decimal_steps_per_second - decimal_rounding_threshold
    quantized_steps = int(decimal_quantized_steps.to_integral_value(decimal.ROUND_HALF_UP))
    return quantized_steps


import numpy as np
from scipy.spatial import distance


def vector_quantize(notes, grid, metric=distance.euclidean):
    # calculate distances between each note and grid point
    distances = np.zeros((len(notes), len(grid)))
    for i, note in enumerate(notes):
        for j, point in enumerate(grid):
            distances[i, j] = metric(note, point)

    # assign each note to nearest grid point
    quantized_notes = []
    for i in range(len(notes)):
        j = np.argmin(distances[i])
        quantized_notes.append(grid[j])

    return quantized_notes

# Define the quantization grid
num_steps = 12
step_size = 1/8
grid = np.zeros((num_steps,))

# Define the codebook
min_pitch = 60
max_pitch = 71
codebook = np.linspace(min_pitch, max_pitch, num_steps, endpoint=True)

# Map MIDI pitches to the quantized grid
def map_pitch_to_quantized_grid(pitch, codebook):
    dist = np.abs(pitch - codebook)
    quantized_pitch = np.argmin(dist)
    return quantized_pitch

# Example usage
midi_pitch = 67  # G4
quantized_pitch = map_pitch_to_quantized_grid(midi_pitch, codebook)
quantized_time = quantized_pitch * step_size
print(f"MIDI pitch: {midi_pitch}, Quantized pitch: {codebook[quantized_pitch]}, Quantized time: {quantized_time}")

def quantize_to_grid(notes, grid_size):
    """
    Quantizes the notes to the given grid size using linear interpolation.
    """
    start_time = min(notes, key=lambda note: note.start_time).start_time
    end_time = max(notes, key=lambda note: note.end_time).end_time
    num_steps = int(math.ceil((end_time - start_time) / grid_size))

    # create an empty grid
    grid = [[] for i in range(num_steps)]

    for note in notes:
        # calculate the start and end steps for the note
        start_step = int(math.floor((note.start_time - start_time) / grid_size))
        end_step = int(math.ceil((note.end_time - start_time) / grid_size))

        # add the note to the grid, performing linear interpolation if necessary
        if start_step == end_step:
            grid[start_step].append(note)
        else:
            for step in range(start_step, end_step):
                t0 = start_time + step * grid_size
                t1 = start_time + (step + 1) * grid_size
                note_start = max(note.start_time, t0)
                note_end = min(note.end_time, t1)
                note_duration = note_end - note_start
                fraction = note_duration / grid_size
                new_note = Note(pitch=note.pitch, velocity=note.velocity,
                                start_time=note_start, end_time=note_end)
                if fraction >= 0.5:
                    grid[step].append(new_note)
                else:
                    grid[step + 1].append(new_note)

    return grid

def quantize_note_to_grid(note_start, note_end, grid_size):
    # Round the note start and end times to the nearest grid position
    quantized_start = round(note_start / grid_size) * grid_size
    quantized_end = round(note_end / grid_size) * grid_size

    # Interpolate the pitch values for the quantized positions
    start_pitch = interpolate_pitch(note_start, note_end, quantized_start)
    end_pitch = interpolate_pitch(note_start, note_end, quantized_end)

    # Create a new quantized note with the interpolated pitch and quantized start and end times
    quantized_note = (quantized_start, quantized_end, start_pitch)

    return quantized_note

def quantize_note_to_step(note_start, note_end, step_size, grid_size):
    """
    Quantizes the note start and end times to the nearest step based on the specified step size (in seconds).
    Interpolates the start and end times to derive the quantized times.
    """
    # Compute the number of steps per grid
    steps_per_grid = int(grid_size / step_size)

    # Compute the start and end step indices
    start_step = round(note_start / step_size)
    end_step = round(note_end / step_size)

    # Compute the interpolated start and end times
    interpolated_start = start_step * step_size
    interpolated_end = end_step * step_size

    # Compute the start and end grid indices
    start_grid = start_step // steps_per_grid
    end_grid = end_step // steps_per_grid

    # Compute the quantized start and end times
    quantized_start = start_grid * grid_size + interpolated_start
    quantized_end = end_grid * grid_size + interpolated_end

    return quantized_start, quantized_end

def interpolate_pitch(note_start, note_end, quantized_time):
    # Determine the pitch at the quantized time using linear interpolation
    t = (quantized_time - note_start) / (note_end - note_start)
    start_pitch, end_pitch = note_start[2], note_end[2]
    quantized_pitch = start_pitch + t * (end_pitch - start_pitch)


def quantize_note_to_step(note_start, note_end, step_size, grid_size, tempo=120, time_signature=(4, 4)):
    """
    Quantizes the note start and end times to the nearest step based on the specified step size (in seconds).
    Interpolates the start and end times to derive the quantized times.
    """
    ticks_per_quarter_note = 480  # default value for MIDI
    ticks_per_second = ticks_per_quarter_note * tempo / (60 * 4)
    steps_per_grid = int(grid_size / step_size * ticks_per_second)

    # Compute the start and end step indices
    start_step = round(note_start * ticks_per_second / step_size)
    end_step = round(note_end * ticks_per_second / step_size)

    # Compute the interpolated start and end times
    interpolated_start = start_step * step_size / ticks_per_second
    interpolated_end = end_step * step_size / ticks_per_second

    # Compute the start and end grid indices
    start_grid = start_step // steps_per_grid
    end_grid = end_step // steps_per_grid

    # Compute the quantized start and end times
    quantized_start = start_grid * grid_size + interpolated_start
    quantized_end = end_grid * grid_size + interpolated_end

    return quantized_start, quantized_end

# def quantize_note_to_step(note_start, note_end, step_size, grid_size, tempo=120, time_signature=(4, 4)):
#     """
#     Quantizes the note start and end times to the nearest step based on the specified step size (in seconds).
#     Interpolates the start and end times to derive the quantized times.
#     """
#     ticks_per_beat = 480  # default value for MIDI
#     ticks_per_second = ticks_per_beat * tempo / 60
#     beats_per_measure = time_signature[0]
#     steps_per_beat = int(ticks_per_beat / step_size)
#     steps_per_measure = steps_per_beat * beats_per_measure
#     steps_per_grid = int(grid_size / step_size * ticks_per_second)
#
#     # Compute the start and end step indices
#     start_step = round(note_start * ticks_per_second / step_size)
#     end_step = round(note_end * ticks_per_second / step_size)
#
#     # Compute the interpolated start and end times
#     interpolated_start = start_step * step_size / ticks_per_second
#     interpolated_end = end_step * step_size / ticks_per_second
#
#     # Compute the start and end grid indices
#     start_grid = start_step // steps_per_grid
#     end_grid = end_step // steps_per_grid
#
#     # Compute the quantized start and end times
#     quantized_start = start_grid * grid_size + interpolated_start
#     quantized_end = end_grid * grid_size + interpolated_end
#
#     return quantized_start, quantized_end