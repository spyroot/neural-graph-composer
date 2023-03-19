"""
Collection utils used internally.
Note some of this will move. Quantization will move to separate
classes where each will use different algorithm.

Author
Mus spyroot@gmail.com
    mbayramo@stanford.edu
"""
import decimal
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
from scipy.spatial import distance


@np.vectorize
def steps_per_second_from_bpm(
        bpm: Optional[Union[float, np.float32]] = 120,
        resolution: Optional[int] = 480) -> Union[float, np.float32]:
    """Calculates steps per second (sps) given bpm and resolution.

    The tempo of a MIDI sequence in BPM (beats per minute)
    We can calculate the number of steps per second by dividing the tempo by 60.
    Steps per second = (120 BPM * 480 standard resolution) / 60 = 960

    :param bpm: The tempo of the MIDI sequence in BPM. Defaults to 120.
    :type bpm: float, optional
    :param resolution: The resolution of the MIDI sequence. Defaults to 480.
    :type resolution: int, optional
    :return: The number of steps per second.
    :rtype: float

    :Example:
    >>> steps_per_second_from_bpm(120, 480)
    960.0
    >>> steps_per_second_from_bpm(np.array([120, 90]), 480)
    array([960., 720.])
    """
    return bpm * resolution / 60.0


@np.vectorize
def from_bpm_spq_to_sps(
        steps_per_quarter: Union[int, float, np.float32], bpm: Union[float, np.float32],
        four_quarter: Optional[int] = 4) -> Union[float, np.float32]:
    """Calculates steps per second (sps) from steps
    per quarter and tempo in beats per minute (BPM).

    :param steps_per_quarter: Steps per quarter note (int, float, or numpy float32).
    :param bpm: Tempo in beats per minute (float or numpy float32).
    :param four_quarter: Optional int indicating the denominator of the time signature. Defaults to 4.
    :return: Steps per second (float or numpy float32).
     :Example:
    >>> from_bpm_spq_to_sps(480, 120)
    8.0
    >>> from_bpm_spq_to_sps(np.array([480, 240]), np.array([120, 60]))
    array([8., 4.])
    >>> from_bpm_spq_to_sps(480.0, 120.0)
    8.0
    >>> from_bpm_spq_to_sps(480, 120.0)
    8.0
    """
    qpm = bpm / four_quarter
    return steps_per_quarter * qpm / 60.0


@np.vectorize
def from_spq_to_sps(
        steps_per_quarter: Union[float, np.float32],
        qpm: Union[float, np.float32]) -> Union[float, np.float32]:
    """Calculates steps per second (sps ) from steps per quarter.
    The tempo of a MIDI sequence in BPM (beats per minute)

    We can calculate the number of steps per second by dividing the tempo by 60.
    Steps per second = (120 BPM * 480 standard resolution) / 60 = 960

    :param steps_per_quarter: Steps per quarter note (int, float, or numpy float32).
    :param qpm: quarter notes per minute
    :return:
    """
    return steps_per_quarter * qpm / 60.0


@np.vectorize
def bpm_to_qpm_np(bpm: Optional[Union[float, np.float32]] = 120.0,
                  beats_per_measure: Optional[int] = 4) -> Union[float, np.float32]:
    """Numeric stable version, Convert beats per minute (BPM) to quarter notes per minute (QPM).
    :param bpm: The tempo in beats per minute (default 120 BPM).
    :param beats_per_measure: The number of beats per measure (default 4).
    :return: The tempo in quarter notes per minute.
    """
    qpm = (bpm * beats_per_measure) / 60.0
    return np.round(qpm, decimals=10)


@np.vectorize
def bpm_to_qpm(bpm: Union[float, np.float32] = 120.0,
               beats_per_measure: Optional[int] = 4) -> Union[float, np.float32]:
    """Convert beats per minute (BPM) to quarter notes per minute (QPM).
    :param bpm: The tempo in beats per minute (default 120 BPM).
    :param beats_per_measure: The number of beats per measure (default 4).
    :return: The tempo in quarter notes per minute.
    """
    qpm = (bpm / 60.0) * beats_per_measure
    return qpm


@np.vectorize
def qpm_to_bpm(
        qpm: Union[float, np.float32] = 120.0,
        quarter_note: int = 4) -> Union[float, np.float32]:
    """ QPM (Quarter Notes per Minute) is a measure of tempo in music,
        BPM (Beats per Minute) , convert QPM to BPM.
    :param qpm: The tempo in QPM (default 120 QPM).
    :param quarter_note: The denominator of the time signature (default 4).
    :return: The tempo in BPM (float or numpy float32).
    :example:
    >>> qpm_to_bpm(120, 4)
    480.0
    """
    return qpm * quarter_note


@np.vectorize
def frames_from_times(
        start_time: Union[float, np.float32],
        end_time: Union[float, np.float32],
        fps: Union[float, np.float32],
        min_frame_occupancy: Optional[float] = 0.0) -> Tuple[int, int]:

    """Convert start and end times to frame indices.
     takes as input the start and end times of an event, the frames per second (fps)

    :param start_time: The start time of the event, in seconds.
    :param end_time: The end time of the event, in seconds.
    :param fps: The frames per second.
    :param min_frame_occupancy: The minimum occupancy threshold for each frame,
                                as a fraction of the total frame duration.
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


def ft_quantize_to_nearest_step(unquantized_time: float,
                                steps_per_second: float,
                                rounding_threshold: float = 0.5) -> int:
    """Returns the nearest step based on the provided steps per second.
    :param unquantized_time:  The unquantized time positive in seconds.
    :param steps_per_second:  The number of steps per second to use for quantization.
    :param rounding_threshold:  The rounding threshold to use. If the fractional part of the
                            result is greater than or equal
                            to the rounding threshold, the result is rounded up to
                             the nearest integer. Otherwise, the result is rounded down.
    :return:  The nearest step based on the provided steps per second.
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
    unquantized_steps = unquantized_time * steps_per_second
    threshold = steps_per_second * rounding_threshold

    if unquantized_steps < 0:
        remainder = abs(unquantized_steps) % 1
        if remainder >= threshold:
            return int(unquantized_steps) - 1
        else:
            return int(unquantized_steps)
    else:
        remainder = unquantized_steps % 1
        if remainder >= threshold:
            return int(unquantized_steps) + 1
        else:
            return int(unquantized_steps)


def quantize_to_nearest_step(un_quantized_time, step_size, tempo=120, time_signature=(4, 4)):
    """
    Quantizes the input time to the nearest step based on the specified step size (in beats).
    """
    ticks_per_beat = 480  # default value for MIDI
    if step_size == 1:
        ticks_per_step = math.floor(ticks_per_beat * time_signature[0] / step_size)
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

def quantize_to_nearest_step_alt(unquantized_time, steps_per_second, rounding_threshold):
    """

    :param unquantized_time:
    :param steps_per_second:
    :param rounding_threshold:
    :return:
    """
    decimal_unquantized_time = decimal.Decimal(str(unquantized_time))
    decimal_steps_per_second = decimal.Decimal(str(steps_per_second))
    decimal_rounding_threshold = decimal.Decimal(str(rounding_threshold))
    decimal_quantized_steps = decimal_unquantized_time * decimal_steps_per_second - decimal_rounding_threshold
    quantized_steps = int(decimal_quantized_steps.to_integral_value(decimal.ROUND_HALF_UP))
    return quantized_steps


def vector_quantize(notes, grid, metric=distance.euclidean):
    """
    :param notes:
    :param grid:
    :param metric:
    :return:
    """
    # calculate distances between each note and grid point
    distances = np.zeros((len(notes), len(grid)))
    for i, note in enumerate(notes):
        for j, point in enumerate(grid):
            distances[i, j] = metric(note, point)

    # assign each note to the nearest grid point
    quantized_notes = []
    for i in range(len(notes)):
        j = np.argmin(distances[i])
        quantized_notes.append(grid[j])

    return quantized_notes


def quantize_and_snap_to_grid(notes, grid_size):
    """Quantizes the notes to the given grid size using linear interpolation.
    Alternative version.
    """
    start_time = min(notes, key=lambda n: note.start_time).start_time
    end_time = max(notes, key=lambda n: note.end_time).end_time
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
                # # new_note = MidiNote(pitch=note.pitch, velocity=note.velocity,
                # #                 start_time=note_start, end_time=note_end)
                # if fraction >= 0.5:
                #     grid[step].append(new_note)
                # else:
                #     grid[step + 1].append(new_note)

    return grid


def quantize_note_to_step_grid(
        note_start: float, note_end: float,
        step_size, grid_size: float) -> Tuple[float, float]:
    """ Quantizes the note start and end times to the nearest
    step based on grid.

    :param note_start: The start time of the note (positive).
    :param note_end: The end time of the note (positive).
    :param step_size: The step size in seconds.
    :param grid_size:  The grid size in seconds.
    :return: tuple: A tuple containing the quantized start and end times (floats).

    """
    if note_start < 0 or note_end < 0:
        raise ValueError("note_start and note_end must be positive.")

    # Number of steps per grid, and start and end indices
    steps_per_grid = int(grid_size / step_size)
    start_step = round(note_start / step_size)
    end_step = round(note_end / step_size)

    # the start and end grid indices
    start_grid = start_step // steps_per_grid
    end_grid = end_step // steps_per_grid

    quantized_start = start_grid * grid_size + (start_step % steps_per_grid) * step_size
    quantized_end = end_grid * grid_size + (end_step % steps_per_grid) * step_size

    return quantized_start, quantized_end


def interpolates_quantize(
        note_start: float, note_end: float,
        step_size: float, grid_size, tempo=120, time_signature=(4, 4)):
    """Quantizes the note start and end times to the nearest step based on the specified step size (in seconds).
    Unlike other version here we interpolate the start and end times to derive the quantized times
    and use grid quantization and interpolation.

    :param note_start:
    :param note_end:
    :param step_size:
    :param grid_size:
    :param tempo:
    :param time_signature:
    :return:
    """
    ticks_per_beat = 480  # default value for MIDI
    ticks_per_second = ticks_per_beat * tempo / 60
    beats_per_measure = time_signature[0]
    steps_per_beat = int(ticks_per_beat / step_size)
    steps_per_measure = steps_per_beat * beats_per_measure
    steps_per_grid = int(grid_size / step_size * ticks_per_second)

    # Compute the start and end step indices
    start_step = round(note_start * ticks_per_second / step_size)
    end_step = round(note_end * ticks_per_second / step_size)

    # Compute the interpolated start and end times
    interpolated_start = start_step * step_size / ticks_per_second
    interpolated_end = end_step * step_size / ticks_per_second

    # compute the start and end grid indices
    start_grid = start_step // steps_per_grid
    end_grid = end_step // steps_per_grid

    # compute the quantized start and end times
    quantized_start = start_grid * grid_size + (start_step % steps_per_grid) * step_size
    quantized_end = end_grid * grid_size + (end_step % steps_per_grid) * step_size
    return quantized_start, quantized_end


@dataclass(frozen=True)
class Pitch:
    """
    TODO move this and merge with with Music Scale
    """
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


def check_quantization(note_start, note_end, step_size, grid_size, expected_start, expected_end):
    quantized_start, quantized_end = quantize_note_to_step_grid(note_start, note_end, step_size, grid_size)
    assert quantized_start == expected_start, f"Expected start: {expected_start}, got: {quantized_start}"
    assert quantized_end == expected_end, f"Expected end: {expected_end}, got: {quantized_end}"


def check_grid_quantization():
    """
    :return:
    """
    # simple case with step_size = 1 and grid_size = 4
    check_quantization(1.2, 2.8, 1, 4, 1, 3)

    # different step_size and grid_size
    check_quantization(1.6, 3.7, 0.5, 2, 1.5, 3.5)

    # note_start and note_end are exactly on the grid
    check_quantization(2, 6, 1, 4, 2, 6)

    # # note_start and note_end are between steps
    check_quantization(1.25, 2.75, 0.5, 2, 1, 3)

    # arger step_size and grid_size
    check_quantization(1.8, 7.2, 2, 8, 2, 8)


def check_for_nearest():
    """Basic test for nearest step quantization.
    :return:
    """
    print(f"ft_quantize_to_nearest_step(1.5, 2) Expect {3} got {ft_quantize_to_nearest_step(1.5, 2)}")
    print(f"ft_quantize_to_nearest_step(1.4, 2) Expect {3} got {ft_quantize_to_nearest_step(1.4, 2)}")
    print(f"ft_quantize_to_nearest_step(1.6, 2) Expect {3} got {ft_quantize_to_nearest_step(1.6, 2)}")
    print(f"ft_quantize_to_nearest_step(-1.5, 2) Expect {-3} got {ft_quantize_to_nearest_step(-1.5, 2)}")
    print(f"ft_quantize_to_nearest_step(1.5, 2, 0.6) Expect {3} got {ft_quantize_to_nearest_step(1.5, 2, 0.6)}")

    # With steps_per_second = 4, there are 4 steps in 1 second, so 1.5 seconds corresponds to 6 steps
    print(f"ft_quantize_to_nearest_step(1.5, 4) Expect {6} got {ft_quantize_to_nearest_step(1.5, 4)}")

    # With steps_per_second = 8, there are 8 steps in 1 second, so 1.5 seconds corresponds to 12 steps
    print(f"ft_quantize_to_nearest_step(1.5, 8) Expect {12} got {ft_quantize_to_nearest_step(1.5, 8)}")

    # With steps_per_second = 16, there are 16 steps in 1 second, so 1.5 seconds corresponds to 24 steps
    print(f"ft_quantize_to_nearest_step(1.5, 16) Expect {24} got {ft_quantize_to_nearest_step(1.5, 16)}")

    # With steps_per_second = 32, there are 32 steps in 1 second, so 1.5 seconds corresponds to 48 steps
    print(f"ft_quantize_to_nearest_step(1.5, 32) Expect {48} got {ft_quantize_to_nearest_step(1.5, 32)}")

    ##
    # unquantized_time = 0.75, steps_per_second = 4
    # 4 steps in 1 second, so 0.75 seconds corresponds to 3 steps
    print(f"ft_quantize_to_nearest_step(0.75, 4) Expect {3} got {ft_quantize_to_nearest_step(0.75, 4)}")
    # unquantized_time = 2.25, steps_per_second = 8
    # 8 steps in 1 second, so 2.25 seconds corresponds to 18 steps
    print(f"ft_quantize_to_nearest_step(2.25, 8) Expect {18} got {ft_quantize_to_nearest_step(2.25, 8)}")
    # unquantized_time = 0.3, steps_per_second = 16
    # 16 steps in 1 second, so 0.3 seconds corresponds to 4.8 steps, rounded down to 4 steps (
    # using the default rounding_threshold of 0.5)
    print(f"ft_quantize_to_nearest_step(0.3, 16) Expect {4} got {ft_quantize_to_nearest_step(0.3, 16)}")

    # unquantized_time = 1.7, steps_per_second = 32
    # 32 steps in 1 second, so 1.7 seconds corresponds to 54.4 steps, rounded down to 54 steps
    # (using the default rounding_threshold of 0.5)
    print(f"ft_quantize_to_nearest_step(1.7, 32) Expect {54} got {ft_quantize_to_nearest_step(1.7, 32)}")

    # unquantized_time = 1.5, steps_per_second = 4, rounding_threshold = 0.4
    # 4 steps in 1 second, so 1.5 seconds corresponds to 6 steps
    # rounding_threshold is not relevant here because the result is an exact integer
    print(f"ft_quantize_to_nearest_step(1.5, 4, 0.4) Expect {6} got {ft_quantize_to_nearest_step(1.5, 4, 0.4)}")

    # unquantized_time = 0.35, steps_per_second = 4, rounding_threshold = 0.4
    # 4 steps in 1 second, so 0.35 seconds corresponds to 1.4 steps
    #  rounding_threshold = 0.4, it should round up to 2 steps
    print(f"ft_quantize_to_nearest_step(0.35, 4, 0.4) Expect {2} got {ft_quantize_to_nearest_step(0.35, 4, 0.4)}")

    # unquantized_time = 0.35, steps_per_second = 4, rounding_threshold = 0.5
    # 4 steps in 1 second, so 0.35 seconds corresponds to 1.4 steps
    #  rounding_threshold = 0.5, it should round down to 1 step
    print(f"ft_quantize_to_nearest_step(0.35, 4, 0.5) Expect {1} got {ft_quantize_to_nearest_step(0.35, 4, 0.5)}")

    # unquantized_time = 0.65, steps_per_second = 4, rounding_threshold = 0.6
    # 4 steps in 1 second, so 0.65 seconds corresponds to 2.6 steps
    #  rounding_threshold = 0.6, it should round down to 2 steps
    print(f"ft_quantize_to_nearest_step(0.65, 4, 0.6) Expect {2} got {ft_quantize_to_nearest_step(0.65, 4, 0.6)}")


if __name__ == '__main__':
    check_for_nearest()
    check_grid_quantization()
