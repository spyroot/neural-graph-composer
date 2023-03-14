import math
from dataclasses import dataclass
from typing import Optional


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


def bpm_to_qpm(bpm: Optional[int] = 120, four_quarter: Optional[int] = 4) -> float:
    return bpm / four_quarter


def qpm_to_bpm(qpm: Optional[int] = 120, quarter_note: Optional[int] = 4):
    """ QPM (Quarter Notes per Minute) is a measure of tempo in music,
        BPM (Beats per Minute) , convert QPM to BPM.
    :param qpm:
    :param quarter_note:
    :return:
    """
    return qpm * quarter_note


def frames_from_times(start_time, end_time, fps, min_frame_occupancy):
    # Will round down because note may start or end in the middle of the frame.
    start_frame = int(start_time * fps)
    start_fps = (start_frame + 1 - start_time * fps)

    # # check for > 0.0 to avoid possible numerical issues
    if min_frame_occupancy > 0.0 and start_fps < min_frame_occupancy:
        start_frame += 1

    end_frame = int(math.ceil(end_time * fps))
    end_frame_occupancy = end_time * fps - start_frame - 1

    if min_frame_occupancy > 0.0 and end_frame_occupancy < min_frame_occupancy:
        end_frame -= 1
        end_frame = max(start_frame, end_frame)

    return start_frame, end_frame


def quantize_to_nearest_step(un_quantized_time,
                             steps_per_second,
                             cutoff=0.5):
    """Quantizes to a nears step based on step per second."""
    un_quantized_steps = un_quantized_time * steps_per_second
    return int(un_quantized_steps + (1 - cutoff))
