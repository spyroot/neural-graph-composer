"""
Class represent Midi time signature message.

Author
Mus spyroot@gmail.com
    mbayramo@stanford.edu
"""
import logging
import math
from functools import cache
from typing import Optional

from .midi_seq import MidiSeq
from .midi_spec import DEFAULT_PPQ
from .midi_spec import DEFAULT_QPM


class MidiTempoSignature(MidiSeq):
    """
    """
    SECONDS_PER_MINUTE = 60.0

    def __init__(self,
                 midi_time: Optional[float] = 0.0,
                 qpm: Optional[float] = float(DEFAULT_QPM),
                 resolution: Optional[int] = int(DEFAULT_PPQ)) -> None:
        """
        :param midi_time: The time at which the tempo signature takes effect, in ticks.
        :param qpm: The tempo, in quarter notes per minute.
        :param resolution: The ticks per quarter note of the MIDI sequence
        """
        super().__init__()

        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)
        self.logger.setLevel(logging.WARNING)

        self.logger.debug(f"Creating tempo signature "
                          f"midi time {midi_time} "
                          f"qpm {qpm} "
                          f"resolution {resolution}")

        if qpm is not None:
            assert isinstance(qpm, (int, float)), f"qpm must be a number, got {type(qpm)}"
            assert qpm > 0, f"qpm must be greater than 0, got {qpm}"
            self.qpm = float(qpm)
        else:
            self.qpm = DEFAULT_QPM

        if resolution is not None:
            assert isinstance(resolution, int), f"resolution must be an integer, got {type(resolution)}"
            assert resolution > 0, f"resolution must be greater than 0, got {resolution}"
            self.resolution = int(resolution)
        else:
            self.resolution = DEFAULT_PPQ

        if midi_time is not None:
            assert isinstance(midi_time, (int, float)), f"midi_time must be a number, got {type(midi_time)}"
            self.midi_time = float(midi_time)
        else:
            self.midi_time = 0.0

    @staticmethod
    def tempo_to_qpm(tempo: float) -> float:
        """Compute the quarter notes per minute (QPM) based on the tempo signature
        events in the MIDI sequence.
        qpm = 60 / (tempo / 1000000)
        :param tempo: take tempo.
        :return: return quarter notes per minute
        """
        return 60.0 / (tempo / 1000000.0)

    @property
    def qpm(self) -> Optional[float]:
        """Return the tempo in quarter notes per minute (QPM).
        """
        return self._qpm

    @qpm.setter
    def qpm(self, value: float) -> None:
        """Set the tempo in quarter notes per minute (QPM).
        """
        self._qpm = value

    @cache
    def average_qpm(self, total_time, tempo_signature):
        """Compute the quarter notes per minute (QPM) based on the tempo signature
          events in the MIDI sequence.
        :return: The QPM for the MIDI sequence
        """
        if len(tempo_signature) == 0:
            return DEFAULT_QPM

        # Compute the average tempo over all tempo signature events in the MIDI sequence
        total_time = total_time
        total_ticks = sum([ts.tempo_ticks for ts in tempo_signature])
        average_ticks_per_second = total_ticks / total_time
        qpm = average_ticks_per_second * 60 / self.resolution
        return qpm

    @staticmethod
    def ppq_to_qpm(ppq, tempo_microseconds_per_quarter_note, ticks_per_beat=4):
        """
        :param ppq:
        :param tempo_microseconds_per_quarter_note:
        :param ticks_per_beat default 4
        :return:
        """
        tick_duration = 1 / (ticks_per_beat * tempo_microseconds_per_quarter_note * 1e-6)
        quarter_note_duration = tick_duration * ppq
        qpm = 60 / quarter_note_duration
        qpm_ticks_per_beat = qpm / (ticks_per_beat * tempo_microseconds_per_quarter_note * 1e-6)
        return qpm_ticks_per_beat

    @property
    def seconds_per_tick(self) -> float:
        """Return the duration of a single tick in seconds."""
        if not self.qpm:
            return 0.0

        ticks_per_second = self.qpm * self.resolution / MidiTempoSignature.SECONDS_PER_MINUTE
        return 1.0 / ticks_per_second

    @property
    def seconds_per_quarter(self) -> float:
        """Return the duration of a quarter note in seconds."""
        return self.seconds_per_tick * self.resolution

    @property
    def bpm(self) -> float:
        """Return the tempo in beats per minute from qpm."""
        return self.qpm / 4.0

    @bpm.setter
    def bpm(self, value: float):
        """Set the tempo in beats per minute."""
        self.qpm = value * 4.0

    def __repr__(self):
        """
        :return:
        """
        return "MidiTempoSignature(time={}, qpm={})".format(
            self.midi_time, self.qpm)

    def __str__(self):
        """
        :return:
        """
        return '{} qpm {:.2f} seconds'.format(
            self.midi_time, self.qpm)

    def __lt__(self, other):
        if math.isclose(self.midi_time, other.midi_time):
            return self.seq < other.seq

        return self.midi_time < other.midi_time


class MidiTimeSignature(MidiSeq):
    """
    Time signature is expressed as 4 numbers.
    nn and dd represent the "numerator" and "denominator" of the
    signature as notated on sheet music.

    The denominator is a negative power of 2: 2 = quarter note, 3 = eighth, etc.
    The cc expresses the number of MIDI clocks in a metronome click.

    NOTE: If there are no time signature events in a MIDI file,
    then the time signature is assumed to be 4/4.

    Examples
    --------
    Instantiate a TimeSignature object with 6/8 time signature at 3.14 seconds:

    >>> ts = MidiTempoSignature(6, 8, 3.14)
    >>> print(ts)
    6/8 at 3.14 seconds
    """

    def __init__(self,
                 midi_time: Optional[float] = 0.0,
                 numerator: Optional[int] = 4,
                 denominator: Optional[int] = 4) -> None:
        """Initialize a new `MidiTimeSignature` object.
        :param midi_time: The time in seconds at which the time signature takes effect.
        :param denominator: The denominator of the time signature as notated on sheet music. Default value is 4.
        :param numerator: The numerator of the time signature as notated on sheet music. Default value is 4
        :raise TypeError If `denominator` or `numerator` is not an integer.
        :raise ValueError If `denominator` or `numerator` is less than or equal to 0, or if
                         `denominator` is not a negative power of 2.
        """
        super().__init__()
        if not isinstance(denominator, int) or not isinstance(numerator, int):
            raise TypeError("Denominator and numerator must be integers.")

        if denominator <= 0 or numerator <= 0:
            raise ValueError("Denominator and numerator must be positive integers.")
        if not math.log2(denominator).is_integer():
            raise ValueError(f"Provided denominator {denominator} must be a negative power of 2 (e.g. 2, 4, 8, 16).")

        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)
        self.logger.setLevel(logging.WARNING)
        self.logger.debug(f"Creating time signature "
                          f"midi time {midi_time} "
                          f"numerator {numerator} "
                          f"numerator {numerator}")

        self.denominator: int = denominator
        self.numerator: int = numerator
        self.midi_time: float = midi_time
        self.seq = 0

    @property
    def measure_length(self) -> float:
        """Return the length of a measure in seconds.
        :return:
        """
        return self.midi_time * self.numerator / self.denominator

    def beats_per_measure(self) -> int:
        """Return the number of beats per measure based on the time signature.
        If the denominator of the time signature is a negative power of 2 (e.g. 2, 4, 8, 16),
        the beat duration is assumed to be a quarter note, and the method returns the
        numerator of the time signature.

        If the denominator is not a negative power of 2, the beat duration is assumed to be
        the length of a measure divided by the denominator, and the method returns the
        length of a measure in beats.

         Examples
        --------
        >>> ts = MidiTimeSignature(numerator=4, denominator=4)
        >>> ts.beat_per_measure()
        4.0

        >>> ts = MidiTimeSignature(numerator=3, denominator=4)
        >>> ts.beat_per_measure()
        3.0

        >>> ts = MidiTimeSignature(numerator=6, denominator=8)
        >>> ts.beat_per_measure()
        2.0

        >>> ts = MidiTimeSignature(numerator=4, denominator=6)
        >>> ts.beat_per_measure()
        4.0 / 3.0
        :return: The number of beats per measure.
        :return:
        """
        if self.denominator == 1:
            return self.numerator

        denominator_pow = int(math.log2(self.denominator))
        if 2 ** denominator_pow != self.denominator:
            raise ValueError("Invalid time signature: denominator is not a negative power of 2")

        return self.numerator * 2 ** (denominator_pow - 2)

    def __repr__(self):
        """
        :return:
        """
        return "MidiTempoSignature(seq={}, numerator={}, denominator={}, time={})".format(
            self.seq, self.numerator, self.denominator, self.midi_time)

    def __str__(self):
        """
        :return:
        """
        return 'seq {} {}/{} at {:.2f} seconds'.format(
            self.seq, self.numerator, self.denominator, self.midi_time)

    def __lt__(self, other):
        """
        :param other:
        :return:
        """
        if math.isclose(self.midi_time, other.midi_time):
            return self.seq < other.seq

        return self.midi_time < other.midi_time
