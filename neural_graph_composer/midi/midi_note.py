"""
Midi note representation.
TODO. Take different sample Ableton , quantize and cross check for all quantization
delta difference and add as unit test.

Author Mus spyroot@gmail.com
"""
import logging
import math
import warnings
from decimal import Decimal
from typing import Optional, Tuple, Dict, Any, Union, Type

import librosa
import numpy as np

from .midi_abstract_event import MidiEvent
from .midi_spec import MAX_MIDI_VELOCITY
from .midi_spec import MAX_MIDI_PITCH
from .midi_spec import PITCH_FREQUENCIES, STANDARD_FREQ, REFERENCE_NOTE
from .midi_spec import SEMITONES_PER_OCTAVE
import json


class MidiNoteJSONDecoder(json.JSONDecoder):
    """Custom JSON decoder for decoding JSON strings into MidiNote objects.

    Usage:
    >>> from json import loads
    >>> decoder = MidiNoteJSONDecoder()
    >>> note_json = '{"__MidiNote__": true, "data": {"pitch": 60, "start_time": 0.0, "end_time": 1.0}}'
    >>> note = decoder.decode(note_json)

    :param object_hook: function that will be called for each dictionary parsed by the JSON decoder.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    @staticmethod
    def object_hook(dct) -> Union[Any, Dict[str, Any]]:
        """Hook that converts a dictionary into a MidiNote object if it has the correct format.
        :param dct: dictionary that was parsed by the JSON decoder.
        :return: MidiNote object or the original dictionary.
        """
        if '__MidiNote__' in dct:
            # Convert dictionary to MidiNote object
            return MidiNote(**dct['data'])
        else:
            return dct

    def decode(self, s, **kwargs) -> Any:
        """Decode a JSON string into a Python object.
         :param s: the JSON string to decode.
         :param kwargs: additional keyword arguments passed to the underlying `json.JSONDecoder.decode` method.
         :return: the Python object represented by the JSON string.
         """
        # decoding from bytes as well as str
        if isinstance(s, bytes):
            s = s.decode('utf-8')
        return super().decode(s, **kwargs)


class MidiNote(MidiEvent):
    """Represents a midi note.
    Note we maintain original information about instrument , program and the rest of data.
    The point here in case we need reconstruct order for particular instrument,
    find overlap etc. It good to have some reverse information.

    At moment midi sequence object hold just a list.
    On alternative implementation that I consider do in the future
    represent events as interval tree.

    """

    def __init__(self,
                 pitch: int,
                 start_time: float,
                 end_time: float,
                 program: Optional[int] = 0,
                 instrument: Optional[int] = 0,
                 velocity: Optional[int] = MAX_MIDI_VELOCITY,
                 quantized_start_step: Optional[int] = -1,
                 quantized_end_step: Optional[int] = -1,
                 voice: Optional[int] = 0,
                 numerator: Optional[int] = 4,
                 denominator: Optional[int] = 4,
                 is_drum: Optional[bool] = False,
                 ) -> None:
        """
        Create a new MIDI Note object. It consumed by MidiNoteSequence.
        :param instrument: midi instrument id.  i.e. what MIDI instrument generate that pitch sound
        :param program:  midi program what midi program generated given midi.
        :param start_time: midi start time is float.  i.e. check Pretty MIDI docs
        :param end_time:  midi end time,  a float Pretty MIDI provide
        :param pitch: MIDI pitch value (0-127)
        :param velocity: The velocity of the note (0-127).
        :param quantized_start_step: for quantization s start step
        :param quantized_end_step:  for quantization s start stop step
        :param numerator: default 4.  Note if value not present in MIDI it implicitly implied 4
        :param denominator: default 4 Note if value not present in MIDI it implicitly implied 4
        :param voice:
        :param is_drum:  a drum part or not.  This can inference from MIDI program Drum
        """
        # Ensure that pitch is within the valid MIDI range.
        if not 0 <= pitch <= MAX_MIDI_PITCH:
            warnings.warn(
                f"Pitch value {pitch} is out of range [0, 127]. Clamping to nearest valid value.")
            pitch = max(0, min(MAX_MIDI_PITCH, pitch))

        # clamp velocity value to range [0, 127]
        if not 0 <= velocity <= MAX_MIDI_VELOCITY:
            warnings.warn(
                f"Velocity value {velocity} is out of range [0, 127]. Clamping to nearest valid value.")
            velocity = max(0, min(MAX_MIDI_VELOCITY, velocity))

        assert end_time > start_time, f"Instrument {instrument} End time value {end_time} " \
                                      f"should be greater than start time value {start_time}"
        assert isinstance(numerator, int) and numerator > 0, \
            f"Numerator value {numerator} is not a valid value."

        # ensure that denominator is a power of 2.
        assert isinstance(denominator, int) and denominator > 0 and ((denominator & (denominator - 1)) == 0), \
            f"Denominator value {denominator} is not a valid value."

        # midi cc pitch
        self._pitch: int = pitch
        self._velocity: int = velocity
        self.pitch_name: str = librosa.midi_to_note(self.pitch)

        # midi instrument, program
        # note if all instruments merge to single instrument
        # then midi seq represent single midi seq.

        # we keep program in case particular note need play on different program.
        # this specifically for program change event.
        self._program: int = min(max(0, program), 255)
        if not 0 <= instrument <= 127:
            warnings.warn(
                "Invalid instrument value, should be in range [0, 127]. Clamping to valid range.")

        self._instrument = min(max(instrument, 0), 127)

        # start time in float
        self._start_time: float = start_time
        # end time in float
        self._end_time: float = end_time

        #
        self.voice: int = voice
        # the numerator describes the number of beats in a bar
        # or number of beats in a measure 4/4 i.e 4 beats in bar
        self.numerator: int = max(min(numerator, 255), 1)

        # denominator describes of what note value a beat is
        # (ie, how many quarter notes in a beat)
        # denominator is a negative power of two
        # 2 represents a quarter-note,
        # 3 represents an eighth-note
        self.denominator: int = max(min(denominator, 255), 1)

        # if given note is drum on and off
        self._is_drum = is_drum

        # if note quantized
        self._quantized_start_step: int = quantized_start_step
        self._quantized_end_step: int = quantized_end_step
        self._is_quantized = quantized_start_step >= 0 and quantized_end_step >= 0

    def __repr__(self) -> str:
        """ Return a string representation of the MidiNote
        object that can be used to recreate the object.
        :return:
        """
        return 'MidiNote(start={:f}, end={:f}, pitch={}, pitch_name={}, velocity={})'.format(
            self.start_time, self.end_time, self.pitch, self.pitch_name, self._velocity)

    def __str__(self) -> str:
        """Return a human-readable string representation of the Midi
        :return:
        """
        return 'Note(start={:f}, end={:f}, pitch={}, pitch_name={}, velocity={})'.format(
            self.start_time, self.end_time, self.pitch, self.pitch_name, self._velocity)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, MidiNote):
            return False
        return (self.pitch == other.pitch and
                self.velocity == other.velocity and
                self.program == other.program and
                self.instrument == other.instrument and
                self.start_time == other.start_time and
                self.end_time == other.end_time and
                self.quantized_start_step == other.quantized_start_step and
                self.quantized_end_step == other.quantized_end_step and
                self.voice == other.voice and
                self.numerator == other.numerator and
                self.denominator == other.denominator and
                self.is_drum == other.is_drum)

    def __lt__(self, other: 'MidiNote') -> bool:
        """

        :param other:
        :return:
        """
        if self.start_time != other.start_time:
            return self.start_time < other.start_time
        return self.end_time < other.end_time

    def __le__(self, other: 'MidiNote') -> bool:
        """Returns True if this note's start time is less than or
        equal to the other note's start time.
        If the start times are equal, the end times are compared.
        """
        if self.start_time < other.start_time:
            return True
        elif self.start_time == other.start_time:
            return self.end_time <= other.end_time
        else:
            return False

    @property
    def velocity(self) -> int:
        """Return the start time of the note in seconds.
        :return: The start time of the note as a float.
        """
        return self._velocity

    @property
    def start_time(self) -> float:
        """Return the start time of the note in seconds.
        :return: The start time of the note as a float.
        """
        return self._start_time

    @start_time.setter
    def start_time(self, value: float) -> None:
        """Set start time in second as float value
        :param value: The start time of the note as a float value in seconds
        :raise ValueError if value is less than 0 or start time is > end time
        """
        if value < 0:
            raise ValueError("Start time cannot be negative.")
        elif self._end_time is not None and value >= self._end_time:
            raise ValueError("Start time must be less than end time.")
        else:
            self._start_time = value

    @property
    def end_time(self) -> float:
        """Return end time as float value
        :return:
        """
        return self._end_time

    @end_time.setter
    def end_time(self, value) -> None:
        """Update midi end time
        :param value: The new end time in seconds.
        :raise ValueError if value less than 0 or new end time is < less than current start time.
        """
        if value < 0:
            raise ValueError("End time cannot be negative.")
        elif self._start_time is not None and value <= self._start_time:
            raise ValueError("End time must be greater than start time.")
        else:
            self._end_time = value

    @property
    def quantized_start_step(self) -> Optional[int]:
        """Return the current quantized start step of the note.
        Returns None if the note has not been quantized yet.
        :return: int or None
        """
        return self._quantized_start_step

    @quantized_start_step.setter
    def quantized_start_step(self, value) -> None:
        """Sets quantized_start_step and update
           state of MIDI note that it quantized.

        :param value: int, the new quantized start step to set.
        :return: None
        :raise ValueError if value is less than 0 or quantized start step is >= quantized end step.
        """
        if value < 0:
            raise ValueError("Quantized start step cannot be negative.")
        elif self._quantized_end_step is not None and value >= self._quantized_end_step:
            raise ValueError("Quantized start step must be less than quantized end step.")
        else:
            self._quantized_start_step = value
            self._is_quantized = True

    @property
    def quantized_end_step(self) -> Optional[int]:
        """ Returns the quantized end step of the note
        in the MIDI grid. If the note has not been quantized yet, returns None.
        :return:
        """
        return self._quantized_end_step

    @quantized_end_step.setter
    def quantized_end_step(self, value) -> None:
        """Return the current quantized end step of the note.
           Returns None if the note has not been quantized yet.
        :param value:
        :return:
        :raise ValueError if value is less than 0 or quantized start step is >= quantized end step.
        """
        if value < 0:
            raise ValueError("Quantized end step cannot be negative.")
        elif self._quantized_start_step is not None and value <= self._quantized_start_step:
            raise ValueError("Quantized end step must be greater than quantized start step.")
        else:
            self._quantized_end_step = value
            self._is_quantized = True

    def is_quantized(self) -> bool:
        """If note quantized return True
        :return:
        """
        return self.quantized_start_step >= 0 and self.quantized_end_step >= 0

    @staticmethod
    def quantize_to_nearest_step(midi_time: float, sps: int,
                                 amount: Optional[float] = 0.5,
                                 is_floor: Optional = True) -> float:
        """Quantize the given midi_time given in float second
        return step number as integer value

        The nearest compute step based on steps per second

        :param is_floor:
        :param midi_time The time in seconds that we use to find the nearest step.
        :param sps: the number of steps per second.
        :param amount is optional parameter is used to adjust the
                      quantization step size for example if 0.5
                      it means that the quantization step size is of the original
                      step size So if the original step size is 1 second per step,
                      then with amount=0.5, the quantization step size
                     becomes 0.5 seconds per step
        :raise ValueError if sps is less than 0 or amount is less than 0
        """

        if sps <= 0:
            raise ValueError("sps must be greater than zero")

        if amount <= 0:
            raise ValueError("amount must be greater than zero")
        if is_floor:
            quantized_step = math.floor(midi_time * sps + (1 - amount))
        else:
            quantized_step = math.ceil(midi_time * sps + (1 - amount))
        return max(0.0, quantized_step)

    @property
    def instrument(self) -> int:
        """
        :return:
        """
        return self._instrument

    @staticmethod
    def steps_to_seconds(steps: int, sps: int) -> float:
        """Converts the given number of steps to seconds based on the given steps per second (sps).

        :param steps: The number of steps to convert to seconds.
        :param sps: The number of steps per second (sps) used to convert steps to seconds.
        :return: The duration in seconds.
        """
        seconds = Decimal(steps) / Decimal(sps)
        return float(seconds.quantize(Decimal('.00001'), rounding='ROUND_HALF_UP'))

    def duration(self) -> float:
        """Duration of the note in seconds.
        :return: The duration of the note in seconds.
        """
        return self.end_time - self.start_time

    def step_duration_in_samples(self) -> int:
        """Return duration of the note in samples.
        The number of time steps between the start and end of the note
        is equivalent to the number of samples between those points.  i.e. interva.
        :return:
        """
        if not self.is_quantized():
            raise ValueError("Note is not quantized.")
        return self.quantized_end_step - self.quantized_start_step + 1

    def compute_step_duration_in_seconds(self, sps: int) -> float:
        """Returns the duration of the note in seconds based on the quantized
           start and end steps and the number of steps per second
        :param sps: step per second
        :raises ValueError: if the note is not quantized or sps is not a positive integer.
        :return: the duration of the note in seconds.
        """
        if not self.is_quantized():
            raise ValueError("Note is not quantized.")

        if sps <= 0:
            raise ValueError("sps must be a positive integer.")

        return np.float64(self.quantized_end_step - self.quantized_start_step) / np.float64(sps)

    def quantize_in_place(self, sps: int = 4, amount: Optional[float] = 0.5, min_step=None):
        """Quantize given note

        For example, if quantized_end_step is 12 and sps is 4,
        then the new time in seconds is 12 / 4 = 3.0.
        So new_end_time would be set to 3.0.

        Given a `steps per second` (sps) value, the `amount` of quantization (default 0.5),
        and an optional `min_step` size in samples, this method quantizes
        the start and end times of the note to the nearest step.
        If the note is already quantized, the method recomputes the quantized start and end steps.

        Quantization is the process of adjusting the timing of musical
        notes to align them with a specific time grid. This is commonly used
        in music production to correct the timing of performances and
        create a more consistent rhythm.


        :param sps: The number of steps per second to use for quantization. Defaults to 4.
        :param amount: The amount of quantization to apply, as a fraction of a step.
                       For example, an amount of 0.5 would quantize the note to the nearest half-step.
        :param min_step: The minimum step size in samples (optional).
        :raises ValueError: If the quantized start or end step is negative,
                            or if the quantized start step is greater
                            than or equal to the quantized end step.
        :return: None
        """
        quantized_start_step = self.quantize_to_nearest_step(self.start_time, sps, amount, is_floor=True)
        quantized_end_step = self.quantize_to_nearest_step(self.end_time, sps, amount, is_floor=False)

        if sps == 1:
            quantized_start_step = round(quantized_start_step)
            quantized_end_step = round(quantized_end_step)
            if quantized_start_step == quantized_end_step:
                quantized_end_step += 1

        if min_step is not None:
            quantized_start_step = max(self.quantized_start_step, quantized_start_step - min_step)
            quantized_end_step = max(self.quantized_end_step, quantized_end_step - min_step)

        # if already quantized , re-compute and update quantized_start_step and end step
        if self.is_quantized():
            # Calculate duration based on new quantized steps
            old_duration = np.divide(quantized_end_step - quantized_start_step, sps)
            logging.debug(f"old duration case on {old_duration}")
            # duration_before_shift = (self.end_time - self.start_time) / sps
            self.quantized_start_step = max(0, quantized_start_step)
            self.quantized_end_step = self.quantized_start_step + old_duration
        else:
            start_shift = max(0.0, -self.start_time)
            end_shift = max(0.0, -self.end_time)
            logging.debug(f"Old self.quantized_start_step {self.quantized_start_step}")
            self.quantized_start_step = quantized_start_step + int(start_shift * sps)
            logging.debug(f"### quantized_end_step {quantized_end_step} shift {end_shift} sps {sps}")
            self.quantized_end_step = quantized_end_step + int(end_shift * sps)
            logging.debug(f"New self.quantized_end_step {self.quantized_end_step}")

            # Set _is_quantized to True when quantifying for the first time.
            self._is_quantized = True

        logging.debug(f"Quantize quantized_start_step {self.quantized_start_step} {self.quantized_end_step}")

        # Do not allow notes to start or end in negative time.
        if self.quantized_start_step < 0:
            raise ValueError("Quantized start step cannot be negative.")
        elif self.quantized_end_step < 0:
            raise ValueError("Quantized end step cannot be negative.")
        elif self.quantized_start_step >= self.quantized_end_step:
            raise ValueError("Quantized start step must be less than quantized end step.")

        # Do not allow notes to start or end in negative time.
        self.quantized_start_step = max(0.0, self.quantized_start_step)
        self.quantized_end_step = max(0.0, self.quantized_end_step)
        logging.debug(f"New2 self.quantized_start_step {self.quantized_start_step}")

        if self.quantized_end_step == self.quantized_start_step:
            logging.debug("Executed this branch")
            self.quantized_end_step = max(min(self.quantized_end_step + 1, 999999), 1)

        logging.debug(f"self.quantized_start_step {self.quantized_start_step} and end step {self.quantized_end_step}")
        # Adjust the end time based on the new quantized end step.
        new_end_time = np.around(np.divide(quantized_end_step, int(sps)), 2)
        new_start_time = np.around(np.divide(quantized_start_step, int(sps)), 2)
        # If the new end time is earlier than the original end time,
        # adjust the start time as well.
        logging.debug(f"old timer {self.end_time} {self.start_time}")
        logging.debug(f"New3 self.quantized_start_step {self.quantized_start_step}")
        logging.debug(f"New new_end_time {new_end_time}")

        new_start_time = max(0.0, min(new_start_time, self.end_time - 0.001))
        new_end_time = max(0.0, min(new_end_time, 999999 - 0.001))

        duration = self.end_time - self.start_time
        if new_end_time < self.end_time:
            logging.debug(f"first case before {self.start_time} {new_end_time}")
            self.start_time = new_start_time
            logging.debug(f"first case set to {self.start_time} and new end time {self.start_time + duration}")
            self.end_time = self.start_time + duration
        else:
            logging.debug(f"second case new new_start_time time {new_start_time} new time {new_end_time}")
            self.end_time = new_end_time
            self.start_time = new_start_time

        # self.start_time = round(max(0.0, new_end_time - duration), 2)
        # self.end_time = round(new_end_time, 2)
        logging.debug(f"new start time {self.start_time} new end time {self.end_time}")
        self.quantized_start_step = max(0, self.quantized_start_step)

        self.quantized_step = sps

    def quantize(self, sps: int = 4, amount: float = 0.5, min_step: Optional[int] = None):
        """Returns a new quantized note with the specified number of steps per second (sps),
        quantization amount, and minimum step size.

        :param sps: the number of steps per second to use for quantization (default is 4).
        :param amount: float, the quantization amount (default is 0.5).
        :param min_step: the minimum step size in samples (optional). If provided,
                     the note will not be quantized to steps smaller than this
                     value. This can be used to prevent excessively small
                     durations that may not be supported by the destination
                     MIDI device or file format.
        :return: new quantized note
        """
        quantized_note = self.make_copy()
        quantized_note.quantize_in_place(sps=sps, amount=amount, min_step=min_step)
        return quantized_note

    def make_copy(self,
                  quantized_start_step: Optional[int] = None,
                  quantized_end_step: Optional[int] = None):
        """Returns a new copy of this note

        :param quantized_start_step: Optional[int], the new quantized start step for the copied note.
                                     Defaults to `None`, which uses the current quantized start step.
        :param quantized_end_step: Optional[int], the new quantized end step for the copied note.
                                   Defaults to `None`, which uses the current quantized end step.
        :return: a new instance of the `MidiNote` class that is a copy of the current instance.
        """
        return MidiNote(pitch=self.pitch,
                        start_time=self.start_time,
                        end_time=self.end_time,
                        program=self.program,
                        instrument=self.instrument,
                        velocity=self._velocity,
                        quantized_start_step=quantized_start_step
                        if quantized_start_step is not None else self.quantized_start_step,
                        quantized_end_step=quantized_end_step
                        if quantized_end_step is not None else self.quantized_end_step,
                        voice=self.voice,
                        numerator=self.numerator,
                        denominator=self.denominator,
                        is_drum=self.is_drum)

    @property
    def pitch(self):
        return self._pitch

    @pitch.setter
    def pitch(self, value):
        assert 0 <= value <= 127, f"Pitch value {value} is outside valid MIDI range (0-127)"
        self._pitch = value

    @property
    def program(self):
        return self._program

    @program.setter
    def program(self, value) -> None:
        """MIDI program
        :param value:
        :return:
        """
        assert 0 <= value <= 255, f"Program value {value} is outside valid range (0-255)"
        self._program = value

    @quantized_start_step.setter
    def quantized_start_step(self, value: float) -> None:
        """If a note start quantized is set we set value and note quantize flag
        :param value:
        :return:
        """
        if value < 0:
            raise ValueError("Quantized start step cannot be negative.")
        self._is_quantized = True
        self._quantized_start_step = value

    @quantized_end_step.setter
    def quantized_end_step(self, value: float) -> None:
        """If a note quantized we set value and note quantize flag.
        :param value:
        :return:
        """
        if value < 0:
            raise ValueError("Quantized end step cannot be negative.")
        self._is_quantized = True
        self._quantized_end_step = value

    def is_four_quarter_note(self) -> bool:
        """Returns True if the duration of the note is exactly 1 beat in 4/4 time signature.

        For example, if the time signature is 4/4 and the
        note starts at 0 seconds and ends at 1 second, this method
        will return True if the note is a quarter note.

        Explanation:
        In 4/4 time signature, there are 4 beats in one measure and each beat is a quarter note.
        Therefore, the duration of one beat in seconds is 60 / tempo, where tempo is the beats per minute.
        If the note's duration is exactly 1 beat, then its duration in seconds should be 60 / tempo seconds.
        This method checks if the actual duration of the note is close enough to the expected duration (within a
        relative tolerance of 1e-9). If the actual duration is close enough to the expected duration, then
        this method will return True.

        :return:  True if the duration of the note is exactly 1 beat in 4/4 time signature.
        """
        if self.denominator != 4 or self.numerator != 4:
            return False

        # duration of one beat in seconds
        beat_duration = 60 / self.denominator
        expected_duration = self.numerator * beat_duration  # duration of one measure in seconds
        actual_duration = self.duration()
        return math.isclose(actual_duration, expected_duration, rel_tol=1e-9)

    def duration_in_seconds(self, tempo: float) -> float:
        """Return duration of the note in seconds.
        It computed based on start and end time
        and duration() returns delta.
        :param tempo: The tempo in beats per minute.
        :return:The duration of the note in seconds.
        """
        return 60 * self.duration() / tempo

    def beat_duration(self, tempo: Optional[int] = 120) -> float:
        """Calculate the duration of one beat in seconds, based on the time signature of the note.
        :return: The duration of one beat in seconds.
        """
        numerator = self.numerator
        denominator = self.denominator

        beat_duration = 60.0 / tempo * (numerator / denominator)
        return beat_duration

    def bpm(self) -> float:
        """Calculate the BPM (beats per minute) of the note, assuming
        a constant tempo throughout the note.
        This method assumes that the tempo is constant
        throughout the duration of the note.

        It calculates the beats per minute (BPM) based on the duration
        of the note and the number of beats in the measure.
        :return: BPM (float)
        """
        # duration of one beat in seconds
        beat_times = np.arange(self.start_time, self.end_time, self.beat_duration())
        bpm = 60 / np.mean(np.diff(beat_times))
        return bpm

    @classmethod
    def from_name(
            cls, name: str, start_time: float, end_time: float,
            velocity: Optional[int] = 127,
            program: Optional[int] = 0,
            instrument: Optional[int] = 0, is_drum: Optional[bool] = False):
        """Create a MidiNote object from the name of the note (e.g. "C4").

        :param name: str, the name of the note (e.g. "C4").
        :param start_time: float, the start time of the note.
        :param end_time: float, the end time of the note.
        :param velocity: Optional[int], the velocity of the note (0-127). Default is 127.
        :param program: Optional[int], the MIDI program that generated the note (0-255). Default is 0.
        :param instrument: Optional[int], the MIDI instrument that generated the note (0-127). Default is 0.
        :param is_drum: Optional[bool], whether the note is a drum part or not. Default is False.
        :return: MidiNote, a MidiNote object.
        """
        pitch_value = librosa.note_to_midi(name)
        return cls(
            pitch=pitch_value,
            start_time=start_time,
            end_time=end_time,
            velocity=velocity,
            program=program,
            instrument=instrument,
            is_drum=is_drum
        )

    @staticmethod
    def note_to_freq(note: int) -> float:
        """Calculate note to freq a base reference A4.

        The note A4 is commonly used as a reference pitch,
        also known as concert pitch, and has a standardized frequency of 440 Hz.

        freq = STANDARD_FREQ * 2^((note - REFERENCE_NOTE) / SEMITONES_PER_OCTAVE)

        frequency = 440 * (2 ** ((note - A4) / 12))

        In Western music, the interval between any two adjacent notes in a chromatic
        scale is called a semitone.

        There are 12 semitones in an octave, and each octave consists of doubling
        the frequency of the note from the previous octave. This system is based
        on the equal temperament tuning, which divides the octave into 12
        equally spaced semitones, allowing for easy transposition and
        modulation between keys. So, the formula used to calculate the frequency
        of a note in Hz is based on this system, where
        the frequency ratio between any two adjacent semitones is equal to the
        twelfth root of 2 (approximately 1.059463).

        :param note:
        :return:
        """
        return STANDARD_FREQ * np.power(2, (note - REFERENCE_NOTE) / SEMITONES_PER_OCTAVE)

    @staticmethod
    def pitch_to_freq(pitch: str, octave: int) -> float:
        """Calculate the frequency of a pitch and octave in Hz.

        :param pitch: musical pitch (e.g. 'C', 'C#', 'D', etc.)
        :param octave: octave number (e.g. 4 for A440)
        :return: frequency in Hz

                :Example:
        >>> MusicNote.pitch_to_freq('A', 4)
        440.0
        >>> MusicNote.pitch_to_freq('C', 4)
        261.6255653005986
        >>> MusicNote.pitch_to_freq('F#', 5)
        739.988845423269
        """
        semitones_above_a4 = (octave - 4) * SEMITONES_PER_OCTAVE + PITCH_FREQUENCIES[pitch]
        return MidiNote.note_to_freq(REFERENCE_NOTE + semitones_above_a4)

    @staticmethod
    def is_four_eighth_note(note: int, tempo: int, time_signature: Tuple[int, int]) -> bool:
        """Checks if the MIDI note value represents a four eighth note,
          which is a note that lasts for half a beat in 4/4 time signature.
        :param tempo: midi tempo
        :param time_signature: midd time signature Tuple (4,4) etc.
        :param note:  note (int): MIDI note value.
        :return: True if the note represents a four eighth note, False otherwise.
        """
        # Calculate the expected duration of a single beat in seconds
        beat_duration = 60 / tempo
        # the duration of a quarter note in seconds
        quarter_note_duration = beat_duration * 4 / time_signature[1]
        #  the expected frequency of the note
        expected_freq = MidiNote.note_to_freq(note)
        #  the expected duration of the note in seconds
        expected_duration = 1 / expected_freq
        # Check if the note duration is within 0.1 of the expected duration
        return abs(expected_duration - quarter_note_duration) < 0.1 * quarter_note_duration

    @property
    def is_drum(self) -> bool:
        return self._is_drum

    @staticmethod
    def from_json(json_dict: Dict[str, Any]):
        """Constructs a `MidiNote` object from a dictionary in JSON format.
        :param json_dict: A dictionary in JSON format.
        :return: A `MidiNote` object.
        """
        return MidiNote(
            pitch=json_dict["pitch"],
            start_time=json_dict["start_time"],
            end_time=json_dict["end_time"],
            program=json_dict["program"],
            instrument=json_dict["instrument"],
            velocity=json_dict["velocity"],
            quantized_start_step=json_dict.get("quantized_start_step"),
            quantized_end_step=json_dict.get("quantized_end_step"),
            voice=json_dict.get("voice"),
            numerator=json_dict.get("numerator"),
            denominator=json_dict.get("denominator"),
            is_drum=json_dict.get("is_drum", False)
        )

    @property
    def event_start_time(self) -> float:
        """Implements MidiEvent so caller construct seq of events
        :return:  midi start time
        """
        return self.start_time

    @property
    def event_end_time(self) -> float:
        """Implements MidiEvent so caller construct seq of events
        :return: midi stop time
        """
        return self.end_time

    def shift_time(self, offset: float) -> None:
        """Shift time
        :param offset: offset that indicate how much we want offset
        :return:
        """
        self._start_time += offset
        self._end_time += offset

    def stretch(self, amount: float) -> None:
        """In place Stretch a note by amount
        :param amount:
        :return:
        """
        self._start_time *= amount
        self._end_time *= amount
