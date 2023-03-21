"""
Midi pitch bend event.
Author Mus spyroot@gmail.com
"""

import numpy as np
from .midi_abstract_event import MidiEvent
from typing import Optional, Union


class MidiPitchBend(MidiEvent):
    """Represents a MIDI pitch bend event.
    """
    def __init__(
            self,
            amount: int,
            midi_time: Optional[Union[float, np.ndarray]] = 0.0,
            program: Optional[int] = 0,
            instrument: Optional[int] = 0,
            is_drum: Optional[bool] = False,
    ) -> None:
        """ Initializes a new instance of the MidiPitchBend class.
        and bounded to particular instrument , track.

        MIDI Pitch Bend event. Is drum if event in drum channel
        :param amount is amount of pitch bend. Based on MIDI it values between -8191 and 8191
        :param program: midi program , default 0
        :param instrument:  midi instrument default 0
        :param midi_time: time of event.  time of event
        :param is_drum: default False
        """
        if not (-8191 <= amount <= 8191):
            raise ValueError("amount must be between -8191 and 8191")
        self.amount: int = amount

        if not (0 <= program <= 127):
            raise ValueError("program must be between 0 and 127")
        self.program: int = program

        if not (0 <= instrument <= 127):
            raise ValueError("instrument must be between 0 and 127")
        self.instrument: int = instrument

        if midi_time is not None and midi_time < 0:
            raise ValueError("midi_time must be non-negative")

        if midi_time is not None:
            if isinstance(midi_time, (float, int)):
                if midi_time < 0:
                    raise ValueError("midi_time must be non-negative")
                self.__midi_time: float = float(midi_time)
            elif isinstance(midi_time, np.ndarray):
                if (midi_time < 0).any():
                    raise ValueError("midi_time must be non-negative")
                self.__midi_time: np.ndarray[float] = midi_time.astype(float)
            else:
                raise TypeError("midi_time must be a float, int, or numpy array.")
        else:
            self.__midi_time: Optional[Union[float, np.ndarray]] = None

        self.__amount: int = min(max(-8191, amount), 8191)
        self.__program: int = min(max(0, program), 255)
        self.__instrument: int = min(max(0, instrument), 255)
        self.__midi_time: Union[float, np.ndarray] = np.clip(midi_time, 0, np.inf) if midi_time is not None else 0.0
        self.__is_drum: bool = is_drum

    def __str__(self) -> str:
        return f"MidiPitchBend(amount={self.amount}, program={self.program}, " \
               f"instrument={self.instrument}, midi_time={self.midi_time}, is_drum={self.is_drum})"

    def __repr__(self) -> str:
        return f"MidiPitchBend(amount={self.amount}, program={self.program}, " \
               f"instrument={self.instrument}, midi_time={self.midi_time}, is_drum={self.is_drum})"

    def __lt__(self, other):
        """
        :param other:
        :return:
        """
        if isinstance(other, MidiPitchBend):
            return self.amount < other.amount
        return NotImplemented

    def __eq__(self, other):
        """
        :param other:
        :return:
        """
        if isinstance(other, MidiPitchBend):
            if isinstance(other, MidiPitchBend):
                return (
                        self.amount == other.amount
                        and self.program == other.program
                        and self.instrument == other.instrument
                        and self.midi_time == other.midi_time
                        and self.is_drum == other.is_drum
                )
        return NotImplemented

    @property
    def event_start_time(self) -> Optional[float]:
        """ Returns the start time of the event.
        :return: The start time of the event.
        :rtype: float
        """
        return self.midi_time

    @property
    def event_end_time(self) -> Optional[float]:
        """Returns the end time of the event.
        :return: The end time of the event.
        :rtype: float
        """
        return self.midi_time

    @property
    def midi_time(self) -> Union[float, np.ndarray]:
        return self.__midi_time

    @midi_time.setter
    def midi_time(self, value: Union[float, np.ndarray]) -> None:
        self.__midi_time = np.clip(value, 0, None)

    @midi_time.setter
    def midi_time(self, value: Optional[float]) -> None:
        if value is not None and value < 0:
            raise ValueError("midi_time must be non-negative")
        self.__midi_time = value

    @property
    def program(self) -> int:
        return self.__program

    @program.setter
    def program(self, value: int) -> None:
        if not (0 <= value <= 127):
            raise ValueError("program must be between 0 and 127")
        self.__program = value

    @property
    def instrument(self) -> int:
        return self.__instrument

    @instrument.setter
    def instrument(self, value: int) -> None:
        if not (0 <= value <= 127):
            raise ValueError("instrument must be between 0 and 127")
        self.__instrument = value

    @property
    def is_drum(self) -> bool:
        return self.__is_drum
