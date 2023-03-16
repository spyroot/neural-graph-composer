"""
Midi pitch bend event.
Author Mus spyroot@gmail.com
"""
from typing import Optional
from .midi_abstract_event import MidiEvent


class MidiPitchBend(MidiEvent):
    def __init__(
            self,
            amount: int,
            midi_time: Optional[float] = None,
            program: Optional[int] = 0,
            instrument: Optional[int] = 0,
            is_drum: Optional[bool] = False,
    ) -> None:
        """
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

        self.amount: int = min(max(-8191, amount), 8191)
        self.program: int = min(max(0, program), 255)
        self.instrument: int = min(max(0, instrument), 255)
        self.midi_time = midi_time
        self.is_drum: bool = is_drum

    @property
    def event_end_start(self):
        return self.start_time

    @property
    def event_end_time(self):
        return self.end_time
