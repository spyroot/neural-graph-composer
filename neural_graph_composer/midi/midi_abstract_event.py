"""
Abstract Midi Event and MidiEvents.
MidiEvents indicate that object implemented get_midi_events
so caller can get all midi events for pitch or pitch bend etc.

Whereas MidiEvent it just event that has
two property that caller can use.

Author Mus spyroot@gmail.com
"""
from abc import abstractmethod
from typing import List


class MidiEvent:
    """Abstract midi event.
    Each class that has some notion of time should use this,
    so we can sort by time et.c
    """

    @property
    @abstractmethod
    def event_end_start(self):
        pass

    @property
    @abstractmethod
    def event_end_time(self):
        pass


class MidiEvents:
    """Abstract class indicate that object has some notion of midi event,
    and it can return collection of midi events.
    """
    @abstractmethod
    def get_midi_events(self) -> List[MidiEvent]:
        pass
