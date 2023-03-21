"""
Abstract Midi Event and MidiEvents.
MidiEvents indicate that object implemented get_midi_events
so caller can get all midi events for pitch or pitch bend etc.

Whereas MidiEvent it just event that has
two property that caller can use.

Author Mus spyroot@gmail.com
"""
from abc import abstractmethod
from typing import List, Optional


class MidiEvent:
    """Abstract midi event.
    Each class that has some notion of time should use this,
    so we can sort by time et.c
    """

    @property
    @abstractmethod
    def event_start_time(self) -> Optional[float]:
        """Abstract property that should be implemented by subclasses.
        Returns the start time of the MIDI event.
        :return: A float representing the start time of the MIDI event, or None if the event has no start time.
        """
        pass

    @property
    @abstractmethod
    def event_end_time(self) -> Optional[float]:
        """Abstract property that should be implemented by subclasses.
        Returns the end time of the MIDI event.
        :return: A float representing the end time of the MIDI event, or None if the event has no start time.
        """
        pass


class MidiEvents:
    """Abstract class indicate that object has some notion of midi event,
    and it can return collection of midi events.
    """
    @abstractmethod
    def get_midi_events(self) -> List[MidiEvent]:
        """ Abstract method that should be implemented by subclasses.
        Returns a list of MIDI events.
        :return:
        """
        pass
