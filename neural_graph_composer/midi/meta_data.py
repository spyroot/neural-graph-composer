"""
Midi meta-data information.
For now we skip this

Author Mus spyroot@gmail.com
"""


class SequenceMetadata:
    """Information from MIDI.
    """
    artist: str
    title: str
    genre: []
    composers: []


class Section:
    section_id: int


class SectionAnnotation:
    section_id: int
    time: float


class SectionGroup:
    num_times: int


class SubsequenceInfo:
    end_time_offset: float
    start_time_offset: float
