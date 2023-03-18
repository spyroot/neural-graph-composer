"""
This class just interface so we seq each MIDI.

Author Mus spyroot@gmail.com
           mbayramo@stanford.edu
"""


class MidiSeq:
    """This just interface for all MIDI subclasses to keep
     internal seq.
    """
    def __init__(self) -> None:
        self.seq = 0
