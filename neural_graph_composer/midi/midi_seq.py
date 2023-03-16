
class MidiSeq:
    """This just interface for all MIDI subclasses to keep
     internal seq.
    """
    def __init__(self) -> None:
        self.seq = 0
