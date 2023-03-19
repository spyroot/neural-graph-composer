from abc import abstractmethod, ABC
from neural_graph_composer.midi.midi_sequence import MidiNoteSequence


class MidiBaseReader(ABC):
    """Base class for MIDI file readers.
    This class defines the interface for reading MIDI files.
    Subclasses should implement the read method to actually read
    the file and return a `MidiNoteSequence` object.
    """
    @staticmethod
    @abstractmethod
    def read(file_path: str) -> MidiNoteSequence:
        raise NotImplementedError("subclasses must implement the read method")

