"""
This class represent midi key signature
Author Mus spyroot@gmail.com
"""
import math
from enum import auto, Enum
from typing import Optional
from neural_graph_composer.midi.midi_seq import MidiSeq


class KeySignatureType(Enum):
    """
    """
    MAJOR = auto()
    MINOR = auto()


class MidiKeySignature(MidiSeq):
    """

    """
    def __init__(
            self, midi_time: Optional[float] = 0.0,
            midi_key: Optional[int] = 0,
            midi_mode: Optional[int] = 0) -> None:
        """MIDI Key signature
        https://www.recordingblogs.com/wiki/midi-key-signature-meta-message

        i.e. The second byte is the meta message type 0x59,
        which shows that this is the key signature meta message.

        The fourth byte has values between -7 and 7 and specifies the key signature
        in terms of number of flats (if negative) or sharps (if positive)

        The fifth and last byte of the message specifies the scale of the MIDI file,
        where if this byte is 0 the scale is major and if the byte is 1 the scale is minor.

         - It depends on the MIDI file type/format specified in the header chunk of the MIDI file.
         - If the file is type 0, then it contains only one MIDI track and the key signature occurs
           in that track and applies to that track. If the file is type 1, then it has several
           tracks that should be played together. Then the key signature would occur only in the
           first track and will apply to all tracks (global). If the file is type 2, then it
           contains several tracks that are separate songs. The key signature would
           be specific to the track in which it occurs.


        For now, we need track this in case key signature changed.

        :param midi_time:  MIDI tick, which specifies at what time something should be done
        :param midi_mode:
        :param midi_key:
        """
        super().__init__()
        self.midi_time = midi_time
        self.midi_key = midi_key
        self.mode = self.major_or_minor(midi_mode)
        self.seq = 0

    @staticmethod
    def major_or_minor(midi_mode: int):
        """

        :param midi_mode:
        :return:
        """
        if midi_mode == 0:
            return KeySignatureType.MAJOR
        elif midi_mode == 1:
            return KeySignatureType.MINOR
        else:
            raise ValueError(f"Invalid midi_mode {midi_mode}")

    def __lt__(self, other):
        """
        :param other:
        :return:
        """
        if math.isclose(self.midi_time, other.midi_time):
            return self.seq < other.seq

        return self.midi_time < other.midi_time

    def __str__(self):
        return f"MidiKeySignature(time={self.midi_time}, key={self.midi_key}, mode={self.mode.name})"

    def __repr__(self):
        return self.__str__()
