"""
This class represent midi key signature
Author Mus spyroot@gmail.com
"""

from enum import auto, Enum
from typing import Optional


class KeySignatureType(Enum):
    MAJOR = auto()
    MINOR = auto()


class MidiKeySignature:
    def __init__(
            self,
            midi_time: Optional[float] = 0.0,
            midi_key: Optional[int] = 0,
            midi_mode: Optional[int] = 0,
    ) -> None:
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
        self.midi_time = midi_time
        self.midi_key = midi_key
        self.mode = self.major_or_minor(midi_mode)

    @staticmethod
    def major_or_minor(midi_mode: int):
        if midi_mode == 0:
            return KeySignatureType.MAJOR
        elif midi_mode == 1:
            return KeySignatureType.MINOR
        else:
            raise ValueError(f"Invalid midi_mode {midi_mode}")

    def __lt__(self, other):
        return self.midi_time < other.midi_time
