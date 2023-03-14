from typing import Optional

# Note: While GM does not define the actual characteristics of any sounds,
# the names in parentheses after each of the synth leads, pads, and sound effects are,
# in particular, intended only as guides.
MidiInstrumentMap = {
    1 - 8: "Piano",
    9 - 16: "Chromatic Percussion",
    17 - 24: "Organ",
    25 - 32: "Guitar",
    33 - 40: "Bass",
    41 - 48: "Strings",
    49 - 56: "Ensemble",
    57 - 64: "Brass",
    65 - 72: "Reed",
    73 - 80: "Pipe",
    81 - 88: "Synth Lead",
    89 - 96: "Synth Pad",
    97 - 104: "Synth Effects",
    105 - 112: "Ethnic",
    113 - 120: "Percussive",
    121 - 128: "Sound Effects",
}


class MidiInstrumentInfo:
    """
    Hold midi instrument information.

    Note usually any drum are on MIDI 10 , i.e 9 if we count from 0.
    """
    def __init__(self,
                 instrument: int,
                 name: Optional[str] = "",
                 is_drum: Optional[bool] = False) -> None:
        """A midi instrument
        :param instrument: id.
        :param name: name of instrument
        :param if instrument is drum instrument.
        """
        self.instrument_num: int = min(max(0, instrument), 255)
        self.name: str = name
        self.is_drum = is_drum
