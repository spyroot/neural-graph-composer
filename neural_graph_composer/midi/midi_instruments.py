"""
Midi instrument representation.

@TODO split notion instrument ID and PRogram.

Author Mus spyroot@gmail.com
"""

from typing import Optional, Any

# Note: While GM does not define the actual characteristics of any sounds,
# the names in parentheses after each of the synth leads,
# pads, and sound effects are,
# in particular, intended only as guides.
MidiInstrumentMap = frozenset({
    range(1, 9): "Piano",
    range(9, 17): "Chromatic Percussion",
    range(17, 25): "Organ",
    range(25, 33): "Guitar",
    range(33, 41): "Bass",
    range(41, 49): "Strings",
    range(49, 57): "Ensemble",
    range(57, 65): "Brass",
    range(65, 73): "Reed",
    range(73, 81): "Pipe",
    range(81, 89): "Synth Lead",
    range(89, 97): "Synth Pad",
    range(97, 105): "Synth Effects",
    range(105, 113): "Ethnic",
    range(113, 121): "Percussive",
    range(121, 129): "Sound Effects",
}.items())


class MidiInstrumentInfo:
    """
    Attributes:
        instrument_num (int): The MIDI program number of the instrument.
        name (str): The name of the instrument.
        is_drum (bool): True if the instrument is a drum instrument.
    """
    __slots__ = ("instrument_num", "name", "is_drum")

    def __init__(self,
                 instrument: int,
                 name: Optional[str] = "",
                 is_drum: Optional[bool] = False) -> None:
        """Create a `MidiInstrumentInfo` object.
        :param instrument: id. (int): The MIDI program number of the instrument.
        :param name: name (Optional[str], optional): The name of the instrument. Defaults to "".
        :param is_drum (Optional[bool], optional): True if the instrument is a drum instrument. Defaults to False
        """
        if not isinstance(instrument, int):
            raise TypeError("Instrument must be an integer.")
        if not isinstance(name, str):
            raise TypeError("Name must be a string.")
        if not isinstance(is_drum, bool):
            raise TypeError("is_drum must be a boolean.")
        if not 0 <= instrument <= 255:
            raise ValueError("Instrument id must be in the range [0, 255].")

        self.instrument_num: int = min(max(0, instrument), 255)
        if name is None or name == "":
            for instrument_range, instrument_name in MidiInstrumentMap:
                if instrument in instrument_range:
                    name = instrument_name
                    break
            else:
                name = "Unknown"

        self.name: str = name
        self.is_drum = is_drum

    def __eq__(self, other: Any) -> bool:
        """Compares two `MidiInstrumentInfo` objects for equality.
        :param other: The object to compare with.
        :return: True if the objects are equal, False otherwise.
        """
        if not isinstance(other, MidiInstrumentInfo):
            return False
        return (self.instrument_num == other.instrument_num
                and self.name == other.name
                and self.is_drum == other.is_drum)

    def __repr__(self) -> str:
        """Returns a string representation of the `MidiInstrumentInfo` object.
        :return: The string representation of the object.
        """
        return f"MidiInstrumentInfo({self.instrument_num}, '{self.name}', {self.is_drum})"

    def __str__(self) -> str:
        """Returns a string representation of the object.
        :return:  A string representation of the object.
        """
        return f"MidiInstrumentInfo({self.instrument_num}, {self.name}, {self.is_drum})"

    def __ne__(self, other: Any) -> bool:
        """Checks if two MidiInstrumentInfo objects are not equal.
        :param other: The object to compare with.
        :return: True if the objects are not equal, False otherwise.
        """
        return not self.__eq__(other)

    def __lt__(self, other: 'MidiInstrumentInfo') -> bool:
        """
        :param other:
        :return:
        """
        return self.instrument_num < other.instrument_num

    def __hash__(self) -> int:
        """Return the hash value of the MidiInstrumentInfo instance, which is
        based on the hash of the instrument number.
        :return: hash value of the instance
        """
        return hash((self.instrument_num, self.name, self.is_drum))

    @classmethod
    def from_dict(cls, data: dict) -> 'MidiInstrumentInfo':
        """Create a `MidiInstrumentInfo` object from a dictionary."""
        instrument_num = data.get('instrument_num')
        name = data.get('name', '')
        is_drum = data.get('is_drum', False)
        return cls(instrument_num, name, is_drum)

    def to_dict(self) -> dict:
        """Return a dictionary representation of the MidiInstrumentInfo instance."""
        return {
            "instrument_num": self.instrument_num,
            "name": self.name,
            "is_drum": self.is_drum,
        }
