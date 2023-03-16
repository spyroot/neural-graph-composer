"""
Midi cc message

MIDI CC, or Continuous Controller, is a type of MIDI message that is used to control
various parameters on a MIDI device, For example MIDI surface generate MIDI CC.

CC values have a range from 0-127, with 0 being the minimum value and
127 being the maximum value.

Other CC message control on and off,
Where value 0 to 63 = Off and 64 to 127 = On.

Author Mus spyroot@gmail.com
"""
import math
import types
from typing import Optional, Any


class MidiBaseControlChange:
    """
    """

    def __init__(
            self,
            cc_number: int,
            description: Optional[str] = "Undefined",
            cc_min: Optional[int] = 0,
            cc_max: Optional[int] = 127,
            is_onoff: Optional[bool] = False) -> None:
        """
        Base class for MIDI, Re-present cc number - description and min and max.

        :param cc_number:
        :param description:
        :param cc_min:
        :param cc_max:
        :param is_onoff:
        """
        self.cc_number: int = cc_number

        if description == "Undefined":
            self.description: str = f"{description} {cc_number}"
        else:
            self.description = description

        self.cc_max = cc_max
        self.cc_min = cc_min
        self.is_onoff: bool = is_onoff

    def __lt__(self, other):
        return self.cc_number < other.cc_number

    def __ne__(self, other):
        return self.cc_number != other.cc_number

    def __repr__(self) -> str:
        return f"MidiBaseControlChange(cc_number={self.cc_number}, description='{self.description}', " \
               f"cc_min={self.cc_min}, cc_max={self.cc_max}, is_onoff={self.is_onoff})"

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, MidiBaseControlChange):
            return self.cc_number == other.cc_number and self.description == other.description
        return NotImplemented


class MidiControlChange(MidiBaseControlChange):
    """
    """
    # MIDI to CC NAME is frozen.
    MIDI_CC_NAME_TO_CC = {
        0: MidiBaseControlChange(description="Bank", cc_number=0),
        1: MidiBaseControlChange(description="Modulation", cc_number=1),
        2: MidiBaseControlChange(description="Breath Controller", cc_number=2),
        3: MidiBaseControlChange(description="Undefined", cc_number=3),
        4: MidiBaseControlChange(description="Foot Controller", cc_number=4),
        5: MidiBaseControlChange(description="Portamento Time", cc_number=5),
        6: MidiBaseControlChange(description="Data Entry Most Significant Bit(MSB)", cc_number=6),
        7: MidiBaseControlChange(description="Volume", cc_number=7),
        8: MidiBaseControlChange(description="Balance", cc_number=8),
        9: MidiBaseControlChange(cc_number=9),
        10: MidiBaseControlChange(description="Pan", cc_number=10),
        11: MidiBaseControlChange(description="Expression", cc_number=11),
        12: MidiBaseControlChange(description="Effect Controller 1", cc_number=12),
        13: MidiBaseControlChange(description="Effect Controller 2", cc_number=13),
        14: MidiBaseControlChange(cc_number=14),
        15: MidiBaseControlChange(cc_number=15),
        16: MidiBaseControlChange(description="General Purpose:", cc_number=16),
        17: MidiBaseControlChange(description="Controller 0-31 Least Significant Bit (LSB)", cc_number=17),
        18: MidiBaseControlChange(cc_number=18),

        64: MidiBaseControlChange(description="Damper Pedal", cc_number=64, is_onoff=True),
        65: MidiBaseControlChange(description="Portamento On/Off Switch", cc_number=65, is_onoff=True),
        66: MidiBaseControlChange(description="Sostenuto On/Off Switch", cc_number=66, is_onoff=True),
        67: MidiBaseControlChange(description="Soft Pedal On/Off Switch", cc_number=67, is_onoff=True),
        68: MidiBaseControlChange(description="Legato FootSwitch", cc_number=68),

        # Another way to “hold notes” (see MIDI CC 64 and MIDI CC 66).
        # However, notes fade out according to their release parameter
        # rather than when the pedal is released.
        69: MidiBaseControlChange(description="Hold 2", cc_number=69, is_onoff=True),

        # controller Example shaping the Voltage Controlled Filter (VCF).
        # Default = Resonance - also(Timbre or Harmonics)
        70: MidiBaseControlChange(description="Sound Controller 1", cc_number=70),
        71: MidiBaseControlChange(description="Sound Controller 2", cc_number=71),
        72: MidiBaseControlChange(description="Sound Controller 3", cc_number=72),
        73: MidiBaseControlChange(description="Sound Controller 4", cc_number=73),
        74: MidiBaseControlChange(description="Sound Controller 5", cc_number=74),
        75: MidiBaseControlChange(description="Sound Controller 6", cc_number=75),
        76: MidiBaseControlChange(description="Sound Controller 7", cc_number=76),
        77: MidiBaseControlChange(description="Sound Controller 8", cc_number=77),
        78: MidiBaseControlChange(description="Sound Controller 9", cc_number=78),
        79: MidiBaseControlChange(description="Sound Controller 10", cc_number=79),

        # 0 to 63 = Off, 64 to 127 = On
        80: MidiBaseControlChange(description="General Purpose CC 80", cc_number=80, is_onoff=True),
        81: MidiBaseControlChange(description="Sound Purpose CC 81", cc_number=81, is_onoff=True),
        82: MidiBaseControlChange(description="General Purpose 82", cc_number=82, is_onoff=True),
        83: MidiBaseControlChange(description="General Purpose 83", cc_number=83, is_onoff=True),
        84: MidiBaseControlChange(description="Portamento CC Control", cc_number=84),
        # undefined
        85: MidiBaseControlChange(cc_number=85),
        86: MidiBaseControlChange(cc_number=86),
        87: MidiBaseControlChange(cc_number=87),

        91: MidiBaseControlChange(description="Effect 1 Depth", cc_number=91),
        92: MidiBaseControlChange(description="Effect 2 Depth", cc_number=92),
        93: MidiBaseControlChange(description="Effect 3 Depth", cc_number=93),
        94: MidiBaseControlChange(description="Effect 4 Depth", cc_number=94),
        95: MidiBaseControlChange(description="Effect 5 Depth", cc_number=95),

        96: MidiBaseControlChange(description="Data Increment", cc_number=96),
        97: MidiBaseControlChange(description="Data Decrement", cc_number=97),
        98: MidiBaseControlChange(description="NRPN 98", cc_number=98),
        99: MidiBaseControlChange(description="NRPN 99", cc_number=99),
        100: MidiBaseControlChange(description="RPN 100", cc_number=100),
        101: MidiBaseControlChange(description="RPN 101", cc_number=101),
        # 102 - 109 Undefined populated in the loop
        # 120 - 127 Channel Mode Messages in the loop
        # Mutes all sounding notes. It does so regardless of release time or sustain.
        120: MidiBaseControlChange(description="All Sound Off", cc_number=120),
        #
        121: MidiBaseControlChange(description="Reset All Controllers", cc_number=121),
        #
        122: MidiBaseControlChange(description="Local On/Off Switch", cc_number=122),
        # Mutes all sounding notes.
        # Release time will still be maintained, and notes held by
        # sustain will not turn off until sustain pedal is depressed.
        123: MidiBaseControlChange(description="All Notes Off", cc_number=123),
        # omni
        124: MidiBaseControlChange(description="Omni Mode Off", cc_number=124),
        125: MidiBaseControlChange(description="Omni Mode On", cc_number=125),
        # Sets device mode to Monophonic. [Channel Mode Message]
        # Mono Mode On (+ poly off, + all notes off).
        # This equals the number of channels, or zero if the number of
        # channels equals the number of voices in the receiver.
        126: MidiBaseControlChange(description="Mono Mode", cc_number=126),
        # Sets device mode to Polyphonic. [Channel Mode Message]
        # Poly Mode On (+ mono off, +all notes off).
        127: MidiBaseControlChange(description="Poly Mode", cc_number=127),
    }

    # add from 31 to 64 controller CC
    MIDI_CC_NAME_TO_CC.update(
        {cc_num: MidiBaseControlChange(
            description=f"Controller {cc_num}",
            cc_number=cc_num) for cc_num in range(31, 64)})

    # Undefined
    MIDI_CC_NAME_TO_CC.update(
        {cc_num: MidiBaseControlChange(cc_number=cc_num) for cc_num in range(102, 120)})

    # Channel msg
    MIDI_CC_NAME_TO_CC.update(
        {cc_num: MidiBaseControlChange(
            description="Channel Mode Messages {cc_num}",
            cc_number=cc_num) for cc_num in range(120, 128)})

    MIDI_CC_NAME_TO_CC = types.MappingProxyType(MIDI_CC_NAME_TO_CC)

    def __init__(self,
                 cc_number: int,
                 cc_value: int,
                 cc_time: Optional[float] = 0.0,
                 program: Optional[int] = 0,
                 instrument: Optional[int] = 0,
                 is_drum: Optional[bool] = False,
                 quantized_step: Optional[int] = 4,
                 quantized_start_step: Optional[int] = -1,
                 quantized_end_step: Optional[int] = -1,
                 ) -> None:
        """
        :param cc_number: MIDI CC number
        :param cc_value:  MID CC value , amount or on/off
        :param cc_time:  MIDI cc time
        :param program: MIDI program
        :param instrument: MIDI instrument
        :param is_drum:  if MIDI CC for a drum channel
        :param quantized_step:
        """
        super().__init__(min(max(0, cc_number), 255))
        self.cc_value: int = min(max(0, cc_value), 255)
        self.program: int = min(max(0, program), 255)
        self.instrument: int = min(max(0, instrument), 255)
        self.is_drum: bool = is_drum
        # positive only
        self.cc_time: float = max(0.0, cc_time)

        # valid between 1 and 32
        self._quantized_step: int = min(max(1, quantized_step), 32)
        self._quantized_start_step: int = quantized_start_step
        self._quantized_end_step: int = quantized_end_step
        self._is_quantized = quantized_start_step >= 0 and quantized_end_step >= 0
        self.seq = 0

    def __lt__(self, other):
        """
        :param other:
        :return:
        """
        if math.isclose(self.cc_time, other.cc_time):
            return self.seq < other.seq

        return self.cc_time < other.cc_time

    def __eq__(self, other):
        """
        :param other:
        :return:
        """
        if not isinstance(other, MidiControlChange):
            return NotImplemented

        return (self.cc_number == other.cc_number and
                self.cc_value == other.cc_value and
                self.cc_time == other.cc_time and
                self.program == other.program and
                self.instrument == other.instrument and
                self.is_drum == other.is_drum and
                self.quantized_step == other.quantized_step and
                self.quantized_start_step == other.quantized_start_step and
                self.quantized_end_step == other.quantized_end_step)

    def __ne__(self, other):
        """
        :param other:
        :return:
        """
        if not isinstance(other, MidiControlChange):
            return NotImplemented

        return not self.__eq__(other)

    @property
    def quantized_step(self) -> int:
        """Return the current quantization step of the note.
        :return: int
        """
        return self._quantized_step

    @quantized_step.setter
    def quantized_step(self, value: int) -> None:
        """Sets quantization step and update state of MIDI CC that it quantized.
        :param value: int, the new quantized step to set.
        :return: None
        :raise ValueError if value is not a positive integer between 1 and 32.
        """
        if value < 1 or value > 32:
            raise ValueError("Quantized step must be between 1 and 32")
        else:
            self._quantized_step = value

    @property
    def quantized_start_step(self) -> Optional[int]:
        """Return the current quantized start step of the note.
        Returns None if the note has not been quantized yet.
        :return: int or None
        """
        return self._quantized_start_step

    @quantized_start_step.setter
    def quantized_start_step(self, value) -> None:
        """Sets quantized_start_step and update
           state of MIDI note that it quantized.

        :param value: int, the new quantized start step to set.
        :return: None
        :raise ValueError if value is less than 0 or quantized start step is >= quantized end step.
        """
        if value < 0:
            raise ValueError("Quantized start step cannot be negative.")
        elif self._quantized_end_step is not None and value >= self._quantized_end_step:
            raise ValueError("Quantized start step must be less than quantized end step.")
        else:
            self._quantized_start_step = value
            self._is_quantized = True

    @property
    def quantized_end_step(self) -> Optional[int]:
        """ Returns the quantized end step of the note
        in the MIDI grid. If the note has not been quantized yet, returns None.
        :return:
        """
        return self._quantized_end_step

    @quantized_end_step.setter
    def quantized_end_step(self, value) -> None:
        """Return the current quantized end step of the note.
           Returns None if the note has not been quantized yet.
        :param value:
        :return:
        :raise ValueError if value is less than 0 or quantized start step is >= quantized end step.
        """
        if value < 0:
            raise ValueError("Quantized end step cannot be negative.")
        elif self._quantized_start_step is not None and value <= self._quantized_start_step:
            raise ValueError("Quantized end step must be greater than quantized start step.")
        else:
            self._quantized_end_step = value
            self._is_quantized = True

    def is_quantized(self) -> bool:
        """If control change quantized return True
        :return:
        """
        return self._is_quantized

    @staticmethod
    def quantize_to_nearest_step(midi_time, sps,
                                 amount: Optional[float] = 0.5):
        """quantize to the nearest step based on step per second."""
        return int(midi_time * sps + (1 - amount))

    def quantize_in_place(self, sps: int = 4, amount: float = 0.5, min_step: Optional[int] = None):
        """Quantize given note
        :param sps: steps per second
        :param amount:  quantization amount.
        :return:
        """
        self.quantized_start_step = self.quantize_to_nearest_step(
            self.cc_time, sps, amount=amount)
        self.quantized_end_step = self.quantize_to_nearest_step(
            self.cc_time, sps, amount=amount)

        self.quantized_start_step = max(self.quantized_start_step, 0)
        assert (self.quantized_start_step >= 0)
        assert (self.quantized_end_step >= 0)

        if self.quantized_end_step == self.quantized_start_step:
            self.quantized_end_step += 1

        self.quantized_step = sps

    def quantize(self, sps: int = 4, amount: float = 0.5, min_step: Optional[int] = None):
        """Returns a new quantized control change with the specified
        number of steps per second (sps), quantization amount,
        and minimum step size.

        :param sps: the number of steps per second to use for quantization (default is 4).
        :param amount: float, the quantization amount (default is 0.5).
        :param min_step: the minimum step size in samples (optional). If provided,
                     the note will not be quantized to steps smaller than this
                     value. This can be used to prevent excessively small
                     durations that may not be supported by the destination
                     MIDI device or file format.
        :return: new quantized control change
        """
        quantized_cc = self.make_copy()
        quantized_cc.quantize_in_place(sps=sps, amount=amount, min_step=min_step)
        return quantized_cc

    def make_copy(self,
                  cc_time: Optional[float] = None,
                  quantized_start_step: Optional[int] = None,
                  quantized_end_step: Optional[int] = None):
        """Returns a new copy of this cc

        :param cc_time: Optional[int], if set will overwrite cc_time with new cc_time for the copied note.
                                     Defaults to `None`, which uses the current cc_time.
        :param quantized_start_step: Optional[int], if set will overwrite new quantized start step for the copied note.
                                     Defaults to `None`, which uses the current quantized start step.
        :param quantized_end_step: Optional[int], the new quantized end step for the copied note.
                                   Defaults to `None`, which uses the current quantized end step.
        :return: a new instance of the `MidiNote` class that is a copy of the current instance.
        """
        return MidiControlChange(
            self.cc_number,
            cc_value=self.cc_value,
            cc_time=cc_time
            if cc_time is not None else self.cc_time,
            program=self.program,
            instrument=self.instrument,
            quantized_start_step=quantized_start_step
            if quantized_start_step is not None else self.quantized_start_step,
            quantized_end_step=quantized_end_step
            if quantized_end_step is not None else self.quantized_end_step,
        )

    def shift_time(self, time: float) -> None:
        """Shift time
        :param time: float time , float time we need a note.
        :return:
        """
        self.cc_time += time
