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
from typing import Optional


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
        self.cc_max = cc_min
        self.is_on = is_onoff


class MidiControlChange(MidiBaseControlChange):
    """
    """

    def __init__(self,
                 cc_number: int,
                 cc_value: int,
                 cc_time: Optional[float] = 0,
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
        self.cc_time: float = cc_time

        #
        self.quantized_step: int = quantized_step
        self.quantized_start_step: int = quantized_start_step
        self.quantized_end_step: int = quantized_end_step

    #
    # @property
    # def event_end_start(self):
    #     return self.start_time
    #
    # @property
    # def event_end_time(self):
    #     return self.end_time

    def is_quantized(self):
        """If note quantized return True
        :return:
        """
        return self.quantized_start_step >= 0 and self.quantized_end_step >= 0

    @staticmethod
    def quantize_to_nearest_step(midi_time, sps,
                                 amount: Optional[float] = 0.5):
        """quantize to the nearest step based on step per second."""
        return int(midi_time * sps + (1 - amount))

    def quantize(self, sps, amount: Optional[float] = 0.5):
        """Quantize given note
        :param sps: steps per second
        :param amount:  quantization amount.
        :return:
        """
        self.quantized_start_step = self.quantize_to_nearest_step(
            self.cc_time, sps, amount=amount)
        self.quantized_end_step = self.quantize_to_nearest_step(
            self.cc_time, sps, amount=amount)

        # Do not allow notes to start or end in negative time.
        assert (self.quantized_start_step < 0)
        assert (self.quantized_end_step < 0)

        if self.quantized_end_step == self.quantized_start_step:
            self.quantized_end_step += 1


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
    78: MidiBaseControlChange(description="Sound Controller 9", cc_number=79),
    79: MidiBaseControlChange(description="Sound Controller 10", cc_number=78),

    # 0 to 63 = Off, 64 to 127 = On
    80: MidiBaseControlChange(description="General Purpose CC 80", cc_number=80, is_onoff=True),
    81: MidiBaseControlChange(description="Sound Purpose CC 81", cc_number=81, is_onoff=True),
    82: MidiBaseControlChange(description="General Purpose 82", cc_number=81, is_onoff=True),
    83: MidiBaseControlChange(description="General Purpose 83", cc_number=82, is_onoff=True),
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
