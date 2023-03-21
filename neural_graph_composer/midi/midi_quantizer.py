"""
Represent quantization
Author Mus spyroot@gmail.com
"""
from abc import abstractmethod
from typing import Optional


class MidiQuantization:
    """
    """
    @abstractmethod
    def is_quantized(self) -> bool:
        """Determines whether all notes in the sequence are quantized.
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def quantize(self, sps: int = 4, amount: Optional[float] = 0.5) -> None:
        """In place quantize midi sequence, i.e.  Quantizes the start and end
         times of each note based on algo , grid, the nearest step etc.
         multiple based on resolution.
        :param sps:
        :param amount:
        :return:
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def last_quantized_steps(self) -> int:
        """It should return last quantized
        :return:
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def total_quantized_steps(self) -> int:
        """It should return total number of steps quantized
        :return:
        """
        raise NotImplementedError

