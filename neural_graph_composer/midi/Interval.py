"""
This class represent interval that mainly used for Sorted Dict.
The idea sort all MIDI information in sorted order where each
MIDI event can overlap.   So we can find the right order
in O(1) time.

Again it on going work.

Author Mus spyroot@gmail.com
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Interval:
    """Class representing a closed interval [start, end]."""

    __slots__ = "start", "end"
    start: int
    end: int

    # def __init__(self, start, end):
    #     self.start = start
    #     self.end = end

    def __repr__(self):
        return f"[{self.start}, {self.end}]"

    def __eq__(self, other):
        if isinstance(other, Interval):
            return self.start == other.start and self.end == other.end
        return False

    def __lt__(self, other):
        if isinstance(other, Interval):
            return self.start < other.start or (self.start == other.start and self.end < other.end)
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, Interval):
            return self < other or self == other
        return NotImplemented

    def __contains__(self, item):
        return self.start <= item <= self.end

    def __hash__(self):
        return hash((self.start, self.end))

    def intersection(self, other):
        if self.overlaps(other):
            new_min = max(self.start, other.start)
            new_max = min(self.end, other.end)
            return Interval(new_min, new_max) if new_min <= new_max else None
        return None

    def union(self, other):
        """Return a new Interval and it union
        :param other:
        :return:
        """
        start = min(self.start, other.start)
        end = max(self.end, other.end)
        return Interval(start, end)

    def overlaps(self, other):
        return max(self.start, other.start) <= min(self.end, other.end)
