"""
This class focuses on music theory. Mapping a chord to scale, generating chords from a
scale, taking scale, and mapping to intervals. All dict and mapping are frozen and represent
constant. In order not to collide in the namespace, all are accessible via  MusicTheory.

The long term motivation here we can do chord inference, chord recognition
chord generation, chord correction etc.

Each value using an enum since is generally more efficient
because enums are singletons

Author Mus spyroot@gmail.com
           mbayramo@stanford.edu
"""
# P1: Perfect Unison
# m2: Minor Second
# M2: Major Second
# m3: Minor Third
# M3: Major Third
# P4: Perfect Fourth
# A4: Augmented Fourth / Diminished Fifth
# d5: Diminished Fifth (Same as Augmented Fourth)
# P5: Perfect Fifth
# A5: Augmented Fifth / Minor Sixth
# m6: Minor Sixth (Same as Augmented Fifth)
# M6: Major Sixth
# m7: Minor Seventh
# M7: Major Seventh
# P8: Perfect Octave
import itertools
import types
from enum import Enum
from typing import Tuple, List, Optional, FrozenSet, Union
from attr import frozen


class ScaleType(Enum):
    """Scale types.  We use enum as singletons"""
    MAJOR = "major"
    MINOR = "minor"
    DORIAN = "dorian"
    PHRYGIAN = "phrygian"
    LYDIAN = "lydian"
    MIXOLYDIAN = "mixolydian"
    AEOLIAN = "aeolian"
    LOCRIAN = "locrian"
    HARMONIC_MINOR = "harmonic_minor"
    MELODIC_MINOR = "melodic_minor"


class ChordType(Enum):
    """Chord type where it defines a name, this enum used as key, so we can
    get in O(1) time respected chord type
    """
    MAJOR = "major"
    MINOR = "minor"
    AUGMENTED = "augmented"
    DIMINISHED = "diminished"
    SUS2 = "sus2"
    SUS4 = "sus4"
    SEVENTH = "seventh"
    MAJOR_SEVENTH = "major_seventh"
    MINOR_SEVENTH = "minor_seventh"
    DOMINANT_SEVENTH = "dominant_seventh"
    AUGMENTED_SEVENTH = "augmented_seventh"
    DIMINISHED_SEVENTH = "diminished_seventh"
    HALF_DIMINISHED_SEVENTH = "half_diminished_seventh"
    MINOR_MAJOR_SEVENTH = "minor_major_seventh"
    ELEVENTH = "eleventh"
    MAJOR_ELEVENTH = "major_eleventh"
    MINOR_ELEVENTH = "minor_eleventh"
    THIRTEENTH = "thirteenth"
    MAJOR_THIRTEENTH = "major_thirteenth"
    MINOR_THIRTEENTH = "minor_thirteenth"
    DOMINANT_NINTH = "dominant_ninth"
    MAJOR_NINTH = "major_ninth"
    MINOR_NINTH = "minor_ninth"
    DOMINANT_SHARP_NINTH = "dominant_sharp_ninth"
    DOMINANT_FLAT_NINTH = "dominant_flat_ninth"
    DOMINANT_THIRTEENTH = "dominant_thirteenth"
    MAJOR_SIXTH = "major_sixth"
    MINOR_SIXTH = "minor_sixth"


class ChordIntervals(Enum):
    M = [4, 3]
    m = [3, 4]
    Seven = [4, 3, 3]
    MajorSeven = [4, 3, 4]
    MinorSeven = [3, 4, 3]
    HalfDiminishedSeven = [3, 3, 4]
    DiminishedSeven = [3, 3, 3]
    AugmentedSeven = [4, 3, 4]
    Sixth = [4, 3, 2]
    MinorSixth = [3, 4, 2]
    Ninth = [4, 3, 3, 4]
    MajorNinth = [4, 3, 4, 3]
    MinorNinth = [3, 4, 3, 4]
    Eleventh = [4, 3, 3, 4, 3]
    MajorEleventh = [4, 3, 4, 3, 3]
    MinorEleventh = [3, 4, 3, 4, 3]
    Thirteenth = [4, 3, 3, 4, 3, 4]
    MajorThirteenth = [4, 3, 4, 3, 3, 4]
    MinorThirteenth = [3, 4, 3, 4, 3, 4]
    AddNine = [4, 3, 7]
    SusTwo = [2, 5]
    SusFour = [5, 2]
    SevenSusFour = [5, 2, 3]


class ScaleInterval(Enum):
    """Interval enum representing the intervals,
      It defined as semitones from the root note.
      reference https://en.wikipedia.org/wiki/Interval_(music)
    """
    # P1: Perfect Unison
    P1 = 0
    # m2: Minor Second
    m2 = 1
    # M2: Major Second
    M2 = 2
    # m3: Minor Third
    m3 = 3
    # M3: Major Third
    M3 = 4
    # P4: Perfect Fourth
    P4 = 5
    # Tritone: Diminished Fifth / Augmented Fourth
    Tritone = 6
    # P5: Perfect Fifth
    P5 = 7
    # m6: Minor Sixth
    m6 = 8
    # M6: Major Sixth
    M6 = 9
    # m7: Minor Seventh
    m7 = 10
    # M7: Major Seventh
    M7 = 11
    # P8: Perfect Octave
    P8 = 12
    # A4: Augmented Fourth / Diminished Fifth
    A4 = 6
    # d5 diminished fifth
    d5 = 6





class Interval(Enum):
    """All Intervals
    """
    # Perfect unison
    P1 = 0
    # Minor second
    m2 = 1
    # 	Major second
    M2 = 2
    # 	Minor third
    m3 = 3
    # 	Major third
    M3 = 4
    P4 = 5
    Tritone = 6
    P5 = 7
    A5 = 8
    m6 = 8
    M6 = 9
    m7 = 10
    M7 = 11
    P8 = 12
    M9 = 14
    m9 = 13
    m10 = 15
    M10 = 16

    d12 = 11
    P11 = 17
    A11 = 18
    M11 = 17
    m11 = 15
    m13 = 20
    M13 = 21

    # M13 = 21
    # m13 = 22

    P13 = 19
    A13 = 20
    m14 = 21
    M14 = 22
    #
    A4 = 6
    # d5 diminished fifth
    d5 = 6
    d7 = 10

    def bases(self):
        """the base intervals that make up the compound interval.
        :return: A tuple of the base intervals.
        """
        value = self.value % 12
        if self == Interval.P1 or self == Interval.P8:
            return (self.value, )
        elif value in (1, 4, 5, 8, 11):
            return (Interval.M2.value, self.value - Interval.M2.value)
        elif value in (2, 3, 6, 7, 9, 10):
            return Interval.M2.value, Interval.M3.value, self.value - Interval.M2.value - Interval.M3.value
        elif value == 0:
            return tuple(0, 0)
        elif value == 11:
            return Interval.M2.value, self.value - Interval.M2.value


@frozen
class MusicTheory:
    """
    MusicTheory is a utility class that provides functionality related to music theory.
    It includes methods to map notes to pitches, generate scales and chords, and extract pitches
    for chords from their intervals. Map intervals to scale etc.  It a part of library
    mapping music to graph representation.

    """
    chord_mapping = {
        frozenset({0, 4, 7}): "maj",
        frozenset({0, 3, 7}): "min",
        frozenset({0, 4, 7, 11}): "maj7",
        frozenset({0, 3, 7, 10}): "m7",
        frozenset({0, 4, 7, 10}): "7",
        frozenset({0, 3, 6}): "dim",
        frozenset({0, 4, 8}): "aug",
        frozenset({0, 5, 7}): "sus4",
        frozenset({0, 2, 7}): "sus2",
        frozenset({0, 3, 6, 9}): "dim7",
        frozenset({0, 4, 7, 9}): "6",
        frozenset({0, 4, 7, 14}): "maj9",
        frozenset({0, 3, 7, 14}): "m9",
        frozenset({0, 4, 7, 10, 14}): "9",
        frozenset({0, 4, 7, 10, 14, 17}): "13",
    }

    # note to index mapping
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    # base intervals
    intervals = {
        Interval.P1: 0,
        Interval.m2: 1,
        Interval.M2: 2,
        Interval.m3: 3,
        Interval.M3: 4,
        Interval.P4: 5,
        Interval.A4: 6,
        Interval.d5: 6,
        Interval.P5: 7,
        Interval.A5: 8,
        Interval.m6: 8,
        Interval.M6: 9,
        Interval.m7: 10,
        Interval.M7: 11,
        Interval.P8: 12,
    }

    # each scale and it respected intervals
    scale_intervals = {
        ScaleType.MAJOR: frozenset(
            [Interval.P1, Interval.M2,
             Interval.M3, Interval.P4,
             Interval.P5, Interval.M6,
             Interval.M7]),
        ScaleType.MINOR: frozenset(
            [Interval.P1, Interval.M2,
             Interval.m3, Interval.P4,
             Interval.P5, Interval.m6,
             Interval.m7]),
        ScaleType.DORIAN: frozenset(
            [Interval.P1, Interval.M2,
             Interval.m3, Interval.P4,
             Interval.P5, Interval.M6,
             Interval.m7]),
        ScaleType.PHRYGIAN: frozenset(
            [Interval.P1, Interval.m2,
             Interval.m3, Interval.P4,
             Interval.P5, Interval.m6,
             Interval.m7]),
        ScaleType.LYDIAN: frozenset(
            [Interval.P1, Interval.M2,
             Interval.M3, Interval.A4,
             Interval.P5, Interval.M6,
             Interval.M7]),
        ScaleType.MIXOLYDIAN: frozenset(
            [Interval.P1, Interval.M2,
             Interval.M3, Interval.P4,
             Interval.P5, Interval.M6,
             Interval.m7]),
        ScaleType.AEOLIAN: frozenset(
            [Interval.P1, Interval.M2,
             Interval.m3, Interval.P4,
             Interval.P5, Interval.m6,
             Interval.m7]),
        ScaleType.LOCRIAN: frozenset(
            [Interval.P1, Interval.m2,
             Interval.m3, Interval.P4,
             Interval.d5, Interval.m6,
             Interval.m7]),
        ScaleType.HARMONIC_MINOR: frozenset(
            [Interval.P1, Interval.M2,
             Interval.m3, Interval.P4,
             Interval.P5, Interval.m6,
             Interval.M7]),
        ScaleType.MELODIC_MINOR: frozenset(
            [Interval.P1, Interval.M2,
             Interval.m3, Interval.P4,
             Interval.P5, Interval.M6,
             Interval.M7]),
    }

    # scale and intervals
    scales = {
        ScaleType.MAJOR: [0, 2, 4, 5, 7, 9, 11],
        ScaleType.MINOR: [0, 2, 3, 5, 7, 8, 10],
        ScaleType.DORIAN: [0, 2, 3, 5, 7, 9, 10],
        ScaleType.PHRYGIAN: [0, 1, 3, 5, 7, 8, 10],
        ScaleType.LYDIAN: [0, 2, 4, 6, 7, 9, 11],
        ScaleType.MIXOLYDIAN: [0, 2, 4, 5, 7, 9, 10],
        ScaleType.AEOLIAN: [0, 2, 3, 5, 7, 8, 10],
        ScaleType.LOCRIAN: [0, 1, 3, 5, 6, 8, 10],
        ScaleType.HARMONIC_MINOR: [0, 2, 3, 5, 7, 8, 11],
        ScaleType.MELODIC_MINOR: [0, 2, 3, 5, 7, 9, 11],
    }

    # rhythm signatures 4/4 etc
    rhythm_signatures = {
        "4/4": (4, 4),
        "3/4": (3, 4),
        "6/8": (6, 8),
        "5/4": (5, 4),
        "7/8": (7, 8),
        "12/8": (12, 8),
        "2/4": (2, 4),
        "2/2": (2, 2),
        "9/8": (9, 8),
        "11/8": (11, 8),
        "5/8": (5, 8),
    }

    # chords types
    chords = {
        ScaleType.MAJOR: ["M", "m", "m", "M", "M", "m", "dim"],
        ScaleType.MINOR: ["m", "dim", "M", "m", "m", "M", "M"],
        ScaleType.DORIAN: ["M", "m", "m", "M", "M", "dim", "m"],
        ScaleType.PHRYGIAN: ["m", "M", "dim", "m", "m", "M", "M"],
        ScaleType.LYDIAN: ["M", "M", "m", "dim", "M", "m", "m"],
        ScaleType.MIXOLYDIAN: ["M", "m", "dim", "m", "M", "m", "M"],
        ScaleType.AEOLIAN: ["m", "dim", "M", "m", "m", "M", "M"],
        ScaleType.LOCRIAN: ["dim", "M", "m", "m", "dim", "M", "m"],
        ScaleType.HARMONIC_MINOR: ["m", "dim", "aug", "m", "M", "M", "dim"],
        ScaleType.MELODIC_MINOR: ["m", "m", "aug", "M", "M", "dim", "dim"]
    }

    # map chord type to respected name.
    chord_name_mapping = {
        "M": "maj",
        "m": "min",
        "dim": "dim",
        "aug": "aug",
        "sus2": "sus2",
        "sus4": "sus4",
        "7": "7",
        "maj7": "maj7",
        "min7": "min7",
        "dom7": "dom7",
        "aug7": "aug7",
        "dim7": "dim7",
        "hdim7": "hdim7",
        "minmaj7": "minmaj7",
        "11": "11",
        "maj11": "maj11",
        "min11": "min11",
        "13": "13",
        "maj13": "maj13",
        "min13": "min13",
        "dom9": "dom9",
        "maj9": "maj9",
        "min9": "min9",
        "dom#9": "dom#9",
        "domb9": "domb9",
        "dom13": "dom13",
        "maj6": "maj6",
        "min6": "min6",
    }

    # dictionary store mapping from chord types
    # values are intervals
    chords_types = {
        ChordType.MAJOR: [0, 4, 7],
        ChordType.MINOR: [0, 3, 7],
        ChordType.AUGMENTED: [0, 4, 8],
        ChordType.DIMINISHED: [0, 3, 6],
        ChordType.SUS2: [0, 2, 7],
        ChordType.SUS4: [0, 5, 7],
        ChordType.SEVENTH: [0, 4, 7, 10],
        ChordType.MAJOR_SEVENTH: [0, 4, 7, 11],
        ChordType.MINOR_SEVENTH: [0, 3, 7, 10],
        ChordType.DOMINANT_SEVENTH: [0, 4, 7, 10],
        ChordType.AUGMENTED_SEVENTH: [0, 4, 8, 10],
        ChordType.DIMINISHED_SEVENTH: [0, 3, 6, 9],
        ChordType.HALF_DIMINISHED_SEVENTH: [0, 3, 6, 10],
        ChordType.MINOR_MAJOR_SEVENTH: [0, 3, 7, 11],
        ChordType.ELEVENTH: [0, 4, 7, 10, 14],
        ChordType.MAJOR_ELEVENTH: [0, 4, 7, 11, 14],
        ChordType.MINOR_ELEVENTH: [0, 3, 7, 10, 14],
        ChordType.THIRTEENTH: [0, 4, 7, 10, 14, 17],
        ChordType.MAJOR_THIRTEENTH: [0, 4, 7, 11, 14, 17],
        ChordType.MINOR_THIRTEENTH: [0, 3, 7, 10, 14, 17],
        ChordType.DOMINANT_NINTH: [0, 4, 7, 10, 14],
        ChordType.MAJOR_NINTH: [0, 4, 7, 11, 14],
        ChordType.MINOR_NINTH: [0, 3, 7, 10, 14],
        ChordType.DOMINANT_SHARP_NINTH: [0, 4, 7, 10, 15],
        ChordType.DOMINANT_FLAT_NINTH: [0, 4, 7, 10, 13],
        ChordType.DOMINANT_THIRTEENTH: [0, 4, 7, 10, 14, 21],
        ChordType.MAJOR_SIXTH: [0, 4, 7, 9],
        ChordType.MINOR_SIXTH: [0, 3, 7, 9]
    }

    pitch_names = {
        0: 'C',
        1: 'C#',
        2: 'D',
        3: 'D#',
        4: 'E',
        5: 'F',
        6: 'F#',
        7: 'G',
        8: 'G#',
        9: 'A',
        10: 'A#',
        11: 'B'
    }

    # Each chord is represented by a key in the dictionary,
    # and its corresponding intervals are stored as a list of integers.
    # each value is semitone intervals 4 and 3 etc.
    # we also store inverse chord_intervals_reverse
    chord_intervals = {
        "M": [4, 3],
        "m": [3, 4],
        "7": [4, 3, 3],
        "M7": [4, 3, 4],
        "m7": [3, 4, 3],
        "m7b5": [3, 3, 4],
        "dim7": [3, 3, 3],
        "aug7": [4, 3, 4],
        "6": [4, 3, 2],
        "m6": [3, 4, 2],
        "9": [4, 3, 3, 4],
        "M9": [4, 3, 4, 3],
        "m9": [3, 4, 3, 4],
        "11": [4, 3, 3, 4, 3],
        "M11": [4, 3, 4, 3, 3],
        "m11": [3, 4, 3, 4, 3],
        "13": [4, 3, 3, 4, 3, 4],
        "M13": [4, 3, 4, 3, 3, 4],
        "m13": [3, 4, 3, 4, 3, 4],
        "add9": [4, 3, 7],
        "sus2": [2, 5],
        "sus4": [5, 2],
        "7sus4": [5, 2, 3]
    }

    #  Intervals are the distances between any two notes
    #  this mapping
    #  Value of the intervals used in chord names and scales are the same.
    # the intervals are usually expressed in relation to the root note of the chord,
    # while in scales, they are expressed in relation to the tonic note of the scale.
    chord_intervals_mapping = {
        "M": [Interval.P1, Interval.M3, Interval.P5],
        "m": [Interval.P1, Interval.m3, Interval.P5],
        "7": [Interval.P1, Interval.M3, Interval.P5, Interval.m7],
        "M7": [Interval.P1, Interval.M3, Interval.P5, Interval.M7],
        "m7": [Interval.P1, Interval.m3, Interval.P5, Interval.m7],
        "m7b5": [Interval.P1, Interval.m3, Interval.d5, Interval.m7],
        "dim7": [Interval.P1, Interval.m3, Interval.d5, Interval.d7],
        "aug7": [Interval.P1, Interval.M3, Interval.A5, Interval.m7],
        "6": [Interval.P1, Interval.M3, Interval.P5, Interval.M6],
        "m6": [Interval.P1, Interval.m3, Interval.P5, Interval.M6],
        "9": [Interval.P1, Interval.M3, Interval.P5, Interval.m7, Interval.M9],
        "M9": [Interval.P1, Interval.M3, Interval.P5, Interval.M7, Interval.M9],
        "m9": [Interval.P1, Interval.m3, Interval.P5, Interval.m7, Interval.M9],
        "11": [Interval.P1, Interval.M3, Interval.P5, Interval.m7, Interval.M9, Interval.P11],
        "M11": [Interval.P1, Interval.M3, Interval.P5, Interval.M7, Interval.M9, Interval.P11],
        "m11": [Interval.P1, Interval.m3, Interval.P5, Interval.m7, Interval.M9, Interval.P11],
        "13": [Interval.P1, Interval.M3, Interval.P5, Interval.m7, Interval.M9, Interval.M13],
        "M13": [Interval.P1, Interval.M3, Interval.P5, Interval.M7, Interval.M9, Interval.M13],
        "m13": [Interval.P1, Interval.m3, Interval.P5, Interval.m7, Interval.M9, Interval.M13],
        "add9": [Interval.P1, Interval.M3, Interval.P5, Interval.M9],
        "sus2": [Interval.P1, Interval.M2, Interval.P5],
        "sus4": [Interval.P1, Interval.P4, Interval.P5],
        "7sus4": [Interval.P1, Interval.P4, Interval.P5, Interval.m7],
    }

    # m7(b5),half-diminished chords
    chord_name_mapping_extended = {
        "maj_seventh": "maj7",
        "min_seventh": "m7",
        "dominant_seventh": "7",
        "augmented_seventh": "aug7",
        "diminished_seventh": "dim7",
        "half_diminished_seventh": "m7(b5)",
        "minor_major_seventh": "m(maj7)",
        "eleventh": "11",
        "major_eleventh": "maj11",
        "minor_eleventh": "m11",
        "thirteenth": "13",
        "major_thirteenth": "maj13",
        "minor_thirteenth": "m13",
    }

    scales = types.MappingProxyType(scales)
    scale_intervals = types.MappingProxyType(scale_intervals)
    intervals = types.MappingProxyType(intervals)
    rhythm_signatures = types.MappingProxyType(rhythm_signatures)
    chord_mapping = types.MappingProxyType(chord_mapping)
    chords_types = types.MappingProxyType(chords_types)
    pitch_names = types.MappingProxyType(pitch_names)
    chord_intervals = types.MappingProxyType(chord_intervals)
    chord_name_mapping = types.MappingProxyType(chord_name_mapping)
    chord_name_mapping_extended = types.MappingProxyType(chord_name_mapping_extended)
    chord_intervals_reverse = {tuple(v): k for k, v in chord_intervals.items()}
    pitch_to_note_name = {v: k for k, v in pitch_names.items()}

    @staticmethod
    def intervals_from_scale(scale_name: Union[str, ScaleType]) -> frozenset:
        """Take a string that represent a scale major, minor,
        Return intervals ["P1", "M2", "M3", "P4", "P5", "M6", "M7"]

        Where:
            # P1: Perfect Unison
            # m2: Minor Second
            # M2: Major Second
            # m3: Minor Third
            # M3: Major Third
            # P4: Perfect Fourth
            # A4: Augmented Fourth / Diminished Fifth
            # d5: Diminished Fifth (Same as Augmented Fourth)
            # P5: Perfect Fifth
            # A5: Augmented Fifth / Minor Sixth
            # m6: Minor Sixth (Same as Augmented Fifth)
            # M6: Major Sixth
            # m7: Minor Seventh
            # M7: Major Seventh
            # P8: Perfect Octave

        :param scale_name: A string or ScaleType that represents a scale, e.g., "major", "minor"
        :return: A set of intervals for the specified scale

        Example:
        To get the intervals for a major scale:
        >>> MusicTheory.intervals_from_scale("major")
        >>> frozenset(["P1", "M2", "M3", "P4", "P5", "M6", "M7"])

        To get the intervals for a minor scale:
        >>> MusicTheory.intervals_from_scale("minor")
        >>> frozenset(["P1", "M2", "m3", "P4", "P5", "m6", "m7"])
        """
        valid_scales = ', '.join([s.value for s in MusicTheory.scale_intervals.keys()])

        if isinstance(scale_name, str):
            scale_name = ScaleType[scale_name.upper()]

        if scale_name not in MusicTheory.scale_intervals:
            raise ValueError(f"Invalid scale name: {scale_name}. "
                             f"Supported scales are: {valid_scales}")
        return MusicTheory.scale_intervals[scale_name]

    @staticmethod
    def chord_pitch_from_interval(
            root: Union[int, str],
            interval_chord: Union[
                List[Union[str, Interval]], FrozenSet[Union[str, Interval]]]) -> List[int]:
        """Given the root pitch and a set of chord intervals,
        returns a list of pitches for the corresponding chord.

        :param root: the root pitch of the chord (0-11, where 0 = C, 1 = C#, etc.)
        :param interval_chord: a list of chord intervals (e.g., ["P1", "M3", "P5"] for a major triad)
        :return: a list of pitches (integers 0-11) for the corresponding chord
        :raises ValueError: if the input `interval_chord` is not a list or frozenset or contains non-string elements
                            if the input `root` is not within the range [0, 11]
        Example:
        To get the pitches for a C major triad:
        >>> MusicTheory.chord_pitch_from_interval(0, ["P1", "M3", "P5"])
        >>> [0, 4, 7]

        To get the pitches for a C minor triad:
        >>> MusicTheory.chord_pitch_from_interval(0, ["P1", "m3", "P5"])
        >>>  [0, 3, 7]
        """
        if not isinstance(interval_chord, (list, frozenset)):
            raise TypeError(f"interval_chord parameter should be a list "
                            f"or frozenset. received {type(interval_chord)}")
        if not all(isinstance(i, str) for i in interval_chord) and \
                not all(isinstance(i, Interval) for i in interval_chord):
            raise TypeError("interval_chord should be a list or "
                            "frozenset of strings. received {type(interval_chord)}")

        if isinstance(root, str):
            if root.upper() not in MusicTheory.pitch_names.values():
                raise ValueError(f"Invalid note name: {root}")
            root = MusicTheory.pitch_to_note_name[root.upper()]
        elif not (0 <= root <= 11):
            raise ValueError("root pitch parameter should be within the range [0, 11]")

        _chord_pitches = []
        for i, interval in enumerate(interval_chord):
            # if caller pass string we convert to Enum
            if isinstance(interval, str):
                interval = Interval[interval.upper()]
            interval_pitch = MusicTheory.intervals[interval]
            chord_pitch = (root + interval_pitch) % 12
            _chord_pitches.append(chord_pitch)
        return _chord_pitches

    @staticmethod
    def get_chord_intervals(chord_type: str) -> List[int]:
        """Given a chord name, returns a list of interval values.

        Example:
         To get the intervals for a "m7b5" chord:
        ```
        MusicTheory.get_chord_intervals("m7b5)
        m7b5
        >>>[3, 3, 4]
        ```
        :param chord_type: A string representing the chord name, e.g., "m7b5"
        :return: A list of interval values for the specified chord [3, 3, 4]
        """
        return MusicTheory.chord_intervals[chord_type]

    @staticmethod
    def get_chord_name(interval_list: List[int]) -> Optional[str]:
        """Given a list of intervals, return the name of the chord that has those intervals,
        or None if no such chord exists.

        :param interval_list: a list of intervals, e.g., [3, 3, 4] for a half-diminished chord
        :return: the name of the chord that corresponds to the given intervals, or None if no such chord exists

        Example:
        To get the name of a chord with the intervals [3, 3, 4]:
        >>> MusicTheory.get_chord_name([3, 3, 4])
        >>> 'm7b5'
        """
        if not isinstance(interval_list, list):
            raise TypeError("interval_list must be a list of integers")
        if not all(isinstance(interval, int) for interval in interval_list):
            raise TypeError("interval_list must be a list of integers")

        for chord_name, intervals in MusicTheory.chord_intervals.items():
            if intervals == interval_list:
                return chord_name
        return None

    @staticmethod
    def get_chord_name_from_intervals(intervals: Union[Tuple[int, ...], List[int]]) -> Optional[str]:
        """Given a tuple of intervals, return the name of the chord that has those intervals,
        or None if no such chord exists.

        :param intervals: a tuple of integers representing the intervals of a chord, e.g., (3, 3, 4)
                          for a half-diminished chord
        :return: the name of the chord that corresponds to the given intervals, or None if no such chord exists

        Example:
        To get the name of a chord with the intervals (3, 3, 4):
        >>> MusicTheory.get_chord_name_from_intervals((3, 3, 4))
        >>> 'm7b5'
        """
        if not isinstance(intervals, (tuple, list)):
            raise TypeError("intervals must be a tuple or list of integers")
        if not all(isinstance(interval, int) for interval in intervals):
            raise TypeError("intervals must be a tuple or list of integers")

        chord_name = MusicTheory.chord_intervals_reverse.get(tuple(intervals))
        return chord_name

    @staticmethod
    def notes_from_scale(root: Union[int, str], scale: Union[str, ScaleType]) -> List[str]:
        """Given a root note and a scale name, returns a list of notes in that scale.

        :param root: the root note of the scale (e.g., "C")
        :param scale: the name of the scale (e.g., "major") or ScaleType enum
        :return: a list of notes in the scale

        Example:
        To get the notes in the C major scale:
        >>> MusicTheory.notes_from_scale("C", "major")
        >>> ["C", "D", "E", "F", "G", "A", "B"]
        """

        if isinstance(scale, str):
            scale = ScaleType(scale)

        if isinstance(root, str):
            if root.upper() not in MusicTheory.pitch_names.values():
                raise ValueError(f"Invalid note name: {root}")
        elif not (0 <= root <= 11):
            raise ValueError("root pitch parameter should be within the range [0, 11]")

        root_idx = MusicTheory.note_names.index(root)
        intervals = MusicTheory.intervals_from_scale(scale)
        sorted_intervals = sorted(intervals, key=lambda x: MusicTheory.intervals[x])

        notes = []
        for i, interval_name in enumerate(sorted_intervals):
            interval_semitones = MusicTheory.intervals[interval_name]
            notes.append(MusicTheory.note_names[(root_idx + interval_semitones) % 12])
        return notes

    @staticmethod
    def chord_from_scale(chord_name: str, scale_notes: List[str]):
        """Generate the notes of chord based on a given chord name.

        :param chord_name:  The name of the chord (e.g., 'CM', 'Dm', 'F7').
        :param scale_notes:  (List[str]): A list of scale notes
                             (e.g., ['C', 'D', 'E', 'F', 'G', 'A', 'B'] for C Major scale).

        Example:
        >>> chord_from_scale('CM', ['C', 'D', 'E', 'F', 'G', 'A', 'B'])
        ['C', 'E', 'G']

        :return:
        """
        root_note, triad_type = chord_name[:-1], chord_name[-1]
        root_index = MusicTheory.note_names.index(root_note)
        intervals = MusicTheory.chord_intervals_mapping.get(triad_type)
        assert intervals is not None
        chord_intervals = MusicTheory.chord_pitch_from_interval(root_index, intervals)

        if not chord_intervals:
            raise ValueError(f"Invalid triad type: {triad_type}")

        # chord_notes = [scale_notes[(root_index + interval) % len(scale_notes)] for interval in chord_intervals]
        chord_notes = [MusicTheory.note_names[interval] for interval in chord_intervals]
        return chord_notes

    @staticmethod
    def triads_from_scale(root: Union[int, str], scale: Union[str, ScaleType]) -> List[str]:
        """Given a root note and a scale name, returns a list of triads in that scale.

        :param root: the root note of the scale (e.g., "C")
        :param scale: the name of the scale (e.g., "major")
        :return: a list of triads in the scale

        Example:
        To get the triads in the C major scale:
        >>> MusicTheory.triads_from_scale("C", "major")
        >>> ["C", "Dm", "Em", "F", "G", "Am", "Bdim"]
        """
        if isinstance(scale, str):
            scale = ScaleType(scale)

        if isinstance(root, str):
            root = MusicTheory.pitch_to_note_name[root.upper()]
        if not (0 <= root <= 11):
            raise ValueError("root pitch parameter should be within the range [0, 11]")

        intervals = MusicTheory.intervals_from_scale(ScaleType(scale))
        sorted_intervals = sorted(intervals, key=lambda x: MusicTheory.intervals[x])

        notes = []
        for i, interval_name in enumerate(sorted_intervals):
            interval_semitones = MusicTheory.intervals[interval_name]
            notes.append(MusicTheory.note_names[(root + interval_semitones) % 12])

        triads = []
        for i, note in enumerate(notes):
            triad = [notes[i], notes[(i + 2) % len(notes)], notes[(i + 4) % len(notes)]]
            triads.append("".join(triad))

        return triads

    @staticmethod
    def triads_from_scale_tuple(root: Union[int, str], scale: Union[str, ScaleType]) -> List[Tuple[List[str], str]]:
        """Given a root note and a scale name, returns a list of
        triads in that scale with their chord types.

        :param root: the root note of the scale (e.g., "C")
        :param scale: the name of the scale (e.g., "major")
        :return: a list of tuples containing the notes of each triad and its chord type

        Example:
        To get the triads in the C major scale:
        >>> MusicTheory.triads_from_scale_tuple("C", "major")
        >>> [(['C', 'E', 'G'], 'maj'), (['D', 'F', 'A'], 'min'),
            (['E', 'G', 'B'], 'min'), (['F', 'A', 'C'], 'maj'), (['G', 'B', 'D'], 'maj'),
             (['A', 'C', 'E'], 'min'), (['B', 'D', 'F'], 'dim')]

        """
        if isinstance(scale, str):
            scale = ScaleType(scale)

        if isinstance(root, str):
            root = MusicTheory.pitch_to_note_name[root.upper()]
        elif not (0 <= root <= 11):
            raise ValueError("root pitch parameter should be within the range [0, 11]")

        if isinstance(root, int):
            root = MusicTheory.note_names[root]

        if root not in MusicTheory.note_names:
            raise ValueError(f"Invalid root note: '{root}'. "
                             f"Valid root notes are {MusicTheory.note_names}")

        if scale not in MusicTheory.scale_intervals:
            valid_scales = ', '.join(str(s) for s in MusicTheory.scale_intervals.keys())
            raise ValueError(f"Invalid scale name: '{scale}'. Valid scale names are {valid_scales}")

        root_idx = MusicTheory.note_names.index(root)
        intervals = MusicTheory.scale_intervals[scale]
        chord_types = MusicTheory.chords[scale]

        # sort so we have correct order
        sorted_intervals = sorted(intervals, key=lambda x: MusicTheory.intervals[x])
        scale_notes = [MusicTheory.note_names[(root_idx + MusicTheory.intervals[interval]) % 12]
                       for interval in sorted_intervals]

        triads = []
        for i, _ in enumerate(scale_notes):
            # generate triad
            notes = [scale_notes[i % 7], scale_notes[(i + 2) % 7], scale_notes[(i + 4) % 7]]
            chord_type = MusicTheory.chord_name_mapping[chord_types[i]]
            triads.append((notes, chord_type))

        return triads

    @staticmethod
    def chord_from_scale_name_extended(root_note: str, scale_name: Union[str, ScaleType]) -> List[str]:
        """Given the root note and scale name, return a list of chords that can be formed
        using the notes in that scale.

        :param root_note: a string representing the root note, e.g., "C"
        :param scale_name: a string representing the scale name, e.g., "major"
        :return: a list of chord names that can be formed using the notes in the specified scale

        Example:
        To get the chords that can be formed using the notes of the C major scale:
        >>> MusicTheory.chord_from_scale_name('C', 'major')
        >>> ['Cmaj7', 'Dm7', 'Em7', 'Fmaj7', 'G7', 'Am7', 'Bm7(b5)']
        """
        # TODO make we we parse root in format C,D but also extract,
        #  root note if root note given as DM Dm etc.

        # root_note, triad_type = root_note[:-1], root_note[-1]
        # root_index = MusicTheory.note_names.index(root_note)

        scale_notes = MusicTheory.notes_from_scale(root_note, scale_name)
        chords = []
        for chord_name, chord_intervals in MusicTheory.chord_intervals.items():
            chord_notes = [scale_notes[interval % len(scale_notes)] for interval in chord_intervals]
            if set(chord_notes).issubset(set(scale_notes)):
                chord = root_note + chord_name
                chords.append(chord)
        return chords

    @staticmethod
    def pitch_to_note(pitch: int) -> str:
        """Given a pitch value (an integer between 0 and 127 inclusive),
          returns the corresponding note name.
        :param pitch: the pitch value (an integer between 0 and 127 inclusive)
        :return: the note name (a string)
        """
        octave = pitch // 12
        pitch_class = pitch % 12
        note_name = MusicTheory.pitch_names[pitch_class]
        return f"{note_name}{octave}"

    # ------
    # @staticmethod
    # def chord_from_scale_name(root_note: str, scale_name: str) -> List[str]:
    #     """
    #     ['CM', 'Dm', 'Em', 'FM', 'GM', 'Am', 'Bdim', 'C6', 'Dm6', 'Em6', 'F6', 'G6', 'Am6',
    #     'Csus2', 'Dsus2', 'Esus2', 'Fsus2', 'Gsus2', 'Asus2', 'Bsus2', 'Csus4', 'Dsus4',
    #     'Esus4', 'Fsus4', 'Gsus4', 'Asus4', 'Bsus4', 'C7', 'Dm7', 'Em7', 'Fmaj7', 'G7', 'Am7', 'Bm7b5']
    #
    #     TODO test and finish it buggy
    #     :param root_note:
    #     :param scale_name:
    #     :return:
    #     """
    #     triads = MusicTheory.triads_from_scale_tuple(root_note, scale_name)
    #     chords = []
    #
    #     triad_type_to_prefix = {
    #         "M": ["M", ""],
    #         "m": ["m"],
    #         "dim": ["dim"],
    #         "aug": ["aug"],
    #     }
    #
    #     for triad_notes, triad_type in triads:
    #         for chord_name in MusicTheory.chord_name_mapping.keys():
    #             if MusicTheory.chord_name_mapping[chord_name] == triad_type:
    #                 print(f"Match {MusicTheory.chord_name_mapping[chord_name]} {triad_type}")
    #                 try:
    #                     chord_notes = MusicTheory.chord_from_scale(chord_name, scale_notes)
    #
    #                     print(f"{triad_notes}")
    #                     chords.append(chord_name)
    #                 except IndexError:
    #                     pass
    #     return chords

    @staticmethod
    def from_semitones(semitones: int) -> Interval:
        """Returns the Interval that represents the given number of semitones
          If no interval corresponds to the given number of semitones,
          raises a ValueError.

          My main source for all intervals
          https://en.wikipedia.org/wiki/Interval_(music)

        :param semitones: an int representing the number of semitones of the interval.
        :return:  Interval that represents the given number of semitones
        :raise ValueError: if no interval corresponds to the given number of semitones
        """
        for interval in Interval:
            if interval.value == semitones:
                return interval
        raise ValueError(f"No interval corresponds to {semitones} semitones.")

    @staticmethod
    def compound_interval_from_bases(a: Interval, b: Interval) -> Interval:
        """Compute the compound interval from two regular intervals.
        Main source for all intervals
        https://en.wikipedia.org/wiki/Interval_(music)
        :param a: Interval a that we use to compound
        :param b: Interval b that we use to compound.
        :return: Compound interval
        """
        semitones = a.value + b.value
        octaves = semitones // 12
        semitones -= octaves * 12
        return MusicTheory.from_semitones(octaves * 12 + semitones)

    @staticmethod
    def compound_interval_to_base(interval: int) -> Tuple[int, int]:
        """ Convert a compound interval to its base interval and number of octaves.
        :return: The compound interval tuple where tuple is size, octave
        :raise  ValueError: If the size is not between 0 and 11 (inclusive), or the octave is negative.
        """
        if interval < 0:
            raise ValueError("Interval must be non-negative.")
        octave = interval // 12
        size = interval % 12
        return size, octave

    @staticmethod
    def base_to_compound_interval(base_interval: int, octave: int) -> int:
        """Convert a base interval and number of octaves to a compound interval.
        :param octave: The number of octaves
        :param base_interval: base interval
        :raise  ValueError: If the size is not between 0 and 11 (inclusive), or the octave is negative.
        :return: The compound interval
        """
        if base_interval < 0 or base_interval > 11:
            raise ValueError("Interval size must be between 0 and 11 (inclusive).")
        if octave < 0:
            raise ValueError("Octave must be non-negative.")
        return octave * 12 + base_interval

    @staticmethod
    def bases(compound_interval: Interval) -> Tuple[int]:
        """
        Get the base intervals that make up the compound interval as a tuple of interval values.
        :param compound_interval: A compound interval.
        :return: A tuple of the base interval values.
        """
        value = compound_interval.value % 12
        if compound_interval == Interval.P1 or compound_interval == Interval.P8:
            return (compound_interval.value,)
        elif value in (1, 4, 5, 8, 11):
            return (Interval.M2.value, compound_interval.value - Interval.M2.value)
        elif value in (2, 3, 6, 7, 9, 10):
            return (
            Interval.M2.value, Interval.M3.value, compound_interval.value - Interval.M2.value - Interval.M3.value)
        elif value == 0:
            return tuple()

    @staticmethod
    def inverse_compound_interval(compound_interval: Interval) -> int:
        """
        The function takes a compound_interval of type Interval, and returns its inverse.
        It first calculates the sum of the values of the base intervals of the given
        compound interval, and then calculates the difference between the total value
        of the compound interval and the base intervals sum. It then constructs the
        inverse of the base intervals by subtracting each base interval value from 12,
        and appends the inverse of the difference to the list of inverse bases if the
        difference is not 0.

        Example:
            M2 = 2 semitones
            m7 = 10 semitones
            M2 + m7 = 2 + 10 = 12 semitones = P8

            So the inverse of a P8 should be a P1:
            P8 = 12 semitones
            12 - 12 = 0 semitones = P1

        Get the inverse of a given compound interval.
        :param compound_interval: A compound interval.
        :return: The inverse of the compound interval.
        """
        base_intervals_sum = sum(interval for interval in MusicTheory.bases(compound_interval))
        # delta
        difference = compound_interval.value - base_intervals_sum
        print(f"difference {difference}")
        # the inverse of the base intervals where 12 is last
        inverse_bases = [12 - val for val in MusicTheory.bases(compound_interval)]
        if difference != 0:
            inverse_bases.append(12 - difference)

        return sum(inverse_bases)


    @staticmethod
    def list_of_chords(root: int, interval: frozenset[str]) -> List[Tuple[str, List[int]]]:

        chord_pitches = MusicTheory.chord_pitch_from_interval(root, interval)
        print(f"chord_pitches pitch {chord_pitches}")
        chords = []
        for i in range(len(chord_pitches)):
            print(f"chord index {i}")
            # root note of the current chord
            root_note = chord_pitches[i]
            print(f"root_note note {root_note}")
            # chord_name = MusicTheory.pitch_names[chord_pitches[i]] + MusicTheory.chord_intervals[i]
            chord_interval = MusicTheory.chord_intervals[i]
            print(f"chord_interval {chord_interval}")
            chord_name = MusicTheory.pitch_names[(root_pitch + chord_pitches[i]) % 12] + MusicTheory.chord_intervals[i]
            print(f"chord_pitches pitch {chord_name}")

            chords.append(chord_name)

        return chords


def some_scale_mapping_demo():
    # scale_mixolydian = MusicTheory.notes_from_scale("F#", "mixolydian")
    # print(scale_mixolydian)
    extended = MusicTheory.chord_from_scale_name_extended('D', "minor")
    print(extended)
    #
    # result = MusicTheory.chord_from_scale_name_extended('Gm', 'harmonic_minor')
    # print(result)
    # assert result == ['GM', 'Gm', 'G7', 'GM7', 'Gm7', 'Gm7b5', 'Gdim7', 'Gaug7', 'G6', 'Gm6', 'G9', 'GM9', 'Gm9', 'G11', 'GM11', 'Gm11', 'G13', 'GM13', 'Gm13', 'Gadd9', 'Gsus2', 'Gsus4', 'G7sus4']

    expected_chords = ['F#M', 'G#m', 'A#dim', 'Bm', 'C#7', 'D#M', 'E#m7b5']
    chords = MusicTheory.chord_from_scale_name_extended('F#', 'mixolydian')
    print(f"F# {chords}")
    # assert chords == expected_chords, f"Expected {expected_chords}, but got {chords}"


def scale_triads_demo():
    """
    :return:
    """

    interval_major = MusicTheory.intervals_from_scale("major")
    interval_minor = MusicTheory.intervals_from_scale("minor")
    print(f"Minor intervals: minor {interval_major} major {interval_minor}")
    interval_major = MusicTheory.intervals_from_scale(ScaleType.MAJOR)
    interval_minor = MusicTheory.intervals_from_scale(ScaleType.MINOR)
    print(f"Minor intervals: minor {interval_major} major {interval_minor}")

    print(f"Chord name: {MusicTheory.get_chord_name([3, 3, 4])}")
    interval_m7b5 = MusicTheory.get_chord_intervals("m7b5")
    print(f"Chord name: {interval_m7b5}")

    # by int
    for root_pitch in range(0, 11):
        interval_list = ["P1", "M3", "P5"]
        chord_pitches = MusicTheory.chord_pitch_from_interval(root_pitch, interval_list)
        print(f"Chord pitch list for intervals root "
              f"{MusicTheory.pitch_names[root_pitch]}: {interval_list} -> {chord_pitches}")

    # by pitch name
    for n in MusicTheory.note_names:
        interval_list = ["P1", "M3", "P5"]
        chord_pitches = MusicTheory.chord_pitch_from_interval(n, interval_list)
        print(f"Chord pitch list for intervals root "
              f"{n}: {interval_list} -> {chord_pitches}")

    chord_name_from_interval = MusicTheory.get_chord_name_from_intervals((3, 3, 4))
    print(f"Chord name from the interval (3, 3, 4): {chord_name_from_interval}")
    chord_name_from_interval = MusicTheory.get_chord_name_from_intervals([3, 3, 4])
    print(f"Chord name from the interval (3, 3, 4): {chord_name_from_interval}")

    c_scale = MusicTheory.notes_from_scale('C', 'major')
    print(c_scale)

    c_minor = MusicTheory.notes_from_scale('C', 'minor')
    print(f"Note in C minor: {c_minor}")

    triads_c_major = MusicTheory.triads_from_scale("C", "major")
    print(f"Majos triads {triads_c_major}")
    triads_c_minor = MusicTheory.triads_from_scale("C", "major")
    print(f"Majos triads {triads_c_minor}")
    triads_major = MusicTheory.triads_from_scale_tuple("C", "major")
    print(f"Majos triads {triads_major}")
    triads_minor = MusicTheory.triads_from_scale_tuple("C", "minor")
    print(f"Majos triads {triads_major}")

    # get the compound interval from two base intervals
    compound_interval = MusicTheory.compound_interval_from_bases(Interval.M3, Interval.P5)
    # get the inverse of the compound interval
    inverse_compound_interval = MusicTheory.inverse_compound_interval(compound_interval)
    print(f"Compound Interval: {compound_interval} - Inverse: {inverse_compound_interval}")

    compound_interval = Interval.M2
    inverse = MusicTheory.inverse_compound_interval(compound_interval)
    print(f"compound interval {compound_interval} {inverse}")

    chord_from_scale_cm = MusicTheory.chord_from_scale(
        'CM', ['C', 'D', 'E', 'F', 'G', 'A', 'B'])
    print(f"Chord from scale C and CM {chord_from_scale_cm}")

    chord_from_scale_dm = MusicTheory.chord_from_scale(
        'Dm', ['D', 'E', 'F', 'G', 'A', 'Bb', 'C'])
    print(f"Chord from scale D minor and Dm {chord_from_scale_dm}")


if __name__ == '__main__':
    scale_triads_demo()
