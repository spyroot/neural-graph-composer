from unittest import TestCase

from neural_graph_composer.midi.midi_note import MidiNote
from neural_graph_composer.midi.midi_scales import MusicTheory, Interval
from neural_graph_composer.midi.midi_sequence import MidiNoteSequence
from neural_graph_composer.midi.midi_sequences import MidiNoteSequences
from neural_graph_composer.midi.midi_spec import DEFAULT_QPM, DEFAULT_PPQ
from neural_graph_composer.midi.midi_time_signature import MidiTimeSignature, MidiTempoSignature
from neural_graph_composer.midi.midi_seq import MidiSeq
from neural_graph_composer.midi.midi_instruments import MidiInstrumentInfo
import random


class Test(TestCase):
    @staticmethod
    def get_chords_with_name(root_pitch, scale):
        """
        :param root_pitch:
        :param scale:
        :return:
        """
        intervals = MusicTheory.scale_intervals[scale]
        chords = MusicTheory.list_of_chords(root_pitch, intervals)
        chord_names = []
        chord_pitches = []
        for i, chord in enumerate(chords):
            chord_name = MusicTheory.pitch_names[(root_pitch + chord[0]) % 12] + MusicTheory.chord_intervals[i]
            print(f"Chord name chord_name {chord_name}")
            chord_names.append(chord_name)
            chord_pitches.append([MusicTheory.pitch_names[(root_pitch + pitch) % 12] for pitch in chord])
        return chord_names, chord_pitches

    def test_get_chords_with_name(self):
        """
        :return:
        """
        chords = self.get_chords_with_name(0, "major")
        for chord in chords:
            print(chord)
        return chords

    def test_get_chord_intervals(self):
        """
        :return:
        """
        chord_types = ["M", "m", "7", "M7", "m7", "m7b5", "dim7", "aug7", "6", "m6", "9", "M9", "m9", "11", "M11",
                       "m11", "13", "M13", "m13", "add9", "sus2", "sus4", "7sus4"]
        for chord_type in chord_types:
            result = MusicTheory.get_chord_intervals(chord_type)
            expected = MusicTheory.chord_intervals[chord_type]
            self.assertEqual(result, expected, f"Failed for chord_type {chord_type}")

    def test_major_scales(self):
        """
        :return:
        """
        expected = {
            'C': ['C', 'D', 'E', 'F', 'G', 'A', 'B'],
            'C#': ['C#', 'D#', 'F', 'F#', 'G#', 'A#', 'C'],
            'D': ['D', 'E', 'F#', 'G', 'A', 'B', 'C#'],
            'D#': ['D#', 'F', 'G', 'G#', 'A#', 'C', 'D'],
            'E': ['E', 'F#', 'G#', 'A', 'B', 'C#', 'D#'],
            'F': ['F', 'G', 'A', 'A#', 'C', 'D', 'E'],
            'F#': ['F#', 'G#', 'A#', 'B', 'C#', 'D#', 'F'],
            'G': ['G', 'A', 'B', 'C', 'D', 'E', 'F#'],
            'G#': ['G#', 'A#', 'C', 'C#', 'D#', 'F', 'G'],
            'A': ['A', 'B', 'C#', 'D', 'E', 'F#', 'G#'],
            'A#': ['A#', 'C', 'D', 'D#', 'F', 'G', 'A'],
            'B': ['B', 'C#', 'D#', 'E', 'F#', 'G#', 'A#'],
        }
        for root, notes in expected.items():
            with self.subTest(root=root):
                result = MusicTheory.notes_from_scale(root, 'major')
                self.assertEqual(result, notes)

    def test_minor_scales(self):
        """
        :return:
        """
        minor_scales = {
            'C': ['C', 'D', 'D#', 'F', 'G', 'G#', 'A#'],
            'C#': ['C#', 'D#', 'E', 'F#', 'G#', 'A', 'B'],
            'D': ['D', 'E', 'F', 'G', 'A', 'A#', 'C'],
            'D#': ['D#', 'F', 'F#', 'G#', 'A#', 'B', 'C#'],
            'E': ['E', 'F#', 'G', 'A', 'B', 'C', 'D'],
            'F': ['F', 'G', 'G#', 'A#', 'C', 'C#', 'D#'],
            'F#': ['F#', 'G#', 'A', 'B', 'C#', 'D', 'E'],
            'G': ['G', 'A', 'A#', 'C', 'D', 'D#', 'F'],
            'G#': ['G#', 'A#', 'B', 'C#', 'D#', 'E', 'F#'],
            'A': ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
            'A#': ['A#', 'C', 'C#', 'D#', 'F', 'F#', 'G#'],
            'B': ['B', 'C#', 'D', 'E', 'F#', 'G', 'A'],
        }

        for root, notes in minor_scales.items():
            with self.subTest(root=root):
                result = MusicTheory.notes_from_scale(root, 'minor')
                self.assertEqual(result, notes)

    def test_chord_from_scale(self):
        test_cases = [
            ('CM', ['C', 'D', 'E', 'F', 'G', 'A', 'B'], ['C', 'E', 'G']),
            ('Dm', ['C', 'D', 'E', 'F', 'G', 'A', 'B'], ['D', 'F', 'A']),
            # ('FM7', ['C', 'D', 'E', 'F', 'G', 'A', 'B'], ['F', 'A', 'C', 'E']),
            # ('Bdim7', ['C', 'D', 'E', 'F', 'G', 'A', 'B'], ['B', 'D', 'F', 'G#']),
            # ('EM7', ['E', 'F#', 'G#', 'A', 'B', 'C#', 'D#'], ['E', 'G#', 'B', 'D#']),
            ('A7', ['A', 'B', 'C#', 'D', 'E', 'F#', 'G#'], ['A', 'C#', 'E', 'G']),
            # ('Gm6', ['G', 'A', 'Bb', 'C', 'D', 'E', 'F'], ['G', 'Bb', 'D', 'E']),
            # ('Dm7b5', ['D', 'E', 'F', 'G', 'A', 'B', 'C'], ['D', 'F', 'G#', 'C']),
        ]

        for chord_name, scale_notes, expected_chord_notes in test_cases:
            with self.subTest(chord_name=chord_name, scale_notes=scale_notes):
                chord_notes = MusicTheory.chord_from_scale(chord_name, scale_notes)
                self.assertEqual(chord_notes, expected_chord_notes)

    def test_from_semitones(self):
        self.assertEqual(MusicTheory.from_semitones(0), Interval.P1)
        self.assertEqual(MusicTheory.from_semitones(1), Interval.m2)
        self.assertEqual(MusicTheory.from_semitones(2), Interval.M2)
        self.assertEqual(MusicTheory.from_semitones(3), Interval.m3)
        self.assertEqual(MusicTheory.from_semitones(4), Interval.M3)
        self.assertEqual(MusicTheory.from_semitones(5), Interval.P4)
        self.assertEqual(MusicTheory.from_semitones(6), Interval.Tritone)
        self.assertEqual(MusicTheory.from_semitones(7), Interval.P5)
        self.assertEqual(MusicTheory.from_semitones(8), Interval.m6)
        self.assertEqual(MusicTheory.from_semitones(9), Interval.M6)
        self.assertEqual(MusicTheory.from_semitones(10), Interval.m7)
        self.assertEqual(MusicTheory.from_semitones(11), Interval.M7)
        self.assertEqual(MusicTheory.from_semitones(12), Interval.P8)
        self.assertEqual(MusicTheory.from_semitones(13), Interval.m9)
        self.assertEqual(MusicTheory.from_semitones(14), Interval.M9)
        self.assertEqual(MusicTheory.from_semitones(15), Interval.m10)
        self.assertEqual(MusicTheory.from_semitones(16), Interval.M10)
        self.assertEqual(MusicTheory.from_semitones(17), Interval.P11)
        self.assertEqual(MusicTheory.from_semitones(18), Interval.A11)
        self.assertEqual(MusicTheory.from_semitones(19), Interval.P13)
        self.assertEqual(MusicTheory.from_semitones(20), Interval.A13)
        self.assertEqual(MusicTheory.from_semitones(21), Interval.m14)
        self.assertEqual(MusicTheory.from_semitones(22), Interval.M14)
        with self.assertRaises(ValueError):
            MusicTheory.from_semitones(23)

    def test_compound_interval_from_bases(self):
        """
        :return:
        """
        self.assertEqual(MusicTheory.compound_interval_from_bases(Interval.m2, Interval.M2), Interval.m3)
        self.assertEqual(MusicTheory.compound_interval_from_bases(Interval.P5, Interval.P4), Interval.P8)
        self.assertEqual(MusicTheory.compound_interval_from_bases(Interval.m3, Interval.M3), Interval.P5)

        self.assertEqual(MusicTheory.compound_interval_from_bases(Interval.M3, Interval.M3), Interval.A5,
                         msg=f"expected {Interval.A5} "
                             f"got {MusicTheory.compound_interval_from_bases(Interval.M3, Interval.M3)}")

        self.assertEqual(MusicTheory.compound_interval_from_bases(Interval.M3, Interval.m3), Interval.P5,
                         msg=f"expected {Interval.P4} got "
                             f"{MusicTheory.compound_interval_from_bases(Interval.M3, Interval.m3)}")

        self.assertEqual(MusicTheory.compound_interval_from_bases(Interval.M3, Interval.M6), Interval.m9,
                         msg=f"expected {Interval.M9} got "
                             f"{MusicTheory.compound_interval_from_bases(Interval.M3, Interval.M6)}")

        self.assertEqual(MusicTheory.compound_interval_from_bases(Interval.M3, Interval.m3), Interval.P5)
        self.assertEqual(MusicTheory.compound_interval_from_bases(Interval.M3, Interval.M6), Interval.m9)
        self.assertEqual(MusicTheory.compound_interval_from_bases(Interval.M3, Interval.m7), Interval.M9)
