"""
 MidiNoteSequence re-present
 a midi note sequence for single midi instrument.

 Each midi sequence store all changes in CV, Pitch Bends,
 and notes or drum events.

Class differentiate internally all drum event from pitch events.
Each stored in separate list.

notes property provide access to note sequence that encapsulates
either drum or none drum sequence.

Author Mus
mbayramo@stanford.edu
spyroot@gmail.com
"""
import copy
import itertools
from functools import cache
from typing import Optional, List, Any

import numpy as np

from .midi_abstract_event import MidiEvents, MidiEvent
from .midi_note import MidiNote
from .midi_pitch_bend import MidiPitchBend
from .midi_spec import DEFAULT_PPQ
from .midi_control_change import MidiControlChange
from .midi_instruments import MidiInstrumentInfo
from .midi_time_signature import MidiTempoSignature


# TODO evaluate two options.
# Either use Interval Tree so we can do quick search based on Interval Trees
# Option to use heap priority queue.
# Note both method need evaluate and bench marked vs sorted list.
#
# For now we store unsorted so if PrettyMIDI emit event in random order
# we need handle in the code.
# import heapq as hq


class MidiNoteSequence(MidiEvents):
    def __init__(self,
                 notes: List[Optional[MidiNote]] = None,
                 drum_events: List[Optional[MidiNote]] = None,
                 instrument: Optional[MidiInstrumentInfo] = None,
                 resolution: Optional[int] = 220,
                 is_debug: Optional[bool] = True):
        """
        Creates a new MidiNoteSequence object.

        4/4 would be four quarter-notes per bar (MIDI default),
        4/2 would be four half-notes per bar (or 8 quarter notes),
        4/8 would be four eighth-notes per bar (or 2 quarter notes), and
        2/4 would be two quarter-notes per Bar.


        microseconds per tick = microseconds per quarter note / ticks per quarter note

        Resolution in ticks/beat (or equivalently ticks/Quarter note).
                      This fixes the smallest time interval to be generated.

        Tempo in microseconds per beat, which determines
                 how many ticks are generated in a set time interval.

        Elapsed time which provides the fixed
                timebase for playing the midi events.

        ticks = resolution *  (1 / tempo) * 1000 * elapsed_time

        :param notes: a list of MidiNote
        :param drum_events:  a list of Drum Events it also represented as Midi notes
                             Note a drums are just CC msg and doesn't hold pitch information.
        :param resolution:
        :param instrument: a MidiInstrumentInfo object
        :param notes: a list of MidiNote objects
        :param drum_events: a list of drum events represented as MidiNote objects
        :param is_debug: whether to print debug messages or not
        """
        # midi instrument information
        if instrument is None:
            self.instrument = MidiInstrumentInfo(0, "Generic")
        else:
            self.instrument = instrument

        self._notes = [] if notes is None else [n for n in notes if not self.instrument.is_drum]
        self._drum_events = [] if notes is None else [n for n in notes if self.instrument.is_drum]
        # midi seq total time
        self._total_time = max([n.end_time for n in self.notes], default=0.0)

        # list of cv changes
        self.control_changes: List[MidiControlChange] = []
        # list of pitch bends
        self.pitch_bends: List[MidiPitchBend] = []

        self.time: float = 0
        self.reference_number: int
        # 240 ticks per 16th note.
        # 480 ticks per 8th note
        # 960 ticks per quarter note
        self.ticks_per_quarter: int = 0

        # total time for this midi seq for given instrument.
        self.total_quantized_steps: int = 0
        self.quantized_step: int = 0
        self.debug = is_debug

        # midi source information
        self.id: str

        self.resolution = resolution

        self.tempo_signature: List[MidiTempoSignature] = []
        self.steps_per_quarter = 0
        self.steps_per_second = 0

    def __repr__(self):
        """
        :return:
        """
        notes = f'MidiNoteSequence({self.instrument}' + ', '.join([str(n) for n in self._notes]) + ')'
        return notes

    def __str__(self):
        """
        :return:
        """
        notes = f'MidiNoteSequence({self.instrument}' + ', '.join([str(n) for n in self._notes]) + ')'
        return notes

    @property
    def total_time(self) -> float:
        """Returns the total time of the midi sequence."
        :return:
        """
        return self._total_time

    @total_time.setter
    def total_time(self, value: float) -> None:
        """Sets the total time of the midi sequence.
        :param value:
        :return:
        """
        if value < 0:
            raise ValueError("total midi sequence time cannot be negative")
        self._total_time = value

    @property
    def drum_events(self) -> List[MidiNote]:
        """Return drum events."""
        return self._drum_events if self.instrument and self.instrument.is_drum else []

    @property
    def notes(self) -> List[MidiNote]:
        """Return notes or drum events."""
        if self.instrument is not None and self.instrument.is_drum:
            return self._drum_events

        return self._notes

    @notes.setter
    def notes(self, notes_list: List[MidiNote]) -> None:
        """Set the list of notes for the midi sequence, if MIDI sequence
        drum internally update _drum_events otherwise internally update _notes
        :param notes_list: A list of MidiNote objects to set as the notes for the sequence.
        :return:
        """
        if not self.instrument.is_drum:
            self._notes = notes_list
            self.total_time = max([n.end_time for n in self._notes], default=0.0)
        else:
            self._drum_events = notes_list
            self.total_time = max([n.end_time for n in self._drum_events], default=0.0)

    def __len__(self):
        return len(self.notes)

    def __getitem__(self, index):
        return self.notes[index]

    def __iter__(self):
        return iter(self.notes)

    def append(self, note: MidiNote):
        """Append a note and update total_time.
        """
        self.notes.append(note)
        self.total_time = max(self.total_time, note.end_time)

    def extend(self, notes: List[MidiNote]):
        """
        :param notes:
        :return:
        """
        self.notes.extend(notes)
        self.total_time = max(self.total_time, max(note.end_time for note in notes))

    def as_note_seq(self) -> List[int]:
        """Return note sequence as list.
        :return: list of ints
        """
        if self.notes:
            return [n.pitch for n in self.notes]

        return []

    def add_note(self, note: MidiNote) -> None:
        """Add a note to midi seq,  if given midi note is drum
        it added to drum seq.
        :param note: MidiNote object
        :raise a ValueError if the instrument of the note does not match the instrument of the sequence.
        :return: None
        """
        if note.instrument != self.instrument.instrument_num:
            raise ValueError(f"Instrument {note.instrument} of note "
                             f"must match instrument of sequence {self.instrument.instrument_num}")

        if note.is_drum:
            self._drum_events.append(note)
            update_len = len(self._drum_events)
        else:
            update_len = len(self._notes)
            self._notes.append(note)

        self.total_time = max(self.total_time, note.end_time)

        if self.debug:
            print(f"instrument {note.instrument} name {self.instrument.name} "
                  f"note {note.pitch} name {note.pitch_name} "
                  f"{note.start_time} {note.end_time} drum "
                  f"{self.instrument.is_drum} seq len {update_len}")

    def calculate_min_step(self) -> float:
        """Calculates the smallest time difference between any two notes in the sequence.
        The distance computed from start to start of another note
        :return: The smallest time difference between any two notes in the sequence.
        """
        if len(self.notes) < 2:
            return 0.0

        sorted_notes = sorted(self.notes, key=lambda note: note.start_time)
        min_step = min((b.start_time - a.start_time for a, b
                        in zip(sorted_notes, sorted_notes[1:])
                        if a.end_time < b.start_time), default=0.0)
        return min_step

    def add_control_changes(self, cv: MidiControlChange) -> None:
        """Add control change
        :param cv: is MidiControlChange
        :return:
        """
        self.control_changes.append(cv)

    def add_pitch_bends(self, pitch_bend: MidiPitchBend) -> None:
        """
        :param pitch_bend: is MidiPitchBend object
        :return: Nothing
        """
        self.pitch_bends.append(pitch_bend)

    def is_quantized(self) -> bool:
        """Determines whether all notes in the sequence are quantized.
        :return:
        """
        return all(note.is_quantized() for note in self.notes)

    def quantize(self, resolution: int):
        """Quantizes the start and end times of each note to the nearest multiple of the resolution."""
        for note in self.notes:
            note.quantize(resolution)

    def stretch(self, factor: float):
        """Stretch each note in sequence by a factor.
        """
        for note in self.notes:
            note.start_time *= factor
            note.end_time *= factor

    def truncate(self, end_time: float):
        """Truncates the sequence so that all notes end before `end_time`.
        :param end_time: The new end time for the sequence
        """
        new_notes = []
        for note in self.notes:
            if note.end_time > end_time:
                # Truncate the note
                note = copy.deepcopy(note)
                note.end_time = end_time
            new_notes.append(note)

        self.notes = new_notes
        self.total_time = end_time

    @staticmethod
    def compute_intervals(seq_sorted_by_time: List[MidiNote],
                          num_splits: List[int],
                          skip_boundary: Optional[bool] = False) -> List[float]:
        """Computes split intervals based on number of n  splits.
        If note on boundary of a split if skip on boundary
        is True and note in segment we append to a list of intervals.

        :param seq_sorted_by_time: a list of note sequence
        :param num_splits: list that store number of desired splits.
        :param skip_boundary:  indicates whatever we want splitting on boundary
        :return: list of floats
        """
        intervals = [0.0]  # start with 0 as the first interval
        notes_in_span = []

        for split_point in num_splits:
            # update list of notes in current span
            while notes_in_span and notes_in_span[0].end_time <= split_point:
                notes_in_span.pop(0)

            # append notes that cross the split point to the list of notes in current span
            while seq_sorted_by_time and seq_sorted_by_time[0].start_time < split_point:
                note = seq_sorted_by_time.pop(0)
                if note.end_time > split_point:
                    notes_in_span.append(note)

                # append split point to intervals if it's not on a boundary
                # or if there are no notes in the current span
                if not (skip_boundary and notes_in_span):
                    intervals.append(split_point)

        return intervals

    def split(self, span: float,
              skip_boundary: Optional[bool] = False):
        """
        :param span: a required window size in second for a slice
        :param skip_boundary: indicates whatever we want splitting on boundary
        :return:
        """
        # sort all by start time
        sorted_notes = sorted(
            list(self.notes),
            key=lambda note: note.start_time)

        if isinstance(span, list):
            # sort if unsorted
            num_splits = sorted(span)
        else:
            num_splits = np.arange(
                span, self.total_time, span
            )

        # compute span interval that we chop
        # note here if we use interval tree that will reduce complexity
        intervals = self.compute_intervals(sorted_notes, num_splits, skip_boundary)

        # add last tail end of chop
        if self.total_time > intervals[-1]:
            intervals.append(self.total_time)

        if len(intervals) <= 1:
            return []

        return MidiNoteSequence.extract_notes(intervals)

    def shift_times(self, shift_time: float) -> None:
        """Shifts the start and end times of all notes in the sequence by `time_shift`.
        Method allows you to shift all the note start and end times.
        """

        if self.is_quantized():
            raise ValueError("You need to quantize first")

        for note in self.notes:
            note.start_time += shift_time
            note.end_time += shift_time

        self.total_time += shift_time
        # sort the notes by start time again
        self.notes.sort(key=lambda note: note.start_time)
        self.total_time += shift_time

    def extract_notes(
            self, split_times: List[float]) -> List[Any]:

        """Extracts notes from a midi sequence."""
        if self.is_quantized():
            raise ValueError("You need quantize first")

        if len(split_times) < 2:
            raise TypeError(f"Split time {split_times} should have start and end.")
        assert all(t1 <= t2 for t1, t2 in zip(split_times[:-1], split_times[1:])), "Unsorted list of splits."

        if any(t >= self.total_time for t in split_times[:-1]):
            raise TypeError(
                f"Split container time pass the total {self.total_time}. "
                "extract subsequence past end of sequence.")

        sub_sequences = [MidiNoteSequence() for _ in range(len(split_times) - 1)]
        for sub_seq in sub_sequences:
            sub_seq.total_time = 0.0

        # sort notes and for each note extract
        for note in sorted(self.notes, key=lambda n: n.start_time):
            if note.start_time < split_times[0]:
                continue
            sbs = next((i for i, t in enumerate(split_times[:-1])
                        if t <= note.start_time < split_times[i + 1]), None)
            if sbs is None:
                break

            sub_seq = sub_sequences[sbs]
            sub_seq.notes.append(copy.deepcopy(note))
            sub_seq.notes[-1].start_time -= split_times[sbs]

            sub_seq.notes[-1].end_time = min(note.end_time, split_times[sbs + 1]) - split_times[sbs]
            sub_seq.notes[-1].end_time = min(sub_seq.notes[-1].end_time,
                                             sub_seq.total_time + sub_seq.notes[-1].start_time)
            sub_seq.total_time = sub_seq.notes[-1].end_time + sub_seq.notes[-1].start_time

        return sub_sequences

    def slice(self, start: float, end: float, is_strict: Optional[bool]):
        """slice notes in specific start and end time.

         This mainly for a case where midi seq has very long empty spaces.

        - Notes starting before note start ignored so all notes > start
        - Notes after end are ignored i.e. all notes < before end
        - Anything else in range include in slice

        :param start: a start time
        :param end: end time
        :param is_strict: dictate if we slice check for quantization
                          otherwise we slice without, later one assume midi note
                          already sorted.
        :return:
        """
        if is_strict and not self.is_quantized():
            raise ValueError("You need quantize sequence first.")

        # create empty clone
        new_midi_seq = self.clone_empty()

        #
        for note in self.notes:
            if note.start_time < start or \
                    note.start_time >= end:
                continue
            #
            new_note = copy.deepcopy(note)
            new_note.end_time = min(note.end_time, end)
            new_midi_seq.add_note(note)

        # update total time based on
        new_midi_seq.total_time = min(self.total_time, end)
        return new_midi_seq

    @staticmethod
    def quantize_to_nearest_step(midi_time,
                                 steps_per_second,
                                 cutoff: Optional[float] = 0.5):
        """quantize to a nearest step based on step per second."""
        return int(midi_time * steps_per_second + (1 - cutoff))

    # semi_quaver=
    # quaver = 1/2
    # crochet = 1
    # minim = 2
    # semibreve = 4

    def quantize(self, quantized_factor):
        """In place quantize this midi sequence.
        :param quantized_factor: is quantized factor
        :return:
        """
        self._quantize(quantized_factor)

    def _quantize(self, amount: Optional[float] = 0.5):
        """Quantize corrects the timing of MIDI notes to a specific
        rhythmic grid notes.

        - Notes start and end time moved to the nearest step.
        - Each second will be divided into quanta time steps.

        - The amount dictates amount of quantization applied to each note.
          For example, you can set the quantization strength to 50%,
          which would move each note halfway between its original timing
          and the quantized grid position.

        :param amount: dictates amount of quantization applied to each note
        :return:
        """
        for note in self._notes:
            # quantize a note based on time to the nearest step
            note.quantize(amount)
            # update total_quantized_steps
            if note.quantized_end_step > self.total_quantized_steps:
                self.total_quantized_steps = note.quantized_end_step

        #  quantize control changes and text annotations.
        for e in itertools.chain(self.control_changes):
            # quantize the event time, disallowing negative time.
            e.quantized_step = self.quantize_to_nearest_step(
                e.time, amount)
            assert (e.quantized_step < 0)

    @cache
    def initial_tempo(self):
        """
        :return:
        """
        if len(self.tempo_signature) == 0:
            return DEFAULT_PPQ

        for seq_tempo in self.tempo_signature:
            if seq_tempo.time == 0:
                return seq_tempo.time

    @cache
    def truncate_to_last_event(self, offset):
        if offset is None:
            return None
        return max([n.end_time for n in self._notes] or [0]) + offset

    def clone(self):
        """Return a clone of this object """
        cloned = MidiNoteSequence()
        cloned._notes = copy.deepcopy(self._notes)
        cloned.control_changes = copy.deepcopy(self.control_changes)
        cloned.pitch_bends = copy.deepcopy(self.pitch_bends)
        cloned.total_time = self.total_time
        cloned.ticks_per_quarter = self.ticks_per_quarter
        cloned.total_quantized_steps = self.total_quantized_steps
        cloned.quantized_step = self.quantized_step
        cloned.instrument = self.instrument
        cloned.resolution = self.resolution
        return cloned

    def clone_empty(self):
        """Return a clone without copy actual notes.
           We only care about total time , signature etc
         """
        cloned = MidiNoteSequence()
        cloned.total_time = self.total_time
        cloned.ticks_per_quarter = self.ticks_per_quarter
        cloned.total_quantized_steps = self.total_quantized_steps
        cloned.quantized_step = self.quantized_step
        cloned.instrument = self.instrument
        cloned.resolution = self.resolution
        return cloned

    def get_midi_events(self) -> List[MidiNote]:
        """We need make it generic abstract, so we can collect all midi event
        :return:
        """
        return self.notes
