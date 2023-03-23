"""
 MidiNoteSequences is encapsulates entire MIDI
 re-present.  Each MIDI instrument stored in separate
 list that store sequence of midi pitch , pitch bend,
 cv information.  The object also store all information
 related to Midi Key change, Midi Tempo and Midi
 Time signatures.

 TODO
 For now we do         v.seq = self.key_signatures[-1].seq + 1
 for seq down the line it will move to more generic method
 and abstract class

Author Mus
mbayramo@stanford.edu
spyroot@gmail.com
"""
import logging
from collections import OrderedDict
from functools import cache
from types import MappingProxyType
from typing import List, Callable, Any, Union
from typing import Tuple, Iterator, Optional
import bisect

from bisect import bisect_right
from functools import cache
from typing import List, Callable, Any
from typing import Tuple, Iterator, Optional

import numpy as np

from .abstract_midi_reader import MidiBaseReader
from .midi_abstract_event import MidiEvent
from .midi_instruments import MidiInstrumentInfo
from .midi_key_signature import MidiKeySignature
from .midi_note import MidiNote
from .midi_sequence import MidiNoteSequence
from .midi_spec import DEFAULT_PPQ, DEFAULT_QPM
from .midi_time_signature import MidiTempoSignature
from .midi_time_signature import MidiTimeSignature
from .sorted_dict import SortedDict


class MidiNoteSequences:
    def __init__(
            self,
            filename: Optional[str] = "",
            resolution: Optional[int] = DEFAULT_PPQ,
            is_debug: Optional[bool] = True,
            midi_seq: Optional[Union[MidiNoteSequence, List[MidiNoteSequence]]] = None,
            time_signatures: Optional[List[MidiTimeSignature]] = None,
            key_signatures: Optional[List[MidiKeySignature]] = None,
            tempo_signatures: Optional[List[MidiTempoSignature]] = None
    ) -> None:
        """
        By default, we create a sequence with initial implicit assumption about time signature 4/4
        and tempo qpm 120.  If midi contains value it will update time step 0.0 if not any other time
        step > will be inserted after.

        Hence, if initially we don't see any tempo signature or time signature but somewhere in the future
        we get time signature it assumed that everything before using default.

        :param resolution:
        :param filename is mandatory argument, so we know what file used to construct internal re-presentation.
        """

        if not isinstance(filename, str):
            raise TypeError(f"filename must be a string, but got {type(filename)}")
        if not isinstance(resolution, int):
            raise TypeError(f"resolution must be an integer, but got {type(resolution)}")
        if not isinstance(is_debug, bool):
            raise TypeError(f"is_debug must be a boolean, but got {type(is_debug)}")

        if midi_seq is not None:
            if isinstance(midi_seq, MidiNoteSequence):
                midi_seq = [midi_seq]
            if not isinstance(midi_seq, list):
                raise TypeError(
                    f"midi_seq must be a MidiNoteSequence "
                    f"or list of MidiNoteSequences, but got {type(midi_seq)}")
            for seq in midi_seq:
                if not isinstance(seq, MidiNoteSequence):
                    raise TypeError(f"All items in midi_seq must "
                                    f"be a MidiNoteSequence, but found {type(seq)}")

        if time_signatures is not None:
            if not isinstance(time_signatures, list):
                raise TypeError(f"time_signatures must be a list "
                                f"of MidiTimeSignature, but got {type(time_signatures)}")
            for ts in time_signatures:
                if not isinstance(ts, MidiTimeSignature):
                    raise TypeError(f"All items in time_signatures"
                                    f" must be a MidiTimeSignature, but found {type(ts)}")
        if key_signatures is not None:
            if not isinstance(key_signatures, list):
                raise TypeError(f"key_signatures must be a list "
                                f"of MidiKeySignature, but got {type(key_signatures)}")
            for ks in key_signatures:
                if not isinstance(ks, MidiKeySignature):
                    raise TypeError(f"All items in key_signatures "
                                    f"must be a MidiKeySignature, but found {type(ks)}")
        if tempo_signatures is not None:
            if not isinstance(tempo_signatures, list):
                raise TypeError(f"tempo_signatures must be a "
                                f"list of MidiTempoSignature, but got {type(tempo_signatures)}")
            for ts in tempo_signatures:
                if not isinstance(ts, MidiTempoSignature):
                    raise TypeError(f"All items in tempo_signatures "
                                    f"must be a MidiTempoSignature, but found {type(ts)}")

        # internal state seq number for each event type.
        self._track_to_idx = {}
        self._last_time_midi_seq_num = 0
        self._last_key_midi_seq_num = 0
        self._last_tempo_midi_seq_num = 0
        self._idx = 0

        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)
        self.logger.setLevel(logging.WARNING)

        if tempo_signatures is None:
            tempo_signatures = [MidiTempoSignature(midi_time=0.0)]

        if key_signatures is None:
            key_signatures = [MidiKeySignature(midi_time=0.0)]

        if time_signatures is None:
            time_signatures = [MidiTimeSignature(midi_time=0.0)]

        if midi_seq is None:
            self.__midi_tracks = None
        else:
            self.__midi_tracks = SortedDict()
            if isinstance(midi_seq, MidiNoteSequence):
                self._add_midi_note_sequence(midi_seq)
            elif isinstance(midi_seq, list):
                for seq in midi_seq:
                    self._add_midi_note_sequence(seq)
            else:
                raise TypeError(
                    "midi_seq must be either a MidiNoteSequence "
                    "or a list of MidiNoteSequence")

        # list of time signature changes
        if time_signatures is None:
            self.time_signatures = []
        else:
            self.time_signatures = time_signatures

        # list of key signature changes
        if key_signatures is None:
            self.key_signatures = []
        else:
            self.key_signatures = key_signatures

        # list of key signature changes
        if tempo_signatures is None:
            self.tempo_signatures = []
        else:
            self.tempo_signatures = tempo_signatures
        #
        self.resolution = resolution

        # total time for entire seq
        self.total_time: float = 0.0

        # a file name for a midi file.
        self.filename: str = filename
        self.source_info = None
        self.is_debug = is_debug
        self.text: str

    @classmethod
    def from_file(cls, file_path: str, reader: MidiBaseReader):
        """Create a new MidiNoteSequence from a MIDI
        file using the provided reader.
        :param reader: reader: The reader object to use for loading the MIDI file.
        :param file_path: file_path: The path to the MIDI file.
        :raises ValueError: If the file could not be loaded or parsed.
        :return:
        """
        try:
            midi_data = reader.read(file_path)
        except Exception as e:
            raise ValueError(f"Unable to load file {file_path}") from e

        return cls(filename=file_path, midi_seq=midi_data)

    def __getitem__(self, track_idx: int) -> MidiNoteSequence:
        """Returns the MIDI note sequence for the instrument with the given instrument id.
        where instrument id is MIDI Instrument id from 0 to 127.
        If a sequence for the given instrument does not exist, a new sequence
        will be created with the default time resolution.

        Caller need check idx before pass otherwise method will raise error.

        :param track_idx: The midi id of the midi track.
        :return:The MIDI note sequence for the given instrument.
        :rtype: MidiNoteSequence
        :raises TypeError: If idx is not an integer.
        :raises ValueError: If idx is less than zero or greater valid MIDI instruments.
        """
        if not isinstance(track_idx, int):
            raise TypeError("index must be an integer")

        if track_idx < 0 or track_idx >= 128:
            raise ValueError("index must be between 0 and 127 (inclusive)")

        if self._track_to_idx is None:
            self._instrument_to_idx = {}

        if track_idx not in self._track_to_idx:
            internal_idx = len(self.__midi_seqs)
            instrument = MidiInstrumentInfo(
                instrument=track_idx,
                name=f"Instrument {0}",
                is_drum=False
            )
            self._track_to_idx[track_idx] = internal_idx
            self.__midi_seqs[internal_idx] = MidiNoteSequence(
                resolution=self.resolution,
                instrument=instrument,
                is_debug=self.is_debug
            )

        internal_idx = self._track_to_idx[track_idx]
        return self.__midi_seqs[internal_idx]

    def __setitem__(self, track_idx: int, value: Union[MidiNoteSequence, List[MidiNote]]) -> None:
        """ Sets the MidiNoteSequence for the given track index.
        :param track_idx: The midi id of the midi track.
        :param value: The new MidiNoteSequence for the given track index.
        :return: None
        :raises TypeError: If idx is not an integer.
        :raises ValueError: If idx is less than zero or greater valid MIDI instruments.
        """
        print(f"__setitem__ WITH {track_idx} {type(value)}")
        if not isinstance(track_idx, int):
            raise TypeError("index must be an integer")

        if track_idx < 0 or track_idx >= 128:
            raise ValueError("index must be between 0 and 127 (inclusive)")

        if isinstance(value, MidiNoteSequence):
            print("Creating from list")
            self.__midi_seqs[track_idx] = value
        elif isinstance(value, list):
            print("Creating from list")
            self.__midi_seqs[track_idx] = MidiNoteSequence.from_notes(notes=value)
        else:
            raise TypeError("value must be a MidiNoteSequence or a list of MidiNotes")

    def __len__(self) -> int:
        """Return len of midi_seq.
        :return:
        """
        if self.__midi_seqs is None:
            return 0

        return len(self.__midi_seqs)

    def __iter__(self):
        """Return an iterator over the keys sorted in ascending order."""
        return iter(self.__midi_seqs.keys())

    def __repr__(self):
        """
        :return:
        """
        time_signatures = "" if not self.time_signatures else f"time_signatures={self.time_signatures}, "
        key_signatures = "" if not self.key_signatures else f"key_signatures={self.key_signatures}, "
        resolution = f"resolution={self.resolution}, "
        tempo_signatures = "" if not self.tempo_signatures else f"tempo_signatures={self.tempo_signatures}, "
        tracks = "" if self.__midi_seqs is None else ', '.join([str(seq) for seq in self.__midi_seqs.values()])
        return f"MidiNoteSequences({time_signatures}{key_signatures}{resolution}{tempo_signatures}" \
               f"total_time={self.total_time:.2f}, {tracks})"

    def __str__(self):
        """
        :return:
        """
        time_signatures = "" if not self.time_signatures else f"time_signatures={self.time_signatures}, "
        key_signatures = "" if not self.key_signatures else f"key_signatures={self.key_signatures}, "
        resolution = f"resolution={self.resolution}, "
        tempo_signatures = "" if not self.tempo_signatures else f"tempo_signatures={self.tempo_signatures}, "
        tracks = "" if self.__midi_seqs is None else ', '.join([str(seq) for seq in self.__midi_seqs.values()])

        return f"MidiNoteSequences({time_signatures}{key_signatures}{resolution}{tempo_signatures}" \
               f"total_time={self.total_time:.2f}, {tracks})"

    def num_track(self):
        """Returns number of instruments
        :return:
        """
        if self.__midi_seqs is not None:
            return len(self.__midi_seqs)
        return 0

    def num_instruments(self):
        """Returns number of instruments
        :return:
        """
        if self.__midi_seqs is not None:
            return len(set(seq.instrument.instrument_num for seq in
                           self.__midi_tracks.values())) if self.__midi_tracks is not None else 0
        return 0

    @cache
    def initial_tempo(self):
        """Return the initial tempo of the MIDI sequence, defined as the tempo
        signature with the earliest MIDI time. If there are no tempo signatures
        in the sequence, return the default PPQ value.

        :return: The initial implicate tempo of the MIDI sequence.
        """
        if len(self.tempo_signatures) == 0:
            return DEFAULT_PPQ

        for seq_tempo in self.tempo_signatures:
            if seq_tempo.midi_time == 0.0:
                return seq_tempo.qpm

        return self.tempo_signatures[0].qpm

    def instrument_midi_seq_len(self, idx: int) -> int:
        """Return the number of MIDI notes in the MIDI sequence for the given instrument.
        :param idx: instrument id, The ID of the instrument
        :return:
        """
        if self.__midi_seqs is None:
            self.__midi_seqs = SortedDict()

        if idx in self.__midi_seqs:
            return len(self.__midi_seqs[idx].notes)

        return 0

    @staticmethod
    def is_close_to_zero(a, b, tol=1e-6) -> bool:
        return abs(a - b) < tol

    def add_time_signatures(self, v: MidiTimeSignature) -> None:
        """Adds a keys signature to the keys signature list. If the length of
        the list is less than or equal to one and the new signature's MIDI time
        is close to 0.0, we update the first element. In all other cases, we insert the signature
        and maintain the order based on MIDI time.
        :param v: A `MidiTimeSignature` object representing the tempo signature to add.
        :return: None
        """

        if self.is_debug:
            logging.debug(f"adding time signature {v}")

        v.seq = self.time_signatures[-1].seq + 1

        # replace default
        if len(self.time_signatures) >= 1 and self.is_close_to_zero(
                self.time_signatures[0].midi_time, v.midi_time):
            self.time_signatures[0] = v
        else:
            bisect.insort_left(self.time_signatures, v, key=lambda k: k)

        self.time_signature.cache_clear()

    def add_key_signatures(self, v: MidiKeySignature) -> None:
        """Adds a keys signature to the keys signature list. If the length of
        the list is less than or equal to one and the new signature's MIDI time
        is close to 0.0, we update the first element. In all other cases, we insert the signature
        and maintain the order based on MIDI time.
        :param v: A `MidiKeySignature` object representing the tempo signature to add.
        :return: None
        """
        if self.is_debug:
            logging.debug(f"adding key signature {v}")

        v.seq = self.key_signatures[-1].seq + 1

        # replace default
        if len(self.key_signatures) >= 1 and self.is_close_to_zero(
                self.key_signatures[0].midi_time, v.midi_time):
            self.key_signatures[0] = v
        else:
            # we track order and if midi time the same we use order that received.
            if self.time_signatures and v.midi_time > self.time_signatures[-1].midi_time:
                self._last_key_midi_seq_num += 1

            bisect.insort_left(self.key_signatures, v, key=lambda k: k)

            # bisect.insort_left(
            #     bisect.insort_left(self.time_signatures, v, key=lambda k: k)
            # )

        self.key_signature.cache_clear()

    def add_tempo_signature(self, v: MidiTempoSignature) -> None:
        """Adds a tempo signature to the tempo signature list. If the length of
        the list is less than or equal to one and the new signature's MIDI time
        is close to 0.0, we update the first element.
        In all other cases, we insert the signature and maintain the order based on MIDI time.
        :param v: A `MidiTempoSignature` object representing the tempo signature to add.
        :return: None
        """
        if self.is_debug:
            logging.debug(f"adding tempo a signature {v}")

        v.seq = self.tempo_signatures[-1].seq + 1

        if len(self.tempo_signatures) >= 1 and self.is_close_to_zero(
                self.tempo_signatures[0].midi_time, v.midi_time):
            self.tempo_signatures[0] = v
        else:
            bisect.insort_left(self.tempo_signatures, v, key=lambda k: k)

        self.tempo_signature.cache_clear()

    def slice_time_signatures(
            self,
            upper_bound: Optional[float] = None,
            sort_first: Optional[bool] = True) -> Iterator[MidiTimeSignature]:
        """Generator that yields time signatures (MidiTimeSignature) within the given bounds.
        Skips time signatures with a MIDI time greater than the upper bound.

        :param upper_bound: Optional float value indicating the upper bound for the MIDI time.
                            Time signatures with a MIDI time greater than the upper bound will be skipped.
        :param sort_first: Optional bool value indicating whether to sort the time_signatures
                           based on their MIDI time before iterating. Default is True.
        :return: Iterator that emits MidiTimeSignature objects.
        """
        if sort_first:
            _time_signatures = sorted(self.time_signatures)
        else:
            _time_signatures = self.time_signatures

        # note we assume unsorted
        for ts in _time_signatures:
            if upper_bound is not None and ts.midi_time > upper_bound:
                continue
            yield ts

    def slice_key_signatures(
            self, upper_bound: Optional[float] = None,
            sort_first: Optional[bool] = True) -> Iterator[MidiKeySignature]:
        """Generator that yields key signatures (MidiKeySignature) within the given bounds.
        Skips key signatures with a MIDI time greater than the upper bound.

        :param upper_bound: Optional float value indicating the upper bound for the MIDI time.
                            Key signatures with a MIDI time greater than the upper bound will be skipped.
        :param sort_first: Optional bool value indicating whether to sort the key_signatures
                           based on their MIDI time before iterating. Default is True.
        :return: Iterator that emits  MidiKeySignature
        """
        if sort_first:
            _key_signatures = sorted(self.key_signatures, key=lambda x: x.midi_time)
        else:
            _key_signatures = self.key_signatures

        for k in _key_signatures:
            if upper_bound is not None and k.midi_time > upper_bound:
                continue
            yield k

    def slice_tempo_signature(
            self, upper_bound: Optional[float] = None,
            sort_first: Optional[bool] = True) -> Iterator[MidiTempoSignature]:
        """Generator that yields tempo changes (MidiTempoSignature) within the given bounds.
         Skips the initial tempo or tempo changes with a MIDI time greater than the upper bound.

        :param upper_bound: Optional float value indicating the upper bound for the MIDI time.
                            Tempo changes with a MIDI time greater than the upper bound will be skipped.
        :param sort_first: Optional bool value indicating whether to sort the tempo_signatures
                           based on their MIDI time before iterating. Default is True.
        :return: Iterator that emits  MidiTempoSignature
        """
        if sort_first:
            _tempo_signatures = sorted(self.tempo_signatures, key=lambda x: x.midi_time)
        else:
            _tempo_signature = self.tempo_signature

        for sig in self.tempo_signatures:
            if sig == self.initial_tempo or upper_bound and sig.midi_time > upper_bound:
                continue
            yield sig

    def is_quantized(self):
        """Return true if all midi sequence for each instrument quantized
        :return:
        """
        return all(n.is_quantized for n in self.__midi_seqs.values())

    def get_all_sequence_events(self) -> List[MidiEvent]:
        """Return a list of all events in the sequence.
        :return: List of all events in the sequence
        """
        all_events = []
        for seq in self.__midi_seqs:
            print("seq type", type(self.__midi_seqs[seq]))
            track = self.__midi_seqs[seq]
            print(f"e {track}")

            e = self.__midi_seqs[seq].get_midi_events()
            all_events.extend(self.__midi_seqs[seq].get_midi_events())

        all_events.sort(key=lambda x: x.start_time)
        return all_events

    def trim(self, event_time, initial_time: float = 0.0):
        """Trim each respected midi sequence to the event time midi event time.
        By default, we use a start position 0.0.
        :param initial_time:
        :param event_time:
        :return:
        """
        for k in self.__midi_seqs:
            self.__midi_seqs[k].slice(initial_time, event_time)

    def truncate(self,
                 initial_time: Optional[float] = 0.0,
                 from_start: Optional[float] = None) -> Optional[float]:
        """Truncate the sequence to end at the last event that occurs within the
        specified offset from the start of the sequence. Returns the time of the
        last event that occurs before the truncated end of the sequence.
        :param initial_time: positive offset from begin.
         :param from_start:  time offset from the start.
        :return:
        """
        if from_start is not None and from_start < 0:
            raise ValueError("offset must be non-negative")

        if from_start is None:
            last_event_time = max(
                [seq_event.event_end_time for seq_event in self.get_all_sequence_events()]
            )
            self.trim(initial_time, last_event_time)
            return last_event_time

        last_event_time = None
        for seq_event in self.get_all_sequence_events():
            if seq_event.event_end_time <= from_start:
                last_event_time = seq_event.event_end_time
            else:
                break

        if last_event_time is not None:
            for midi_seq in self.__midi_seqs:
                midi_seq.trim(0.0, last_event_time)

        return last_event_time

    @property
    def instruments(self) -> List[MidiInstrumentInfo]:
        """Return list of all instruments
        :return:
        """
        return [s.instrument for s in self.__midi_seqs.values()]

    @staticmethod
    def is_sorted(target_list: list, key: Callable[[Any], float]) -> bool:
        """ Checks if the target_list is sorted by the given key function
        :param target_list:
        :param key: key function to extract the value to compare for sorting
        :return: True if the list is sorted, False otherwise
        """
        for i in range(len(target_list) - 1):
            if key(target_list[i]) > key(target_list[i + 1]):
                return False
        return True

    @cache
    def time_signature(self, start_time: float) -> Tuple[int, int]:
        """Return the time signature (denominator, numerator) based on the MIDI start time.
        For example, for a pitch start time.
        :param start_time: The MIDI start time.
        :return: A tuple representing the time signature (denominator, numerator).
        """
        # inc ase start time negative
        if start_time < 0:
            return DEFAULT_PPQ, DEFAULT_QPM

        # default assumed signature
        if self.time_signatures is None or len(self.time_signatures) == 0:
            return 4, 4

        i = bisect_right(self.time_signatures, MidiTimeSignature(midi_time=start_time)) - 1
        return self.time_signatures[i].numerator, self.time_signatures[i].denominator

    @cache
    def tempo_signature(self, start_time: float) -> Tuple[int, int]:
        """Return time signature based on midi start time.
          For example for a pitch start time.
        :param start_time:
        :return:
        """
        # inc ase start time negative
        if start_time < 0:
            return DEFAULT_PPQ, DEFAULT_QPM

        if self.tempo_signature is None or len(self.tempo_signatures) == 0:
            return DEFAULT_PPQ, DEFAULT_QPM

        i = bisect_right(self.tempo_signatures, MidiTempoSignature(midi_time=start_time)) - 1
        return int(self.tempo_signatures[i].qpm), self.tempo_signatures[i].resolution

    @cache
    def key_signature(self, start_time: float) -> Tuple[int, int]:
        """Return key signature tuple.
        :param start_time:
        :return:
        """
        if self.key_signatures is None or len(self.key_signatures) == 0:
            return 0, 0
        i = bisect_right(self.key_signatures, MidiKeySignature(midi_time=start_time)) - 1
        return self.key_signatures[i].mode, self.key_signatures[i].midi_key

    @property
    def __midi_seqs(self):
        """
        :return:
        """
        if self.__midi_tracks is None:
            self.__midi_tracks = SortedDict()
        return self.__midi_tracks

    @__midi_seqs.setter
    def __midi_seqs(self, midi_seqs: Union[MidiNoteSequence, List[MidiNoteSequence]]) -> None:
        """
        :param midi_seqs:
        :return:
        """
        if isinstance(midi_seqs, MidiNoteSequence):
            midi_seqs = [midi_seqs]

        for midi_seq in midi_seqs:
            instrument_program = midi_seq.instrument.instrument_num
            if instrument_program in self._track_to_idx:
                existing_idx = self._track_to_idx[instrument_program]
                if isinstance(midi_seq, MidiNoteSequence):
                    self.__midi_tracks[existing_idx].merge(midi_seq)
                else:
                    self.__midi_tracks[existing_idx].extend(midi_seq)
            else:
                if isinstance(midi_seq, MidiNoteSequence):
                    self.__midi_tracks[instrument_program] = midi_seq
                else:
                    self.__midi_tracks[instrument_program] = MidiNoteSequence(
                        instrument=MidiInstrumentInfo(instrument_program, midi_seq[0].program, False))
                    self.__midi_tracks[instrument_program].notes = midi_seq
                self._track_to_idx[instrument_program] = instrument_program

    def _add_midi_note_sequence(
            self, midi_seqs: Union[MidiNoteSequence, List[MidiNoteSequence]]) -> None:
        """Add a MidiNoteSequence or a list of MidiNoteSequence
          to the midi_seqs dictionary and update the  mapping accordingly.

          We construct number of track based on number of instrument.
          and the instrument inferred from instrument_num.

        :param midi_seqs: The MidiNoteSequence or list of MidiNoteSequence objects to be added.
        :return: None
        """
        if not isinstance(midi_seqs, list):
            midi_seqs = [midi_seqs]

        for midi_seq in midi_seqs:
            instrument_program = midi_seq.instrument.instrument_num
            if instrument_program in self._track_to_idx:
                existing_idx = self._track_to_idx[instrument_program]
                self.__midi_seqs[existing_idx].merge(midi_seq)
            else:
                self.__midi_seqs[instrument_program] = midi_seq
                self._track_to_idx[instrument_program] = instrument_program

                # self._instrument_to_idx[instrument_program] = self._idx
                # self._idx += 1
        # self._instrument_to_idx = OrderedDict(
        #     sorted(self._instrument_to_idx.items(), key=lambda t: t[1])
        # )

    def _create_new_midi_seq(self, name: str, program: int, is_drum: bool):
        """Create a midi seq.
        :return:
        """
        print(f"Creating instrument {name} {program} and midi seq id {self._idx}")
        idx = len(self._track_to_idx)
        instrument = MidiInstrumentInfo(
            instrument=program,
            name=name,
            is_drum=is_drum
        )
        self.__midi_seqs[self._idx] = MidiNoteSequence(
            resolution=self.resolution,
            instrument=instrument,
            is_debug=self.is_debug
        )
        self._track_to_idx[program] = self._idx
        self._idx += 1

    def create_track(self, track_idx: int, program: int, name: str, is_drum: bool) -> int:
        """Create instrument, only if given instrument not present
        and return midi seq id.

        :param track_idx: MIDI track.
        :param program: The MIDI program number (integer between 0 and 127 inclusive).
        :param name: The name of the instrument (string).
        :param is_drum: A boolean indicating if the instrument is a drum instrument.
        :return: The MIDI seq id of the instrument.
        :raises ValueError: If the program is not an integer between 0 and 127,
                    if the name is not a string, or if is_drum is not a boolean value.
        """
        if not (isinstance(program, (int, np.integer)) and 0 <= program < 128):
            raise ValueError(
                f"Program must be an integer between 0 and 127 "
                f"inclusive, received {program} type {type(program)}")

        if not isinstance(name, str):
            raise ValueError("Name must be a string.")

        if not isinstance(is_drum, bool):
            raise ValueError("is_drum must be a boolean value.")

        track_idx = int(track_idx)
        program = int(program)

        if self._track_to_idx is None:
            self._track_to_idx = {}

        if track_idx in self._track_to_idx:
            return self._track_to_idx[track_idx]

        instrument = MidiInstrumentInfo(
            instrument=program,
            name=name,
            is_drum=is_drum
        )

        seq = MidiNoteSequence(
            resolution=self.resolution,
            instrument=instrument,
            is_debug=self.is_debug
        )

        self.__midi_seqs[track_idx] = seq
        self._track_to_idx[track_idx] = track_idx
        # self.__midi_seqs[track_idx] = seq
        # self._track_to_idx[track_idx] = track_idx
        return track_idx

    def get_track(self, track_idx: int, name: str, program: int, is_drum: bool) -> MidiNoteSequence:
        """Returns the MIDI note sequence for the instrument with the given id.
        Where id is midi seq id.

        If a sequence for the given instrument does not exist, a new sequence
        will be created with the default time resolution.

        :param track_idx: The id of midi seq of the instrument.
        :param program: instrument program number.
        :param name: instrument name
        :param is_drum:  drum or note.
        :return: The MIDI note sequence for the given instrument.
        :raises ValueError: If idx is less than zero or greater valid MIDI instruments.
        """
        if track_idx < 0 or track_idx >= 128:
            raise ValueError("Invalid instrument id")

        if self.__midi_seqs is None:
            self.__midi_seqs = SortedDict()

        if track_idx not in self.__midi_seqs:
            self.create_track(track_idx, program, name, is_drum)

        return self.__midi_seqs[track_idx]

    def set_sequence(self, track_idx: int, sequence: Union[List[MidiNote], MidiNoteSequence]) -> None:
        """Set the MidiNoteSequence for the given track index."""
        if not isinstance(sequence, MidiNoteSequence):
            sequence = MidiNoteSequence(resolution=self.resolution, is_debug=self.is_debug)
            sequence.midi_notes = sequence.midi_notes + sequence

        self.__midi_seqs[track_idx] = sequence

