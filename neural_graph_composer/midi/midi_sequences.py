"""
 MidiNoteSequences is encapsulates entire MIDI
 re-present.  Each MIDI instrument stored in separate
 list that store sequence of midi pitch , pitch bend,
 cv information.  The object also store all information
 related to Midi Key change, Midi Tempo and Midi
 Time signatures.

Author Mus
mbayramo@stanford.edu
spyroot@gmail.com
"""
import bisect
import collections
import logging
from bisect import bisect, bisect_right
from functools import cache
from typing import List, Callable, Any
from typing import Tuple, Iterator, Optional

from neural_graph_composer.midi.midi_abstract_event import MidiEvent
from neural_graph_composer.midi.midi_key_signature import MidiKeySignature
from neural_graph_composer.midi.midi_sequence import MidiNoteSequence
from neural_graph_composer.midi.midi_spec import DEFAULT_PPQ, DEFAULT_QPM
from neural_graph_composer.midi.midi_time_signature import MidiTempoSignature
from neural_graph_composer.midi.midi_time_signature import MidiTimeSignature


class MidiNoteSequences:
    def __init__(
            self,
            filename: Optional[str] = "",
            resolution: Optional[int] = DEFAULT_PPQ,
            is_debug: Optional[bool] = True,
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

        # internal state seq number for each event type.
        self._last_time_midi_seq_num = 0
        self._last_tempo_midi_seq_num = 0
        self._last_key_midi_seq_num = 0

        if tempo_signatures is None:
            tempo_signatures = [MidiTempoSignature(midi_time=0.0)]

        if key_signatures is None:
            key_signatures = [MidiKeySignature(midi_time=0.0)]

        if time_signatures is None:
            time_signatures = [MidiTimeSignature(midi_time=0.0)]

        self.midi_seqs = None

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
        #
        self.collection_name: str
        # a file name for a midi file.
        self.filename: str = filename
        self.text: str
        self.source_info = None
        self.is_debug = is_debug

    def __getitem__(self, idx: int) -> MidiNoteSequence:
        """Returns the MIDI note sequence for the instrument with the given instrument id.
        where instrument id is MIDI Instrument id from 0 to 127.
        If a sequence for the given instrument does not exist, a new sequence
        will be created with the default time resolution.

        Caller need check idx before pass otherwise method will raise error.

        :param idx: The midi id of the instrument.
        :return: The MIDI note sequence for the given instrument.
        :rtype: MidiNoteSequence
        :raises TypeError: If idx is not an integer.
        :raises ValueError: If idx is less than zero or greater valid MIDI instruments.
        """
        if not isinstance(idx, int):
            raise TypeError("index must be an integer")

        if idx < 0 or idx >= 128:
            raise ValueError("index must be between 0 and 127 (inclusive)")

        if self.midi_seqs is None:
            self.midi_seqs = collections.OrderedDict()

        if idx not in self.midi_seqs:
            self.midi_seqs[idx] = MidiNoteSequence(resolution=self.resolution)

        return self.midi_seqs[idx]

    def num_instruments(self):
        """Returns number of instruments
        :return:
        """
        if self.midi_seqs is not None:
            return len(self.midi_seqs)
        return 0

    def get_instrument(self, idx: int) -> MidiNoteSequence:
        """Returns the MIDI note sequence for the instrument with the given ID.
        If a sequence for the given instrument does not exist, a new sequence
        will be created with the default time resolution.
        :param idx: The ID of the instrument.
        :return: The MIDI note sequence for the given instrument.
        :raises ValueError: If idx is less than zero or greater valid MIDI instruments.
        """
        if self.midi_seqs is None:
            self.midi_seqs = collections.OrderedDict()

        if idx < 0 or idx >= 128:
            raise ValueError("Invalid instrument id")

        if idx not in self.midi_seqs:
            self.midi_seqs[idx] = MidiNoteSequence(
                resolution=self.resolution,
                is_debug=self.is_debug
            )

        return self.midi_seqs[idx]

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

    def __repr__(self):
        return 'MidiNoteSequences(' + ', '.join([str(self.midi_seqs[k]) for k in self.midi_seqs]) + ')'

    def __str__(self):
        return 'MidiNoteSequences(' + ', '.join([str(self.midi_seqs[k]) for k in self.midi_seqs]) + ')'

    def instrument_midi_seq_len(self, idx: int):
        """Return the number of MIDI notes
        in the MIDI sequence for the given instrument.

        :param idx: instrument id, The ID of the instrument
        :return:
        """
        if self.midi_seqs is None:
            self.midi_seqs = collections.OrderedDict()

        if idx in self.midi_seqs:
            return len(self.midi_seqs[idx].notes)

    @staticmethod
    def is_close_to_zero(a, b, tol=1e-6):
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
    
        # replace default
        if len(self.time_signatures) >= 1 and self.is_close_to_zero(
                self.time_signatures[0].midi_time, v.midi_time):
            self.time_signatures[0] = v
        else:
            if self.time_signatures and v.midi_time > self.time_signatures[-1].midi_time:
                self._last_time_midi_seq_num += 1
                
            # bisect.insort_left(self.time_signatures, v, key=lambda k: k.midi_time)
            bisect.insort_left(self.time_signatures, v, key=lambda k: k.midi_time)

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

        # replace default
        if len(self.key_signatures) >= 1 and self.is_close_to_zero(
                self.key_signatures[0].midi_time, v.midi_time):
            self.key_signatures[0] = v
        else:
            bisect.insort_left(self.key_signatures, v, key=lambda k: k.midi_time)
            self._last_key_midi_seq_num += 1

        self.key_signature.cache_clear()

    def add_tempo_signature(self, v: MidiTempoSignature) -> None:
        """Adds a tempo signature to the tempo signature list. If the length of
        the list is less than or equal to one and the new signature's MIDI time
        is close to 0.0, we update the first element. In all other cases, we insert the signature
        and maintain the order based on MIDI time.
        :param v: A `MidiTempoSignature` object representing the tempo signature to add.
        :return: None
        """
        if self.is_debug:
            logging.debug(f"adding tempo signature {v}")

        if len(self.tempo_signatures) >= 1 and self.is_close_to_zero(
                self.tempo_signatures[0].midi_time, v.midi_time):
            self.tempo_signatures[0] = v
        else:
            bisect.insort_left(self.tempo_signatures, v, key=lambda k: k.midi_time)
            self._last_tempo_midi_seq_num += 1

        self.tempo_signature.cache_clear()

    def slice_time_signatures(
            self,
            upper_bound: Optional[float] = None,
            sort_first: Optional[bool] = True) -> Iterator[MidiTimeSignature]:
        """Generator return time signature, if upper_bound
        is provided will skip time step that > upper_bound.

        :param upper_bound: if upper_bound indicate will use as
                            upper_bound for event time
        :param sort_first: sort first
        :return:
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
        """Generator return key signature, if upper_bound is provided
        will skip time step that > upper_bound.
        Note that we use bisect hence order sorted

        :param upper_bound:
        :param sort_first: will sort all key signature based on time.
        :return: Iterator that emits  MidiKeySignature
        """
        if sort_first:
            _key_signatures = sorted(self.key_signatures)
        else:
            _key_signatures = self.key_signatures

        for k in _key_signatures:
            if upper_bound is not None and k.midi_time > upper_bound:
                continue
            yield k

    def slice_tempo_signature(
            self, upper_bound: Optional[float] = None,
            sort_first: Optional[bool] = True) -> Iterator[MidiTempoSignature]:
        """Generator yield tempo changes if tempo changed i.e.
         Note that we use bisect hence order sorted
        :param upper_bound:
        :param sort_first: will sort all tempo signature based on time.
        :return: Iterator that emits  MidiKeySignature
        """
        if sort_first:
            _tempo_signature = self.tempo_signature
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
        return all(n.is_quantized for n in self.midi_seqs.values())

    def get_all_sequence_events(self) -> List[MidiEvent]:
        """Return a list of all events in the sequence.
        :return: List of all events in the sequence
        """
        all_events = []
        for seq in self.midi_seqs:
            all_events.extend(seq.get_midi_events())

        all_events.sort(key=lambda x: x.start_time)
        return all_events

    def trim(self, event_time, initial_time=0.0):
        """Trim each respected sequence to the event time midi event time.
        By default, we use a start position 0.0.
        :param initial_time:
        :param event_time:
        :return:
        """
        for midi_seq in self.midi_seqs:
            midi_seq.trim(initial_time, event_time)

    def truncate_to_last_event(self, offset: Optional[float] = None) -> Optional[float]:
        """Truncate the sequence to end at the last event that occurs within the
        specified offset from the start of the sequence. Returns the time of the
        last event that occurs before the truncated end of the sequence.
        :param offset:
        :return:
        """
        if offset is not None and offset < 0:
            raise ValueError("offset must be non-negative")

        # offset is optional and if not specified,
        # truncate to the last event in the sequence.
        if offset is None:
            last_event_time = max([seq_event.event_end_time for seq_event in self.get_all_sequence_events()])
            self.trim(0.0, last_event_time)
            return last_event_time

        # truncate the sequence to end at the last event that occurs within the
        # specified offset from the start of the sequence.
        last_event_time = None
        for seq_event in self.get_all_sequence_events():
            if seq_event.event_end_time <= offset:
                last_event_time = seq_event.event_end_time
            else:
                break
        if last_event_time is not None:
            for midi_seq in self.midi_seqs:
                midi_seq.trim(0.0, last_event_time)

        return last_event_time

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
        """Return time signature based on midi start time.
          For example for a pitch start time.
        :param start_time:
        :return:
        """
        # inc ase start time negative
        if start_time < 0:
            return DEFAULT_PPQ, DEFAULT_QPM

        # default assumed signature
        if self.time_signatures is None or len(self.time_signatures) == 0:
            return 4, 4

        i = bisect_right(self.time_signatures, MidiTimeSignature(midi_time=start_time)) - 1
        return self.time_signatures[i].denominator, self.time_signatures[i].numerator
        # for i, ts_sig in enumerate(self.time_signatures):
        #     if start_time < ts_sig.midi_time:
        #         print(f"Start time {start_time} {ts_sig.midi_time} ")
        #         return self.time_signatures[i - 1].denominator, self.time_signatures[i - 1].numerator

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
        return self.tempo_signatures[i].qpm, self.tempo_signatures[i].resolution
        # for i, tempo_sig in enumerate(self.tempo_signatures):
        #     if start_time < tempo_sig.midi_time:
        #         return self.tempo_signatures[i - 1].qpm, self.tempo_signatures[i - 1].resolution

    @cache
    def key_signature(self, start_time: float) -> Tuple[int, int]:
        """Return time signature based on midi start time.
          For example for a pitch start time.
          @todo add test
        :param start_time:
        :return:
        """
        # inc ase start time negative
        if start_time < 0:
            return DEFAULT_PPQ, DEFAULT_QPM

        if self.tempo_signature is None or len(self.key_signatures) == 0:
            return 0, 0

        i = bisect_right(self.key_signatures, MidiKeySignature(midi_time=start_time)) - 1
        return self.key_signatures[i].mode, self.key_signatures[i].midi_key

        # for i, key_sig in enumerate(self.key_signatures):
        #     if start_time < key_sig.midi_time:
        #         return self.key_signatures[i - 1].mode, self.key_signatures[i - 1].midi_key
