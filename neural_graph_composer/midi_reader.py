import collections
import io
import logging
import sys
from typing import Tuple, Iterator, Optional
import logging

from mido import MidiFile

from .midi.midi_sequence import MidiNoteSequence
from .midi.midi_sequences import MidiNoteSequences
from .custom_pretty_midi import CustomPrettyMIDI
from .midi.midi_control_change import MidiControlChange
from .midi.midi_instruments import MidiInstrumentInfo
from .midi.midi_key_signature import MidiKeySignature, KeySignatureType
from .midi.midi_note import MidiNote
from .midi.midi_pitch_bend import MidiPitchBend
from .midi.midi_spec import DEFAULT_PPQ
from .midi.midi_time_signature import MidiTempoSignature, MidiTimeSignature
from pretty_midi import pretty_midi, PitchBend, Note, Instrument

_DEFAULT_QUARTERS_PER_MINUTE = 120.0


class MidiParserError(Exception):
    pass


class MidiReader:
    """
    This is a Python module that contains a MidiReader class which provides methods
    to read a MIDI file and convert it to an internal representation
    (MidiNoteSequences). The module uses the pretty_midi
    library to read MIDI files.
    """

    def __init__(self, is_debug: Optional[float] = bool):
        """
        Midi Reader , read midi file via PrettyMIDI interface and construct
        internal representations.
        """
        self.debug = is_debug

    @staticmethod
    def read_tempo_changes(sequence: MidiNoteSequences, midi):
        """Read tempo change from midi each change is tuple
        :param sequence:
        :param midi:
        :return:
        """
        tempo_times, tempo_qpms = midi.get_tempo_changes()
        for time_in_seconds, tempo_in_qpm in zip(tempo_times, tempo_qpms):
            sequence.add_tempo_signature(
                MidiTempoSignature(time_in_seconds, qpm=tempo_in_qpm)
            )

    @staticmethod
    def read_key_signature(midi_seq: MidiNoteSequences, midi):
        """Reads midi key signature according
        https://www.recordingblogs.com/wiki/midi-key-signature-meta-message
        :param midi_seq: MidiNoteSequence
        :param midi:
        :return:
        """
        for midi_key in midi.key_signature_changes:
            midi_seq.add_key_signatures(
                MidiKeySignature(midi_key.time,
                                 midi_key.key_number % 12,
                                 midi_key.key_number // 12)
            )

    @staticmethod
    def read_time_signature(midi_seq: MidiNoteSequences, midi):
        """Read time signature from pretty midi
        :param midi_seq:
        :param midi:
        :return:
        """
        print("Reading time signature changes.")
        for sc in midi.time_signature_changes:
            try:
                logging.debug(f"Reading time signature changes "
                              f"num {sc.numerator} denom {sc.denominator} time {sc.time}.")
                midi_seq.add_time_signatures(
                    MidiTimeSignature(
                        numerator=sc.numerator,
                        denominator=sc.denominator,
                        midi_time=sc.time
                    )
                )
            except ValueError:
                raise MidiParserError(
                    'Invalid time signature denominator {}' % sc.denominator)

    @staticmethod
    def read_pretty_midi(seq_file) -> CustomPrettyMIDI:
        """ Read pretty midi.
        :param seq_file:
        :return:
        """
        midi_data = None
        is_error = False
        try:
            midi_data = CustomPrettyMIDI(seq_file)
        except Exception as err:
            logging.error('Error reading MIDI file %s: %s', seq_file, err)
            raise err
            is_error = True

        # try to pass raw buffer
        if is_error:
            try:
                midi_data = CustomPrettyMIDI(io.BytesIO(seq_file))
            except Exception as _:
                raise MidiParserError(
                    'Midi decoding error %s: %s' %
                    (sys.exc_info()[0], sys.exc_info()[1]))

        return midi_data

    @staticmethod
    def read_instrument(
            midi_seq: MidiNoteSequences,
            midi_data: CustomPrettyMIDI) -> Iterator[Tuple[Instrument, int, pretty_midi.Note]]:
        """Generator read midi file and returns MIDI program,
        midi idx midi instrument drum or not,  note
        :param midi_seq:
        :param midi_data:
        :return:
        """

        # for i, instrument in enumerate(midi_data.instruments):
        #     yield from ((instrument, i, note) for note in instrument.notes if note.end >= midi_seq[i].total_time)
        for i, instrument in enumerate(midi_data.instruments):
            # for each instrument iterate over all node and get time
            for n in instrument.notes:
                # update total time
                if not midi_seq.total_time or n.end >= midi_seq[i].total_time:
                    # update entire total for all music instruments
                    if n.end >= midi_seq.total_time:
                        midi_seq.total_time = n.end
                    # update for particular instrument total time
                    midi_seq[i].total_time = n.end
                    yield instrument, i, n
                else:
                    logging.debug(
                        f"Skipping instrument {instrument}: "
                        f"note {n} end time after instrument time {midi_seq.total_time}")
                    logging.warning(
                        "instrument {} note pitch {} interval {}-{} vel {} "
                        "skipped".format(instrument, n.pitch, n.start, n.end, n.velocity))
                    logging.warning(
                        "seq total time {}".format(midi_seq.total_time))

    @staticmethod
    def read_instrument_pitch_bends(
            midi_seq: MidiNoteSequences,
            midi_data: CustomPrettyMIDI) -> Iterator[Tuple[Instrument, int, pretty_midi.PitchBend]]:
        """Generator read midi file and returns
        MIDI program, midi idx midi instrument drum or not,  note
        :param midi_seq:
        :param midi_data:
        :return:
        """
        for i, instrument in enumerate(midi_data.instruments):
            # for each instrument iterate over all node and get time
            for n in instrument.pitch_bends:
                if not midi_seq.total_time or n.end >= midi_seq.total_time:
                    yield instrument, i, n

    @staticmethod
    def read_instrument_control_changes(
            midi_seq: MidiNoteSequences,
            midi_data: CustomPrettyMIDI) -> Iterator[Tuple[Instrument, int, pretty_midi.ControlChange]]:
        """Generator read midi file and returns
        MIDI program, midi idx midi instrument drum or not,  note
        :param midi_seq:
        :param midi_data:
        :return:
        """
        for i, instrument in enumerate(midi_data.instruments):
            for cv in instrument.control_changes:
                if not midi_seq.total_time or cv.time >= midi_seq.total_time:
                    yield instrument, i, cv

    @staticmethod
    def read(seq_file, is_debug: Optional[bool] = True) -> MidiNoteSequences:
        """Main midi reader, take mido file or path to a file and
          converts to internal representation MidiNoteSequences

        Each MidiNoteSequences container separate
        MidiNoteSequence for each instrument.

        :param is_debug:
        :param seq_file:  mido.MidiFile or string to a file.
        :return:
        """
        if not isinstance(seq_file, str) and not isinstance(seq_file, bytes):
            if not isinstance(seq_file, MidiFile):
                raise ValueError(f"Illegal argument. input seq must be instance of "
                                 f"mido.MidiFile or path to a file and got {type(seq_file)}")

        midi_data = MidiReader.read_pretty_midi(seq_file)

        midi_data.ticks_per_quarter = midi_data.resolution
        print(f"midi_data.ticks_per_quarter {midi_data.ticks_per_quarter}")

        # midi_seq.filename = seq_file
        midi_seqs = MidiNoteSequences(is_debug=is_debug, resolution=midi_data.ticks_per_quarter)
        # read key signature , tempo signature and time signature.
        # for not we assume it type-0 MIDI.
        MidiReader.read_key_signature(midi_seqs, midi_data)
        MidiReader.read_tempo_changes(midi_seqs, midi_data)
        MidiReader.read_time_signature(midi_seqs, midi_data)

        print(f"Self len of key signature {len(midi_seqs.key_signatures)}")
        print(f"Self len of key signature {len(midi_seqs.time_signatures)}")
        print(f"Self len of key signature {len(midi_seqs.tempo_signatures)}")

        print(f"key key {midi_seqs.key_signatures}")
        print(f"key temo {midi_seqs.tempo_signatures}")
        print(f"key temo {midi_seqs.time_signatures}")

        midi_seqs.filename = seq_file
        midi_seqs.resolution = midi_data.resolution

        # read all instruments and construct midi seq
        # all instruments are merged
        for instrument, instrument_idx, note in MidiReader.read_instrument(midi_seqs, midi_data):
            print(f"Reading {instrument_idx}")
            midi_seq = midi_seqs.get_instrument(instrument_idx)
            midi_seq.instrument = MidiInstrumentInfo(
                instrument_idx,
                name=instrument.name,
                is_drum=instrument.is_drum)

            num, denom = midi_seqs.time_signature(note.start)
            midi_seq.add_note(
                MidiNote(
                    note.pitch,
                    note.start,
                    note.end,
                    program=instrument.program,
                    instrument=instrument_idx,
                    velocity=note.velocity,
                    is_drum=instrument.is_drum,
                    numerator=num,
                    denominator=denom,
                )
            )

        for instrument, instrument_idx, b in MidiReader.read_instrument_pitch_bends(midi_seqs, midi_data):
            midi_seq = midi_seqs.get_instrument(instrument_idx)
            midi_seq.add_pitch_bends(
                MidiPitchBend(
                    b.pitch,
                    midi_time=b.time,
                    program=instrument.program,
                    instrument=instrument_idx,
                    is_drum=instrument.is_drum
                )
            )

        for instrument, instrument_idx, cv in MidiReader.read_instrument_control_changes(midi_seqs, midi_data):
            midi_seq = midi_seqs.get_instrument(instrument_idx)
            midi_seq.add_control_changes(
                MidiControlChange(
                    cv.number,
                    cv.value,
                    cc_time=cv.time,
                    program=instrument.program,
                    instrument=instrument_idx,
                    is_drum=instrument.is_drum,
                )
            )

        return midi_seqs

    @staticmethod
    def write(sequence: MidiNoteSequences,
              from_start: Optional[float] = None,
              to_end: Optional[float] = None):
        """
        :param sequence:
        :param from_start: By default, we write from the start til the end.
        :param to_end:  Till the last midi event.
        :return:
        """
        ticks_per_quarter = sequence.resolution or DEFAULT_PPQ
        max_event_time = sequence.truncate_to_last_event(to_end)
        initial_seq_tempo = sequence.initial_tempo()

        pm = CustomPrettyMIDI(
            resolution=ticks_per_quarter,
            initial_tempo=sequence.initial_tempo().qpm)

        instrument = pretty_midi.Instrument(0)
        pm.instruments.append(instrument)
        for ts in sequence.slice_time_signatures(max_event_time):
            pm.time_signature_changes.append(
                pretty_midi.TimeSignature(
                    ts.numerator,
                    ts.denominator,
                    ts.midi_time)
            )

        # populate key signatures.
        for sk in sequence.slice_key_signatures(max_event_time):
            key_number = sk.midi_key
            if sk.mode == KeySignatureType.MINOR:
                key_number += 12
            pm.key_signature_changes.append(
                pretty_midi.KeySignature(key_number, sk.midi_time)
            )

        # populate tempos.
        for seq_tempo in sequence.slice_tempo_signature(max_event_time):
            tick_scale = 60.0 / (pm.resolution * seq_tempo.qpm)
            tick = pm.time_to_tick(seq_tempo.midi_time)
            pm.update_scale((tick, tick_scale))
            pm.update_tick_to_time(0)

        # populate instrument names by first creating an instrument map between
        # instrument index and name.
        # Then, going over this map in the instrument event for loop
        inst_infos = {}
        for i, inst_info in enumerate(sequence.instruments):
            inst_infos[inst_info.instrument_num] = inst_info.name

        for seq in sequence:
            notes = [pretty_midi.Note(n.velocity, n.pitch, n.start_time, n.end_time) for n in seq.notes]
            pitch_bends = [pretty_midi.PitchBend(sb.amount, sb.midi_time) for sb in seq.pitch_bends
                           if max_event_time and sb.midi_time < max_event_time]
            control_changes = [pretty_midi.ControlChange(cc.cc_number, cc.cc_value, cc.cc_time)
                               for cc in seq.control_changes
                               if max_event_time and cc.cc_time < max_event_time]

            instrument = pretty_midi.Instrument(seq.program, seq.instrument.is_drum)
            instrument.notes = notes
            instrument.pitch_bends = pitch_bends
            instrument.control_changes = control_changes
            pm.instruments.append(instrument)

        return
