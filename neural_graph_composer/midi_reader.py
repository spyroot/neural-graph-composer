"""
Main interface consumed to get MIDI information from file to internal representation.
Currently, it only supports PrettyMIDI via customized implementation CustomPrettyMIDI
It reads and converts MIDI files to an internal representation.


Author Mus spyroot@gmail.com
"""
import io
import logging
import os
import tempfile
from typing import Tuple, Iterator, Optional, Union

import mido
import numpy as np
from mido import MidiFile
from pretty_midi import pretty_midi, Instrument

from .custom_pretty_midi import CustomPrettyMIDI
from .midi.abstract_midi_reader import MidiBaseReader
from .midi.midi_control_change import MidiControlChange
from .midi.midi_note import MidiNote
from .midi.midi_pitch_bend import MidiPitchBend
from .midi.midi_sequence import MidiNoteSequence
from .midi.midi_sequences import MidiNoteSequences
from .midi.midi_spec import DEFAULT_PPQ
from .midi.midi_time_signature import MidiTempoSignature, MidiTimeSignature
from .midi.midi_key_signature import MidiKeySignature, KeySignatureType


class MidiReaderError(Exception):
    pass


class MidiReader(MidiBaseReader):
    """
    This is a Python module that contains a MidiReader class which provides methods
    to read a MIDI file and convert it to an internal representation
    (MidiNoteSequences). The module uses the pretty_midi
    library to read MIDI files.

    It can extract information about the MIDI file, such as the tempo, time signature, and key signature.
    It can also extract the notes and chords played in the MIDI file and their timing information.
    """

    def __init__(self, is_debug: Optional[float] = bool):
        """
        Midi Reader, read midi file via CustomPrettyMIDI interface and construct
        internal representations.

        The mean interface client need consume read()
        """
        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)
        self.logger.setLevel(logging.WARNING)
        self._callback = []
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
                MidiTempoSignature(time_in_seconds.astype(float),
                                   qpm=tempo_in_qpm.astype(float),
                                   resolution=midi.resolution)
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
        """Read time signature from pretty midi and constructs internal
        representation in midi_seq.
        :param midi_seq:
        :param midi:
        :return:
        """
        logging.info("Reading time signature changes.")
        for sc in midi.time_signature_changes:
            try:
                logging.debug(f"Reading time signature changes "
                              f"numerator: {sc.numerator} denominator: {sc.denominator} time {sc.time}.")
                midi_seq.add_time_signatures(
                    MidiTimeSignature(
                        numerator=sc.numerator,
                        denominator=sc.denominator,
                        midi_time=sc.time
                    )
                )
            except ValueError:
                raise MidiReaderError(
                    f"Invalid time signature denominator {sc.denominator}")

    @staticmethod
    def _read_pretty_midi(seq_file: Union[str, bytes]) -> CustomPrettyMIDI:
        """Read a MIDI file with the given file path or bytes-like
        object and return a CustomPrettyMIDI object.

        If the input is a file path, the file will be opened and read as
        binary data. If the input is a bytes-like object
        a bytes stream will be created from it and read as binary data.

        :param seq_file: A file path or bytes-like object to read the MIDI file from.
        :return: A CustomPrettyMIDI object representing the contents of the MIDI file.
                            Note it important since regular Pretty Midi will not work.
        :raises MidiParserError: If an error occurs while parsing the MIDI file.
        """
        if isinstance(seq_file, bytes):
            file_data = io.BytesIO(seq_file)
        else:
            file_data = seq_file

        try:
            midi_data = CustomPrettyMIDI(file_data)
        except Exception as err:
            logging.error('Error reading MIDI file %s: %s', seq_file, err)
            raise err

        return midi_data

    @staticmethod
    def read_instrument(
            midi_seq: MidiNoteSequences,
            midi_data: CustomPrettyMIDI) -> Iterator[Tuple[Instrument, int, pretty_midi.Note, int]]:
        """Generator read midi file and returns MIDI program,
        midi idx midi instrument drum or not,  note
        :param midi_seq:
        :param midi_data:
        :return:  Tuple where first element is Instrument,
        """
        for i, instrument in enumerate(midi_data.instruments):
            # for each instrument iterate over all node and get time
            seq_id = midi_seq.create_track(i, instrument.program, instrument.name, instrument.is_drum)
            for n in instrument.notes:
                # update total time
                if not midi_seq.total_time or n.end >= midi_seq[seq_id].total_time:
                    # update entire total for all music instruments
                    if n.end >= midi_seq.total_time:
                        midi_seq.total_time = n.end
                    # update for particular instrument total time
                    midi_seq[seq_id].total_time = n.end
                    yield instrument, i, n, seq_id
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
            midi_data: CustomPrettyMIDI) -> Iterator[Tuple[Instrument, int, pretty_midi.PitchBend, int]]:
        """Generator read midi file and returns
        MIDI program, midi idx midi instrument drum or not,  note

        :param midi_seq:The MidiNoteSequence to which key signatures will be added.
        :param midi_data: from which key signature changes will be read.
        :return:
        """
        logging.info("Reading pitch information.")
        for i, instrument in enumerate(midi_data.instruments):
            seq_id = midi_seq.create_track(i, instrument.program, instrument.name, instrument.is_drum)
            # for each instrument iterate over all node and get time
            for n in instrument.pitch_bends:
                if not midi_seq.total_time or n.end >= midi_seq.total_time:
                    yield instrument, i, n, seq_id

    @staticmethod
    def read_instrument_control_changes(
            midi_seq: MidiNoteSequences,
            midi_data: CustomPrettyMIDI) -> Iterator[Tuple[Instrument, int, pretty_midi.ControlChange, int]]:
        """Generator read midi file and returns
        MIDI program, midi idx midi instrument drum or not,  note

        :param midi_seq: The MidiNoteSequence to which key signatures will be added.
        :param midi_data: from which key signature changes will be read.
        :return:
        """
        logging.info("Reading control change information.")
        for i, instrument in enumerate(midi_data.instruments):
            seq_id = midi_seq.create_track(i, instrument.program, instrument.name, instrument.is_drum)
            for cv in instrument.control_changes:
                if not midi_seq.total_time or cv.time >= midi_seq.total_time:
                    yield instrument, i, cv, seq_id

    @staticmethod
    def mido_to_pretty_midi(mido_file: mido.MidiFile) -> CustomPrettyMIDI:
        """Converts a mido.MidiFile object to a pretty_midi.PrettyMIDI object.
        :param mido_file: The mido.MidiFile to convert.
        :return: The resulting pretty_midi.PrettyMIDI object.
        """
        with tempfile.NamedTemporaryFile() as temp_file:
            mido_file.save(temp_file.name)
            pretty_midi_file = CustomPrettyMIDI(temp_file.name)
        return pretty_midi_file

    @staticmethod
    def read(seq_file: Union[str, bytes, MidiFile], is_debug: Optional[bool] = True) -> MidiNoteSequences:
        """This method takes either a path to a MIDI file or a `mido.MidiFile`
          converts to internal representation MidiNoteSequences

        Each MidiNoteSequences object contains one MidiNoteSequence
        object for each instrument in the MIDI file.

        :param seq_file:  Path to a MIDI file, a `bytes` object representing the contents of a MIDI file, or a
                     `MidiFile` object.
        :param is_debug: A path to a MIDI file or a `mido.MidiFile` object.

        :return: The resulting `MidiNoteSequences`
        :rtype: MidiNoteSequences
        """
        if isinstance(seq_file, str) or isinstance(seq_file, bytes):
            midi_data = MidiReader._read_pretty_midi(seq_file)
        elif isinstance(seq_file, MidiFile):
            midi_data = MidiReader.mido_to_pretty_midi(seq_file)
        else:
            raise ValueError(
                f"Illegal argument. input seq must be a path to a file or an instance of "
                f"mido.MidiFile, but got {type(seq_file)}")

        midi_data.ticks_per_quarter = midi_data.resolution
        logging.debug(
            f"ticks per quarter: {midi_data.ticks_per_quarter} "
            f"resolution: {midi_data.resolution}")

        # midi_seq.filename = seq_file
        midi_seqs = MidiNoteSequences(is_debug=is_debug, resolution=midi_data.ticks_per_quarter)
        # read key signature , tempo signature and time signature.
        # for not we assume it type-0 MIDI.
        MidiReader.read_key_signature(midi_seqs, midi_data)
        MidiReader.read_tempo_changes(midi_seqs, midi_data)
        MidiReader.read_time_signature(midi_seqs, midi_data)

        logging.debug(f"num of key signatures {len(midi_seqs.key_signatures)} "
                      f"num time signatures {len(midi_seqs.time_signatures)} "
                      f"num tempo signature {len(midi_seqs.tempo_signatures)}")

        midi_seqs.filename = seq_file
        midi_seqs.resolution = midi_data.resolution

        # read all instruments and construct midi seq
        # all instruments are merged
        for instrument, instrument_idx, note, seq_id in MidiReader.read_instrument(midi_seqs, midi_data):
            current_seq = midi_seqs.get_instrument(
                seq_id, instrument.name, instrument.program, instrument.is_drum)
            # read
            num, denom = midi_seqs.time_signature(note.start)
            current_seq.add_note(
                MidiNote(
                    note.pitch,
                    note.start,
                    note.end,
                    program=instrument.program,
                    instrument=instrument.program,
                    velocity=note.velocity,
                    is_drum=instrument.is_drum,
                    numerator=num,
                    denominator=denom,
                )
            )

        for instrument, instrument_idx, b, seq_id in MidiReader.read_instrument_pitch_bends(midi_seqs, midi_data):
            current_seq = midi_seqs.get_track(
                seq_id, instrument.name, instrument.program, instrument.is_drum)
            current_seq.add_pitch_bends(
                MidiPitchBend(
                    b.pitch,
                    midi_time=b.time,
                    program=instrument.program,
                    instrument=instrument_idx,
                    is_drum=instrument.is_drum
                )
            )

        for instrument, instrument_idx, cv, seq_id in MidiReader.read_instrument_control_changes(midi_seqs, midi_data):
            current_seq = midi_seqs.get_track(
                seq_id, instrument.name, instrument.program, instrument.is_drum)
            current_seq.add_control_changes(
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
    def to_pretty_midi(
            sequence: MidiNoteSequences,
            from_start: Optional[float] = None,
            to_end: Optional[float] = None) -> CustomPrettyMIDI:
        """Take MidiNoteSequences and return CustomPrettyMIDI.
        :param sequence: a sequence that we're converting to PrettyMIDI.
        :param from_start: The time to start writing the MIDI data from. If None, the start time is 0.
        :param to_end: The time to stop writing the MIDI data at. If None, the end time is the time of the last event.
        :return: CustomPrettyMIDI
        :rtype CustomPrettyMIDI
        """
        ticks_per_quarter = sequence.resolution or DEFAULT_PPQ
        max_event_time = sequence.truncate(to_end)
        initial_seq_tempo = sequence.initial_tempo()

        logging.debug(f"{initial_seq_tempo} {ticks_per_quarter} {max_event_time}")

        pm = CustomPrettyMIDI(
            resolution=ticks_per_quarter,
            initial_tempo=initial_seq_tempo)

        instrument = pretty_midi.Instrument(0)
        pm.instruments.append(instrument)
        for ts in sequence.slice_time_signatures(max_event_time):
            pm.time_signature_changes.append(
                pretty_midi.TimeSignature(
                    ts.numerator,
                    ts.denominator,
                    ts.midi_time)
            )

        #  all key signatures.
        for sk in sequence.slice_key_signatures(max_event_time):
            key_number = sk.midi_key
            if sk.mode == KeySignatureType.MINOR:
                key_number += 12
            pm.key_signature_changes.append(
                pretty_midi.KeySignature(key_number, sk.midi_time)
            )

        # write all tempo signature, TODO add meta
        for seq_tempo in sequence.slice_tempo_signature(max_event_time):
            tick_scale = 60.0 / (pm.resolution * seq_tempo.qpm)
            print(f"Setting scale {type(seq_tempo.midi_time)}")
            tick = pm.time_to_tick(seq_tempo.midi_time)
            pm.update_scale((tick, tick_scale))
            pm.update_tick_time([0])

        inst_infos = {}
        for i, inst_info in enumerate(sequence.instruments):
            inst_infos[inst_info.instrument_num] = inst_info.name

        for k in sequence:
            # read all notes from internal representation to pretty midi
            notes = [
                pretty_midi.Note(n.velocity, n.pitch, n.start_time, n.end_time)
                for n in sequence[k].notes]
            pitch_bends = [
                pretty_midi.PitchBend(sb.amount, sb.midi_time)
                for sb in sequence[k].pitch_bends
                if max_event_time and sb.midi_time < max_event_time
            ]

            control_changes = [
                pretty_midi.ControlChange(cc.cc_number, cc.cc_value, cc.cc_time)
                for cc in sequence[k].control_changes
                if max_event_time and cc.cc_time < max_event_time]

            # create instrument
            instrument = pretty_midi.Instrument(
                sequence[k].program, sequence[k].instrument.is_drum
            )
            instrument.notes = notes
            instrument.pitch_bends = pitch_bends
            instrument.control_changes = control_changes
            pm.instruments.append(instrument)

        if from_start is not None or to_end is not None:
            original_times = np.array([0, pm.get_end_time()])
            new_times = np.array([from_start if from_start is not None else 0,
                                 to_end if to_end is not None else pm.get_end_time()])
            new_times = new_times.clip(0, pm.get_end_time())
            pm.adjust_times(original_times, new_times)

        logging.debug(f"pretty midi time: {pm.get_end_time()} "
                      f"estimate_tempo: {pm.estimate_tempo()} "
                      f"resolution: {pm.resolution}")
        return pm

    @staticmethod
    def write(sequence: MidiNoteSequences,
              filename: str,
              from_start: Optional[float] = None,
              to_end: Optional[float] = None,
              do_overwrite: Optional[float] = False) -> None:
        """Write sequence to a midi file.

        :param sequence:  a MidiNoteSequences that we want bounce to a file.
        :param do_overwrite:  if we want to overwrite.  this mainly for unit testing.
        :param filename: path to a file. Note we never overwrite unless overwrite provided.
        :param from_start: By default, we write from the start til the end.
        :param to_end:  Till the last midi event.
        :raises FileExistsError: If a file with the same name already exists and do_overwrite is False.
        :raises TypeError: If from_start or to_end are not numeric values.
        :raises ValueError: If the sequence is not a valid MidiNoteSequences or the filename is invalid.
         :return: None
        """
        if not isinstance(sequence, MidiNoteSequences) or not sequence:
            raise ValueError("The sequence must be instance of MidiNoteSequences")

        if not isinstance(filename, str) or not filename:
            raise ValueError("The provided filename is invalid. It should be a non-empty string.")

        if do_overwrite is False and os.path.exists(filename):
            raise FileExistsError(
                f"A file with the name '{filename}' "
                f"already exists. Aborting to avoid overwriting.")

        if from_start is not None and (not isinstance(from_start, (int, float)) or from_start < 0):
            raise ValueError("The 'from_start' parameter should be a non-negative number.")
        if to_end is not None and (not isinstance(to_end, (int, float)) or to_end < 0):
            raise ValueError("The 'to_end' parameter should be a non-negative number.")
        if from_start is not None and to_end is not None and from_start > to_end:
            raise ValueError("The 'from_start' parameter should be less than or equal to 'to_end'.")

        pretty_midi_seq = MidiReader.to_pretty_midi(sequence, from_start=from_start, to_end=to_end)
        pretty_midi_seq.write(filename)

    @classmethod
    def from_file(cls, file_path: str) -> MidiNoteSequences:
        """
        :param file_path:
        :return:
        """
        try:
            return cls.read(file_path)
        except Exception as e:
            raise ValueError(f"Unable to load file {file_path}") from e

