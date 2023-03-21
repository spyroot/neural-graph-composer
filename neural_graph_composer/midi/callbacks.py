"""
Callback used in different spots.
Author Mus spyroot@gmail.com
"""
from typing import List
from pretty_midi import pretty_midi

from neural_graph_composer.midi.midi_note import MidiNote
from neural_graph_composer.midi.midi_instruments import MidiInstrumentInfo


def save_midi_callback(file_path: str,
                       instrument: MidiInstrumentInfo,
                       notes: List[MidiNote],
                       total_time: float,
                       tempo: int = 120.,
                       resolution: int = 220):
    """
    :param tempo:  The tempo (in BPM) to use for the MIDI file. Default 120.0 (match Pretty MIDI)
    :param resolution:  The resolution (in ticks per beat) to use for the MIDI file. Default 220 (match Pretty MIDI)
    :param file_path: The file path to save the MIDI file to.
    :param instrument:  An object containing information about the MIDI instrument.
    :param notes:  A list of MidiNote objects representing the notes to be written to the MIDI file.
    :param total_time: The total time of the MIDI sequence in seconds.
    :raise  ValueError: If the notes list is empty or None, or if the instrument object is None.
    :return:
    """
    if not notes:
        raise ValueError("The notes list is empty or None")

    if instrument is None:
        raise ValueError("The instrument object is None")

    midi = pretty_midi.PrettyMIDI(
        initial_tempo=tempo,
        resolution=resolution)

    default = pretty_midi.Instrument(program=0)
    midi.instruments.append(default)

    pm_instrument = pretty_midi.Instrument(
        program=instrument.instrument_num,
        is_drum=instrument.is_drum,
        name=instrument.name)

    pm_notes = [pretty_midi.Note(
        velocity=note.velocity,
        pitch=note.pitch,
        start=note.start_time,
        end=note.end_time) for note in notes]

    pm_instrument.notes.extend(pm_notes)
    midi.instruments.append(pm_instrument)
    midi.write(file_path)
