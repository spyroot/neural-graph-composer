"""
All test utils.
Generators of data etc.

Author Mus spyroot@gmail.com
           mbayramo@stanford.edu.
"""

import random
import tempfile
from typing import Optional, Generator, Tuple, List, Dict

from neural_graph_composer.custom_pretty_midi import CustomPrettyMIDI
from neural_graph_composer.midi.midi_note import MidiNote
from neural_graph_composer.midi.midi_sequence import MidiNoteSequence
from neural_graph_composer.midi.midi_instruments import MidiInstrumentInfo
import mido


def mido_to_pretty_midi(mido_file: mido.MidiFile) -> CustomPrettyMIDI:
    """
    :param mido_file:
    :return:
    """
    with tempfile.NamedTemporaryFile() as temp_file:
        mido_file.save(temp_file.name)
        pretty_midi_file = CustomPrettyMIDI(temp_file.name)
    return pretty_midi_file


def generate_scale_dictionary() -> Dict[str, List[int]]:
    # [2, 2, 1, 2, 2, 2, 1]
    scales = {
        'chromatic': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "major": [0, 2, 4, 5, 7, 9, 11],
        "natural_minor": [0, 2, 3, 5, 7, 8, 10],
        "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
        "melodic_minor": [0, 2, 3, 5, 7, 9, 11],
        "dorian": [0, 2, 3, 5, 7, 9, 10],
        "phrygian": [0, 1, 3, 5, 7, 8, 10],
        "lydian": [0, 2, 4, 6, 7, 9, 11],
        "mixolydian": [0, 2, 4, 5, 7, 9, 10],
        "locrian": [0, 1, 3, 5, 6, 8, 10],
        'whole_tone': [0, 2, 4, 6, 8, 10],
        'blues': [0, 3, 5, 6, 7, 10],
        'pentatonic_major': [0, 2, 4, 7, 9],
        'pentatonic_minor': [0, 3, 5, 7, 10],
        'spanish': [0, 1, 3, 4, 5, 6, 8, 10],
        'egyptian': [0, 2, 5, 7, 10]
    }
    scale_dict = {scale_name: [i + 12 * j for j in range(8) for i in scales[scale_name]] for scale_name in scales}
    return scale_dict


def generate_triads(
        num_triads: int,
        octave: int,
        start_pitch: Optional[int] = 12,
        scale: Optional[List[int]] = [0, 2, 4, 5, 7, 9, 11],
        note_shift: Optional[int] = 0,
        start_time: Optional[float] = 0.0,
        duration: Optional[float] = 0.5,
        end_time: Optional[float] = None,
        instrument_id: Optional[int] = 0) -> Generator[Tuple[MidiNote, MidiNote, MidiNote], None, None]:
    """
    Generates a list of random triads as tuples of MidiNote's (root note, third, fifth).

    :param duration:
    :param note_shift: shift in semitones for each successive triad.
    :param num_triads: the number of triads to generate
    :param octave: the octave range of the generated notes
    :param start_pitch: the starting pitch of the generated notes within the octave
    :param start_time: the start time of the first triad in the sequence (default: 0.0)
    :param end_time: the end time of the last triad in the sequence (default: None)
    :param scale: the scale to generate the triads within (a list of semitone intervals from the root note)
    :param instrument_id:  if we provide we use this as instrument id.
    :return: a list of triads as tuples of (root note, third, fifth)

    for triad in generate_triads(num_triads=5, octave=4, start_pitch=60, scale=[0, 2, 4, 5, 7, 9, 11]):
    print(triad)

    # Example usage:
    # Generate 5 triads starting at pitch 60 and shifted by 2 semitones for each successive triad.
    # The notes in each triad will have random start and end times within the given duration.

    triads = generate_triads(num_triads=5, octave=4,
    start_pitch=60, note_shift=2, duration=1.0, scale=[0, 2, 4, 5, 7, 9, 11])
    for triad in triads:
        print(triad)

    # Example usage:
        # Generate 5 triads starting at pitch 60 and shifted by 2 semitones for each successive triad.
        # The notes in each triad will have random start and end times within the given duration.

    triads = generate_triads(num_triads=5, octave=4, start_pitch=60, note_shift=2, duration=1.0)
    for triad in triads:
        print(triad)

    """
    if end_time is None:
        total_duration = float(num_triads * duration)
    else:
        total_duration = end_time - start_time

    for i in range(num_triads):
        root_pitch = start_pitch + note_shift + scale[(i * 3) % len(scale)]
        root_pitch = min(max(root_pitch, 0), 127)

        if root_pitch > start_pitch + 6:
            root_pitch -= 12

        root_pitch += 12 * octave
        third_pitch = root_pitch + scale[2]
        fifth_pitch = root_pitch + scale[4]

        inversion = random.randint(0, 2)
        if inversion == 1:
            root_pitch += 12
        elif inversion == 2:
            root_pitch += 24

        start = start_time + total_duration * i / num_triads
        end = start + duration / num_triads
        velocity = random.randint(1, 127)

        yield (MidiNote(root_pitch, start, end, velocity, instrument=instrument_id),
               MidiNote(third_pitch, start, end, velocity, instrument=instrument_id),
               MidiNote(fifth_pitch, start, end, velocity, instrument=instrument_id))


def generate_notes(num_notes: int, octave: int, start_pitch: int,
                   start_time: float = 0.0, end_time: Optional[float] = None) -> Generator[MidiNote, None, None]:
    """
    Generates a list of random MidiNote objects.

    :param num_notes: the number of MidiNote objects to generate
    :param octave: the octave range of the generated notes
    :param start_pitch: the starting pitch of the generated notes within the octave
    :param start_time: the start time of the first note in the sequence (default: 0.0)
    :param end_time: the end time of the last note in the sequence (default: None)
    :return: a list of MidiNote objects

    Example:
        To generate 4 MidiNotes in octave 4 starting from pitch
        60 with start time of 0.0 and end time of 4.0:
        ```
        notes = list(generate_notes(num_notes=4, octave=4, start_pitch=60, start_time=0.0, end_time=4.0))
        ```
    """
    if end_time is None:
        duration = float(num_notes)
    else:
        duration = end_time - start_time

    for i in range(num_notes):
        pitch = start_pitch + i % 12
        if pitch > start_pitch + 6:
            pitch -= 12
        pitch += 12 * octave
        start = start_time + duration * i / num_notes
        end = start + duration / num_notes
        velocity = random.randint(1, 127)
        note = MidiNote(pitch, start, end, velocity)
        yield note


def generate_midi_sequence(
        num_notes: int = 10,
        min_pitch: int = 60,
        max_pitch: int = 72,
        velocity: int = 64,
        duration: float = 0.5,
        random_start_end_time: bool = False,
        resolution: int = 220,
        is_drum: bool = False,
        instrument_number: int = 0,
        instrument_name: str = 'Acoustic Grand Piano'
) -> MidiNoteSequence:
    """
    Generate MidiNoteSequence for unit test.  Each MidiNoteSequence
    contains num_notes.
    :param num_notes:
    :param min_pitch:
    :param max_pitch:
    :param velocity:
    :param duration:
    :param random_start_end_time:
    :param resolution:
    :param is_drum:
    :param instrument_number:
    :param instrument_name:
    :return:

    Example:
        Generate a MidiNoteSequence with 20 notes, with random pitches between 40 and 80,
        velocity of 100, duration of 2.0 seconds, and non-drum instrument.
        ```
        midi_seq = generate_midi_sequence(
        num_notes=20, min_pitch=40, max_pitch=80,
        velocity=100, duration=2.0, is_drum=False
        )
        ```
    """
    notes = []
    for i in range(num_notes):
        pitch = random.randint(min_pitch, max_pitch)
        start_time = random.uniform(0, duration) if random_start_end_time else (i * duration)
        end_time = random.uniform(start_time, duration) if random_start_end_time else ((i + 1) * duration)
        note = MidiNote(pitch=pitch, start_time=start_time, end_time=end_time, velocity=velocity, is_drum=is_drum)
        notes.append(note)

    instrument_info = MidiInstrumentInfo(instrument_number, instrument_name)
    midi_seq = MidiNoteSequence(notes=notes, instrument=instrument_info, resolution=resolution)
    return midi_seq
