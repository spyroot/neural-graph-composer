"""
Const represent all constant used as part of MIDI specs.

Author
Mus spyroot@gmail.com
    mbayramo@stanford.edu
"""

MIN_MIDI_VELOCITY = 1
MAX_MIDI_VELOCITY = 127

MIN_MIDI_PROGRAM = 0
MAX_MIDI_PROGRAM = 127

PIANO_MIN_MIDI_PITCH = 21
PIANO_MAX_MIDI_PITCH = 108

MIDI_PITCHES = 128

MIN_MELODY_EVENT = -2
MAX_MELODY_EVENT = 127

MIN_MIDI_PITCH = 0
MAX_MIDI_PITCH = 127

DEFAULT_MIN_MIDI_CC = 0
DEFAULT_MAX_MIDI_CC = 127

NOTES_PER_OCTAVE = 12
NOTES_PER_OCTAVE = NOTES_PER_OCTAVE
NUM_MIDI_PITCHES = MAX_MIDI_PITCH - MIN_MIDI_PITCH + 1

REFERENCE_NOTE = 69  # MIDI note value of A4
STANDARD_FREQ = 440  # Standard frequency of A4 in Hz
SEMITONES_PER_OCTAVE = 12  # Number of semitones in an octave

# mapping from some name to CC
# on reverse we construct map that hold reverse
# this mainly for construct CC to name.
# each value is either max or percentage
# from collections import namedtuple

# 3, 9 14-15, 20-31, 85-90 , 102-119
MIDI_UNDEFINED_CC = x = set(range(14, 16)).union(set(range(20, 32))).union(range(85, 91)).union(
    range(102, 120)).union([3, 9])

CONTROL_AND_PROGRAMS = (list(range(96, 104)) + list(range(112, 120)) + list(range(120, 128)))

DEFAULT_MIDI_STEPS_PER_BAR = 16
DEFAULT_MIDI_STEPS_PER_QUARTER = 4

# PPQ stands for "pulses per quarter note" a
# nd refers to the resolution or granularity
# of a MIDI sequence. It represents the number of clock ticks per quarter note.
# The PPQ value is used to determine the timing and duration of MIDI events in a sequence.
DEFAULT_PPQ = 480

# QPM stands for "quarter notes per minute". It is a measure of tempo in music,
# representing the number of quarter notes that would be played in one minute.
DEFAULT_QPM = 120
DEFAULT_MIDI_QUARTERS_PER_MINUTE = 120.0

# Default absolute quantization.
DEFAULT_STEPS_PER_SECOND = 100

# semi_quaver=
# quaver = 1/2
# crochet = 1
# minim = 2
# semibreve = 4

# Table of frequencies for each pitch
PITCH_FREQUENCIES = {
    'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11
}
