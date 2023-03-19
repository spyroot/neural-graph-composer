import numpy as np


def test_grid_based():
    """

    :return:
    """
    # Example list of MIDI pitches
    midi_pitches = [
        {'pitch': 60, 'start': 0.0, 'stop': 1.0},
        {'pitch': 62, 'start': 0.5, 'stop': 2.0},
        {'pitch': 64, 'start': 1.0, 'stop': 1.5},
        {'pitch': 67, 'start': 2.0, 'stop': 3.0},
        {'pitch': 69, 'start': 3.5, 'stop': 4.5},
        {'pitch': 71, 'start': 4.0, 'stop': 5.0}
    ]

    # determine the minimum and maximum start times and durations of the MIDI pitches
    min_start_time = min([p['start'] for p in midi_pitches])
    max_stop_time = max([p['stop'] for p in midi_pitches])
    max_duration = max([p['stop'] - p['start'] for p in midi_pitches])

    # we use 8 steps by default, define the time steps for the grid
    time_step = max_duration / 8

    # Create the 2D array to represent the grid
    num_pitch_steps = 128  # example: use 128 pitch steps

    #  determine the number of time steps
    num_time_steps = int(np.ceil((max_stop_time - min_start_time) / time_step))
    grid = np.zeros((num_pitch_steps, num_time_steps))

    # snap each MIDI pitch list and assign each pitch to the grid
    for pitch in midi_pitches:
        pitch_step = pitch['pitch']
        start_time_step = int(np.floor((pitch['start'] - min_start_time) / time_step))
        stop_time_step = int(np.ceil((pitch['stop'] - min_start_time) / time_step))
        grid[pitch_step, start_time_step:stop_time_step] = 1
