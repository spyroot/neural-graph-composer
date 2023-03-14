spss = [1, 2, 4, 8, 16, 32]
start = 0.2
end = 1.4
amount = 0.5

# note = MidiNote(pitch=60, velocity=100, start_time=0.2, end_time=1.4)

# for sps in spss:
#     print(sps, start / (1.0 / sps) + (1.0 - amount), end / (1.0 / sps) + (1.0 - amount))

import math


def compute_nearest_steps(step_per_sec, start_time, stop_time, cutoff=1.5, tempo=120):
    # Calculate adjusted MIDI start time in seconds based on step per second
    midi_start_time = start_time

    # Calculate adjusted MIDI stop time in seconds based on step per second
    midi_stop_time = stop_time

    # Calculate duration in seconds
    duration = midi_stop_time - midi_start_time

    # Calculate total number of steps that can fit into the duration
    total_steps = duration * step_per_sec

    # Calculate nearest step based on cutoff
    nearest_step = math.floor(total_steps / cutoff) * cutoff

    # Calculate nearest start step
    nearest_start_step = math.floor(midi_start_time * step_per_sec / cutoff) * cutoff

    # Calculate nearest stop step
    nearest_stop_step = round((midi_stop_time) * step_per_sec / cutoff) * cutoff
    return (nearest_start_step, nearest_stop_step)


start_time = 0.2
stop_time = 1.4
cutoff = 0.5
step_per_sec = 32

nearest_steps = compute_nearest_steps(step_per_sec, start_time, stop_time, cutoff)
print(nearest_steps)

# Output: (2.0, 11.0)


#
# nearest_steps = compute_nearest_steps(step_per_sec, start_time, stop_time)
# print(nearest_steps)  # Output: (2, 10)


# for sps in spss:
#     print(compute_nearest_steps(sps, 0.2, 1.4))

# sps 1
# note.quantize_in_place(1)
# self.assertEqual(note.start_time, 0)
# self.assertEqual(note.end_time, 1)
# sps 2
# self.assertEqual(note.quantized_start_step, 0)
# self.assertEqual(note.quantized_end_step, 3)
# sps 4
# self.assertEqual(note.quantized_start_step, 1)
# self.assertEqual(note.quantized_end_step, 6)
# sps 8
# self.assertEqual(note.quantized_start_step, 2)
# self.assertEqual(note.quantized_end_step, 12)
# sps 16
# self.assertEqual(note.quantized_start_step, 3)
# self.assertEqual(note.quantized_end_step, 24)

# self.assertEqual(note.quantized_start_step, 6)
# self.assertEqual(note.quantized_end_step, 32)
#
# import math
#
# step_per_sec = 1
note_start_time = 0.2
note_stop_time = 1.4
#
# # Calculate the start and stop steps based on the step per second value
# start_step = round(note_start_time * step_per_sec)
# stop_step = round(note_stop_time * step_per_sec)
#
# print("Start step: ", start_step)
# print("Stop step: ", stop_step)
#

# import math
#
# step_per_sec = 32
# note_start_time = 1.0
# note_stop_time = 10.0
# cutoff = 0.5
#
# # Calculate the duration of the note in seconds
# note_duration = note_stop_time - note_start_time
#
# # Calculate the total number of steps that can fit into the note duration at the given step per second value
# total_steps = note_duration * step_per_sec
#
# # Calculate the nearest step based on the given cutoff value
# nearest_step = math.floor(total_steps / cutoff) * cutoff
#
# # Calculate the nearest start step based on the given cutoff value
# nearest_start_step = math.floor(note_start_time * step_per_sec / cutoff) * cutoff


import math
import math

step_per_sec = 1
note_start_time = 1
note_stop_time = 1.6
cutoff = 0.5

# Calculate the start and stop steps based on the step per second value
start_step = round(note_start_time * step_per_sec / cutoff) * cutoff
stop_step = round(note_stop_time * step_per_sec / cutoff) * cutoff

print("Start step: ", start_step)
print("Stop step: ", stop_step)