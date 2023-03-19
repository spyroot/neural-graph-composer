import collections
import random

from sortedcontainers import SortedDict

from neural_graph_composer.midi.Interval import Interval
from neural_graph_composer.midi.midi_instruments import MidiInstrumentInfo
from neural_graph_composer.midi.midi_note import MidiNote
from neural_graph_composer.midi.midi_sequence import MidiNoteSequence
from neural_graph_composer.midi.midi_sequences import MidiNoteSequences
from neural_graph_composer.midi.sorted_dict import SortedDict


def interval_test():
    a = Interval(3, 3)
    b = Interval(3, 5)
    c = Interval(2, 3)

    print(max(a, b))
    print(min(a, b))
    print(a.overlaps(b))
    print(b.overlaps(a))
    print(c.overlaps(a))
    print(a.overlaps(c))

    print(a.intersection(b))
    print(b.intersection(a))
    print(a.intersection(c))
    print(c.intersection(a))


def negative_interval():
    a = Interval(-5, -2)
    b = Interval(-5, -1)
    c = Interval(-6, -1)
    d = Interval(-4, -1)

    print(a < b)
    print(b < a)
    print(c < a)
    print(d < a)

    print(a.overlaps(b))
    print(b.overlaps(a))


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
    Generate note for unit test
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


def instruments():
    midi_seq1 = generate_midi_sequence(num_notes=10, random_start_end_time=True, instrument_number=0)
    midi_seq2 = generate_midi_sequence(num_notes=10, random_start_end_time=True, instrument_number=1)
    midi_seq3 = generate_midi_sequence(num_notes=10, random_start_end_time=True, instrument_number=2)
    seq_list = [midi_seq2, midi_seq1, midi_seq3]  # intentionally out of order

    test = MidiNoteSequences(midi_seq=seq_list)
    print(test.instruments)
    # if midi_seq is None:
    #     self.midi_seqs = SortedDict()
    # else:
    #     self.midi_seqs = SortedDict()
    #     if isinstance(midi_seq, MidiNoteSequence):
    #         self.midi_seqs[midi_seq.instrument.instrument_num] = midi_seq
    #     elif isinstance(midi_seq, list):
    #         for seq in midi_seq:
    #             if seq.instrument.instrument_num not in self.midi_seqs:
    #                 self.midi_seqs[seq.instrument.instrument_num] = seq
    #     else:
    #         raise TypeError("midi_seq must be either a MidiNoteSequence or a list of MidiNoteSequence")


def interval_tree_test():
    """
    :return:
    """
    sid = SortedDict()
    sid[Interval(3, 3)] = 'c'
    sid[Interval(2, 3)] = 'b'
    sid[Interval(3, 4)] = 'd'
    sid[Interval(1, 1)] = 'a'

    print(sid.values())

    test_slice = SortedDict()
    test_slice[Interval(0, 5)] = 'a'
    test_slice[Interval(3, 7)] = 'b'
    test_slice[Interval(6, 10)] = 'c'
    test_slice[Interval(6, 8)] = 'd'
    test_slice[Interval(9, 12)] = 'e'
    test_slice[Interval(5, 6)] = 'xx'

    # print(test_slice)

    for interval in test_slice:
        print("Intervals", interval)

    print(test_slice.keys())
    test_slice[Interval(6, 10)] = 'cc'
    print(test_slice.values())

    del test_slice[Interval(6, 10)]
    print(test_slice.values())

    del test_slice[Interval(5, 6)]
    print(test_slice.values())


if __name__ == '__main__':
    interval_tree_test()
    instruments()

    # negative_interval()
    # interval_test()
    # interval_tree_test()
    #
    # # create an empty SortedIntervalDict
    # my_dict = SortedIntervalDict()
    #
    # # add an interval and value to the dictionary
    # my_dict[Interval(0, 4)] = "value1"
    # my_dict[Interval(4, 8)] = "value2"
    # my_dict[Interval(8, 10)] = "value3"
    #
    # # get the value associated with an interval
    # value = my_dict[Interval(0, 4)]
    #
    # # iterate over the intervals in the dictionary
    # for interval in my_dict:
    #     print(interval)
    #
    # # check if an interval is in the dictionary
    # if Interval(0, 4) in my_dict:
    #     print("Interval is in the dictionary")
    #
    # test_slice = SortedIntervalDict()
    # test_slice[Interval(0, 5)] = 'a'
    # test_slice[Interval(3, 7)] = 'b'
    # test_slice[Interval(6, 10)] = 'c'
    # test_slice[Interval(6, 8)] = 'd'
    # test_slice[Interval(9, 12)] = 'e'
    # for interval in test_slice:
    #     print("Intervals", interval)

    # print(test_slice.keys())
    # SortedIntervalDict({[0, 5]: 'a', [3, 7]: 'b', [6, 10]: 'c', [9, 12]: 'e'})

    # _dict = test_slice[1:6]
    # print(_dict)
    # SortedIntervalDict({[3, 7]: 'b', [6, 10]: 'c', [9, 12]: 'd'})
