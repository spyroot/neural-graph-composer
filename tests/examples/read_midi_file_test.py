from neural_graph_composer.midi_reader import MidiReader

# midi_data = MidiReader.midi_to_tensor("../neural_graph_composer/dataset/quantization/midi_test24_2_instrument.mid")


def read_test():
    midi_data = MidiReader.read("../../neural_graph_composer/dataset/quantization/original.mid")
    expected = MidiReader.read("../../neural_graph_composer/dataset/quantization/32.mid")
    original = midi_data.__midi_seqs[0]
    expected.seq = expected.__midi_seqs[0]
    print("start test")
    for i in range(10):
        n = original.notes[i]
        quantized_note = n.quantize(sps=32)
        print(f"####### {original[i].start_time} {original[i].end_time} "
              f"vs {quantized_note.start_time} {quantized_note.end_time}")


if __name__ == '__main__':
    read_test()
    # midi_data = midi_to_tensor("../neural_graph_composer/dataset/unit_test/test-2-tracks-type-2.mid")
    # print(midi_data)
    # print(midi_data.num_instruments())
