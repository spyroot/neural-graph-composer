from neural_graph_composer.midi_reader import MidiReader

if __name__ == '__main__':
    midi_data = MidiReader.midi_to_tensor("../neural_graph_composer/dataset/unit_test/midi_test24_2_instrument.mid")
    print(midi_data)
    print(midi_data.num_instruments())

    # midi_data = midi_to_tensor("../neural_graph_composer/dataset/unit_test/test-2-tracks-type-2.mid")
    # print(midi_data)
    # print(midi_data.num_instruments())
