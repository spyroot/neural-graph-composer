"""
Different check experiments.
"""
import networkx as nx
import torch
from matplotlib import pyplot as plt

from neural_graph_composer.midi.midi_note import MidiNote
from neural_graph_composer.midi.midi_sequence import MidiNoteSequence
from neural_graph_composer.midi_dataset import MidiDataset
from neural_graph_composer.midi_graph_builder import MidiGraphBuilder
from neural_graph_composer.midi_reader import MidiReader
import numpy as np


def example_one():
    """
    """
    midi_seqs = MidiReader.read(
        'neural_graph_composer/dataset/unit_test/test-c-major-scale.mid')
    print(midi_seqs)

    graph_builder = MidiGraphBuilder()
    graph_builder.build(midi_seqs, is_per_instrument=False)
    g = graph_builder.sub_graphs[0]

    for node in g:
        notes = graph_builder.hash_to_notes[node]
        print(f"Hash: {node}, Notes: {notes}")
        for neighbor in g.neighbors(node):
            nei = graph_builder.hash_to_notes[neighbor]
            weight = g.get_edge_data(node, neighbor)['weight']
            print(f"  --> Neighbor: {nei}, Weight: {weight}")

    pos = nx.spring_layout(g, seed=42)

    # Draw nodes and edges
    fig, ax = plt.subplots(figsize=(12, 12))
    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=150, node_color='lightblue')
    nx.draw_networkx_edges(g, pos, ax=ax, edge_color='gray', alpha=0.5)

    # Label nodes
    labels = nx.get_node_attributes(g, 'label')
    label_pos = {k: (v[0], v[1] + 0.05) for k, v in pos.items()}
    nx.draw_networkx_labels(g, label_pos, labels, font_size=8, font_family='sans-serif')

    # Set axis and title
    ax.set_axis_off()
    ax.set_title('MIDI Note Graph', fontweight='bold', fontsize=16)

    # Save figure
    plt.savefig('small_graph.png', dpi=300)
    plt.show()

    seq = MidiNoteSequence(notes=notes)
    builder = MidiGraphBuilder()

    graph = builder.build_sequence(seq)
    for node in graph.nodes:
        notes = builder.hash_to_notes[node]
        print(f"Hash: {node}, Notes: {notes}")
        for neighbor in graph.neighbors(node):
            nei = builder.hash_to_notes[neighbor]
            weight = graph.get_edge_data(node, neighbor)['weight']
            print(f"  --> Neighbor: {nei}, Weight: {weight}")


def example_two():
    """

    :return:
    """
    notes = [
        MidiNote(pitch=60, start_time=0.0, end_time=0.5),
        MidiNote(pitch=61, start_time=1.0, end_time=1.5),
        MidiNote(pitch=62, start_time=2.0, end_time=2.5),
        MidiNote(pitch=62, start_time=3.0, end_time=3.5),
        MidiNote(pitch=60, start_time=4.0, end_time=4.5),
        MidiNote(pitch=60, start_time=6.0, end_time=6.5),
        MidiNote(pitch=60, start_time=7.0, end_time=7.5),
        MidiNote(pitch=61, start_time=8.0, end_time=8.5),
        MidiNote(pitch=62, start_time=9.0, end_time=9.5),
        MidiNote(pitch=62, start_time=10.0, end_time=10.5),
    ]

    seq = MidiNoteSequence(notes=notes)
    builder = MidiGraphBuilder()

    graph = builder.build_sequence(seq)
    for node in graph.nodes:
        notes = builder.hash_to_notes[node]
        print(f"Hash: {node}, Notes: {notes}")
        for neighbor in graph.neighbors(node):
            nei = builder.hash_to_notes[neighbor]
            weight = graph.get_edge_data(node, neighbor)['weight']
            print(f"  --> Neighbor: {nei}, Weight: {weight}")


def example_three():
    """
    :return:
    """
    midi_seqs = MidiReader.read(
        'neural_graph_composer/dataset/unit_test/test-c-major-scale.mid')

    graph_builder = MidiGraphBuilder()
    graph_builder.build(midi_seqs, is_per_instrument=False)
    for pyg_data in graph_builder:
        print(f"data {pyg_data}")
        print(f"x    {pyg_data.x}")
        print(f"y    {pyg_data.y}")
        print(f"label {pyg_data.label}")


def index_to_hash():
    """Iterate and does reverse check, so we can
      get Data.x and Data.y -> hash -> index -> pitch set
    :return:
    """
    midi_dataset = MidiDataset(root="./data",
                               per_instrument_graph=False)

    train_mask = midi_dataset[0].train_mask
    data_x = midi_dataset.data.x[train_mask]
    data_y = midi_dataset.data.y[train_mask]
    data_label = midi_dataset.data.label[train_mask]

    print("Data.y", data_y)
    print("Data.x", data_x)
    print("Data.label", data_label)

    for i in range(data_x.shape[0]):
        node_features = data_x[i]
        original_index = data_y[i].item()
        hash_of_index = midi_dataset.index_to_hash[original_index]
        original_set_of_notes = midi_dataset.hash_to_notes[hash_of_index]
        original_set_tensor = torch.tensor(list(original_set_of_notes))
        original_set_zero = torch.zeros((data_x.shape[0],))
        original_set_tensor = torch.cat((original_set_tensor, original_set_zero), dim=0)[:data_x.shape[1]].unsqueeze(0)
        node_features = node_features.unsqueeze(0)
        sorted_node_features, _ = torch.sort(node_features)
        sorted_original_set_tensor, _ = torch.sort(original_set_tensor)
        if not torch.equal(sorted_node_features, sorted_original_set_tensor):
            print(f"Error for index {i}, hash {hash_of_index}, notes {original_set_of_notes}:")
            print(sorted_node_features, sorted_original_set_tensor)


def index_checker():
    """

    :return:
    """
    # we build
    graph_builder = MidiGraphBuilder(
        None, is_instrument_graph=True)

    # we use to same files
    raw_paths = ['data/raw/a_night_in_tunisia_2_jc.mid',
                 'data/raw/a_night_in_tunisia_2_jc.mid']

    #
    pyg_data_list = []
    for raw_path in raw_paths:
        midi_seqs = MidiReader.read(raw_path)
        graph_builder.build(midi_seqs)
        graphs = graph_builder.graphs()
        if not graphs:
            raise ValueError("No sub graphs found in graph builder.")
        for g in graphs:
            pyg_data_list.append(g)

    index_to_hash = graph_builder.index_to_hash
    hash_to_index = graph_builder.hash_to_index
    hash_to_notes = graph_builder.hash_to_notes

    node_index = 55
    original_index_graph_0 = pyg_data_list[0].y[node_index].item()
    original_index_graph_1 = pyg_data_list[1].y[node_index].item()
    print(f"Original index: {original_index_graph_0}")
    print(f"Original index: {original_index_graph_1}")

    #
    node_hash_graph_0 = index_to_hash[original_index_graph_0]
    node_hash_graph_1 = index_to_hash[original_index_graph_1]
    print(f"Node hash value: {node_hash_graph_0}")
    print(f"Node hash value: {node_hash_graph_1}")
    #
    notes_set0 = hash_to_notes[node_hash_graph_0]
    notes_set1 = hash_to_notes[node_hash_graph_1]
    print(f"Set of notes: {notes_set0}")
    print(f"Set of notes: {notes_set1}")

    #
    node_features0 = pyg_data_list[0].x[original_index_graph_0]
    node_features1 = pyg_data_list[1].x[original_index_graph_1]
    print(f"Node features: {node_features0}")
    print(f"Node features: {node_features1}")

    midi_dataset = MidiDataset(root="./data_test",
                               midi_files=raw_paths,
                               per_instrument_graph=False)

    train_mask = midi_dataset[0].train_mask
    data_x = midi_dataset.data.x[train_mask]
    data_y = midi_dataset.data.y[train_mask]
    data_label = midi_dataset.data.label[train_mask]

    print("Data.y", data_y)
    print("Data.x", data_x)
    print("Data.label", data_label)

    node_index = 55
    original_index = data_y[node_index].item()
    print("original index", original_index)
    # original index 21

    for key, value in graph_builder.index_to_hash.items():
        assert key in midi_dataset.index_to_hash
        assert value == midi_dataset.index_to_hash[key]

    for key, value in graph_builder.hash_to_notes.items():
        assert key in midi_dataset.hash_to_notes
        assert value == midi_dataset.hash_to_notes[key]

    for key, value in graph_builder.hash_to_index.items():
        assert key in midi_dataset.hash_to_index
        assert value == midi_dataset.hash_to_index[key]

    for key, value in graph_builder.notes_to_hash.items():
        assert key in midi_dataset.notes_to_hash
        assert value == midi_dataset.notes_to_hash[key]

    print("Feature vector", data_x[33])
    print("label ", data_y[33])
    hash_of_33 = midi_dataset.index_to_hash[
        data_y[33].item()
    ]
    print("Hash of index 33", midi_dataset.hash_to_notes[hash_of_33])

    for i in range(data_x.shape[0]):
        node_features = data_x[i]
        original_index = data_y[i].item()
        hash_of_index = midi_dataset.index_to_hash[original_index]
        original_set_of_notes = midi_dataset.hash_to_notes[hash_of_index]
        original_set_tensor = torch.tensor(list(original_set_of_notes))
        original_set_zero = torch.zeros((5,))
        original_set_tensor = torch.cat((original_set_tensor, original_set_zero), dim=0)[:5].unsqueeze(0)
        node_features = node_features.unsqueeze(0)

        sorted_node_features, _ = torch.sort(node_features)
        sorted_original_set_tensor, _ = torch.sort(original_set_tensor)
        if not torch.equal(sorted_node_features, sorted_original_set_tensor):
            print(f"Error for index {i}, hash {hash_of_index}, notes {original_set_of_notes}:")
            print(node_features, original_set_tensor)


def tolerance_checker():
    """This test generally better perform
    with understanding drift between a notes and then check
    note groups or not.
    :return:
    """
    # we use to same files
    raw_paths = ['data/raw/a_night_in_tunisia_2_jc.mid']

    midi_dataset = MidiDataset(root="./data_test",
                               midi_files=raw_paths,
                               per_instrument_graph=False,
                               tolerance=0.5)

    train_mask = midi_dataset[0].train_mask
    data_x = midi_dataset.data.x[train_mask]
    data_y = midi_dataset.data.y[train_mask]
    data_label = midi_dataset.data.label[train_mask]

    print("Data.y", data_y)
    print("Data.x", data_x)
    print("Data.label", data_label)


def different_datasets():
    """Create different type dataset.
       Include velocity will add velocity vector.
    :return:
    """
    # include velocity
    # midi_dataset = MidiDataset(root="./data",
    #                            per_instrument_graph=False,
    #                            include_velocity=True)
    # print(midi_dataset[0])
    #
    # # no velocity
    # midi_dataset = MidiDataset(root="./data",
    #                            per_instrument_graph=False,
    #                            include_velocity=False)
    # print(midi_dataset[0])

    # adjust tolerance
    midi_dataset = MidiDataset(root="./data",
                               per_instrument_graph=False,
                               include_velocity=False,
                               tolerance=0.5)
    print(midi_dataset[0])
    print(midi_dataset[8].x)

    midi_dataset = MidiDataset(root="./data",
                               per_instrument_graph=False,
                               include_velocity=False,
                               tolerance=0.5)
    print(midi_dataset[0])
    print(midi_dataset[8].x)

    # per instrument graph
    midi_dataset = MidiDataset(root="./data",
                               per_instrument_graph=False,
                               include_velocity=False,
                               tolerance=0.5)
    print(midi_dataset[0])
    print(midi_dataset[8].x)


def read_empty_files():
    """
    :return:
    """
    midi_seqs = MidiReader.read(
        '/Users/spyroot/dev/neural-graph-composer/data/raw/58486a_nocturne_op_55_no_1_(nc)smythe.mid')


# def time_difference_distribution(midi_note_sequences):
#     """
#
#     :param midi_note_sequences:
#     :return:
#     """
#     time_differences = []
#
#     for seq in midi_note_sequences:
#         start_times = sorted([n.start_time for n in seq.notes if not n.is_drum and n.velocity != 0])
#         differences = np.diff(start_times)
#         time_differences.extend(differences)
#
#     return np.array(time_differences)

# Example usage:
# midi_note_sequences = [...]  # List of MidiNoteSequence
# time_differences = time_difference_distribution(midi_note_sequences)
#
# # Calculate percentiles to choose a suitable tolerance value
# tolerance_90th_percentile = np.percentile(time_differences, 90)
# tolerance_95th_percentile = np.percentile(time_differences, 95)
# tolerance_99th_percentile = np.percentile(time_differences, 99)
#
# print("90th percentile:", tolerance_90th_percentile)
# print("95th percentile:", tolerance_95th_percentile)
# print("99th percentile:", tolerance_99th_percentile)


import numpy as np


def rescale_velocities(velocities, target_min_velocity=64, target_max_velocity=127):
    velocities = np.array(velocities)

    # Calculate the current min and max velocities
    current_min_velocity = np.min(velocities)
    current_max_velocity = np.max(velocities)

    # Rescale the velocities to the target range
    rescaled_velocities = ((velocities - current_min_velocity) * (target_max_velocity - target_min_velocity) / (
            current_max_velocity - current_min_velocity)) + target_min_velocity

    # Clip the rescaled velocities to the valid MIDI range (


def rescale_velocities(velocities, target_min_velocity=64, target_max_velocity=127):
    velocities = np.array(velocities)

    # Calculate the current min and max velocities
    current_min_velocity = np.min(velocities)
    current_max_velocity = np.max(velocities)

    # Rescale the velocities to the target range
    rescaled_velocities = ((velocities - current_min_velocity) * (target_max_velocity - target_min_velocity) / (
            current_max_velocity - current_min_velocity)) + target_min_velocity

    # Clip the rescaled velocities to the valid MIDI range (0 to 127)
    clipped_velocities = np.clip(rescaled_velocities, 0, 127).astype(int)

    return clipped_velocities.tolist()


# Example usage:
velocities = [20, 26, 23, 20, 40, 20, 60, 20, 30, 20, 20, 30]
rescaled_velocities = rescale_velocities(velocities)
print(rescaled_velocities)


def scale_relative_velocities(velocities, scaling_factor=1.0):
    """

    :param velocities:
    :param scaling_factor:
    :return:
    """
    velocities = np.array(velocities)

    # Calculate the average velocity
    avg_velocity = np.mean(velocities)
    print(avg_velocity)

    # Calculate the ratio between each note's velocity and the average velocity
    velocity_ratios = velocities / avg_velocity
    print(velocity_ratios)
    # Scale the velocities using the scaling factor
    scaled_velocities = velocity_ratios * scaling_factor * avg_velocity
    print(scaled_velocities)
    # Clip the scaled velocities to the valid MIDI range (0 to 127)
    clipped_velocities = np.clip(scaled_velocities, 0, 127).astype(int)

    return clipped_velocities.tolist()


def instrument_time_differences(midi_note_sequence):
    start_times = sorted([n.start_time for n in midi_note_sequence.notes
                          if not midi_note_sequence.instrument.is_drum])
    differences = np.diff(start_times)
    return differences


def tolerance_test():
    """Test for different method ,
     hungarian_rhapsody has crazy timing so nice to explore on this piece.
    """
    midi_seqs = MidiReader.read(
        '/data/raw/1033w_hungarian_rhapsody_12_(nc)smythe.mid')

    time_differences = []

    for seq in midi_seqs:
        print(midi_seqs[seq].notes)
        start_times = sorted([n.start_time for n in midi_seqs[seq].notes
                              if not midi_seqs[seq].instrument.is_drum])
        differences = np.diff(start_times)
        time_differences.extend(differences)

    time_differences = np.array(time_differences)
    tolerance_95th_percentile = np.percentile(time_differences, 95)
    print(tolerance_95th_percentile)

    tolerance_values = []

    for seq in midi_seqs:
        time_differences = instrument_time_differences(midi_seqs[seq])
        if len(time_differences) > 0:
            tolerance_95th_percentile = np.percentile(time_differences, 95)
            tolerance_values.append(tolerance_95th_percentile)
            print(f"Instrument {seq}: 95th percentile tolerance = {tolerance_95th_percentile}")

    # You can also calculate the average tolerance across all instruments
    average_tolerance = np.mean(tolerance_values)
    print(f"Average tolerance across all instruments: {average_tolerance}")

    for seq in midi_seqs:
        if midi_seqs[seq].instrument.is_drum:
            continue
        s = midi_seqs[seq]
        for n in s.notes:
            # print(f"Using tollerance {tolerance_values[seq]}")
            # t = round(round(n.start_time / float(tolerance_values[seq])) * float(tolerance_values[seq]), 3)
            method_one = round(round(n.start_time / float(tolerance_values[seq])) * float(tolerance_values[seq]), 3)
            method_two = round(n.start_time / float(tolerance_values[seq])) * float(tolerance_values[seq])
            method_three = round(n.start_time / 0.5) * 0.5
            method_four = round(n.start_time / 0.2) * 0.2
            method_five = round(n.start_time, 3)

            print(f"{s.instrument.name} start {n.pitch_name} time {n.start_time} "
                  f"m1 {method_one} "
                  f"m2:{method_two} "
                  f"m3:{method_two} "
                  f"m4: {method_four} "
                  f"m5: {method_five} ")

    scaled_velocities = scale_relative_velocities(velocities, scaling_factor)
    print(scaled_velocities)


def data_type():
    raw_paths = ['data/raw/a_night_in_tunisia_2_jc.mid']
    midi_dataset = MidiDataset(root="./data_test",
                               midi_files=raw_paths,
                               per_instrument_graph=False,
                               tolerance=0.5,
                               include_velocity=True)
    print(midi_dataset.x[0])

    ds = MidiDataset(root="./data",
                     per_instrument_graph=False,
                     tolerance=0.5,
                     include_velocity=True)
    print(ds[0].x[0])


if __name__ == '__main__':
    """
    """
    # read_empty_files()
    # print("Tolerance checker:")
    # tolerance_checker()
    # print("Dataset creation checker:")
    # different_datasets()
    data_type()
