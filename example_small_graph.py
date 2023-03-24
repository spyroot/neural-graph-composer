import networkx as nx
import torch
from matplotlib import pyplot as plt

from neural_graph_composer.midi.midi_note import MidiNote
from neural_graph_composer.midi.midi_sequence import MidiNoteSequence
from neural_graph_composer.midi_dataset import MidiDataset
from neural_graph_composer.midi_graph_builder import MidiGraphBuilder
from neural_graph_composer.midi_reader import MidiReader


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
    # transform = T.Compose([
    #     T.NormalizeFeatures(),
    # ])

    ds = MidiDataset(root="./data",
                     per_instrument_graph=False)

    train_mask = ds[0].train_mask
    print("mask", train_mask)
    print(ds[0])
    data = ds[0]
    data.y[train_mask]
    print(data.x[100])
    print(data.y[100])
    print(data.label[100])

    print(list(ds.index_to_hash.values())[0])
    print(list(ds.hash_to_index.values())[0])
    print(list(ds.hash_to_notes.values())[0])
    print(list(ds.notes_to_hash.values())[0])

    # node_hash = ds.index_to_hash[data.y[100].item()]
    # original_index = ds.hash_to_index[node_hash]
    # node_features = data.x[original_index]
    # print(node_features)
    #
    # print(original_index)
    # print(ds.index_to_hash[8])
    # print(ds.hash_to_notes[node_hash])
    #
    # print(data.y[100].item())
    # print(data.label[100].item())
    # note_set_hash = ds.index_to_hash[data.y[100].item()]
    # print(note_set_hash)
    # print(ds.hash_to_notes[note_set_hash])
    #
    n_nodes = data.y.size(0)
    # original_indices = [i for i in range(n_nodes)]
    # node_index = 100
    #
    # # get the original index of the node
    # original_index = original_indices[data.y[node_index].item()]
    # print(original_indices)
    #
    # # # use the original index to get the corresponding node hash value
    # node_hash = ds.index_to_hash[original_index]
    # print(node_hash)
    #
    # # # use the node hash value to get the set of notes it represents
    # notes_set = ds.hash_to_notes[node_hash]
    # print(notes_set)
    #
    # # # use the original index to get the node features
    # node_features = data.x[original_index]
    # # # print the set of notes and the corresponding node features
    # # print("Node hash value:", node_hash)
    # # print("Set of notes:", notes_set)
    # print("Node features:", node_features)
    #
    # 6221080727990553348
    # frozenset({58, 61, 63})
    # Node
    # features: tensor([36., 0., 0., 0., 0.])

    original_indices = [i for i in range(n_nodes)]
    node_index = 100

    # get the original index of the node
    original_index = original_indices[data.y[node_index].item()]
    print(original_indices)

    # use the original index to get the set of notes represented by the node
    notes_set = ds.hash_to_notes[ds.index_to_hash[data.y[node_index].item()]]
    print("note set", notes_set)
    # use the set of notes to look up the corresponding hash value
    node_hash = ds.notes_to_hash[notes_set]
    print(node_hash)

    # use the node hash value to get the node features
    node_features = data.x[original_index]
    print("Node features:", node_features)

    train_ds = MidiDataset(root="./data")
    data_x = train_ds.data.x[train_mask]
    data_y = train_ds.data.y[train_mask]
    data_label = train_ds.data.label[train_mask]

    node_index = 100
    original_index = data_y[node_index].item()

    # get the corresponding node hash value
    node_hash = train_ds.index_to_hash[original_index]

    # get the set of notes it represents
    notes_set = train_ds.hash_to_notes[node_hash]

    # get the node features
    node_features = data_x[node_index]

    print("Original index:", original_index)
    print("Node hash value:", node_hash)
    print("Set of notes:", notes_set)
    print("Node features:", node_features)

    # train_ds.data = train_ds.data[train_ds.data['train_mask']]
    #
    # node_index = 100
    # original_index = train_ds.data.y[node_index].item()
    #
    # # use the original index to get the set of notes represented by the node
    # notes_set = ds.hash_to_notes[ds.index_to_hash[original_index]]
    # print("Note set", {notes_set})
    # node_hash = ds.notes_to_hash[notes_set]
    # print(node_hash)
    #
    # # use the node hash value to get the node features
    # node_features = train_ds.data.x[node_index]
    # print("Node features:", node_features)


if __name__ == '__main__':

    graph_builder = MidiGraphBuilder(
        None, is_instrument_graph=True)

    raw_paths = ['data/raw/a_night_in_tunisia_2_jc.mid',
                 'data/raw/a_night_in_tunisia_2_jc.mid']

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
