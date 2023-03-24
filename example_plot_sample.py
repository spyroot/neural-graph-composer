import matplotlib.pyplot as plt
import networkx as nx
from neural_graph_composer.midi_graph_builder import MidiGraphBuilder
from neural_graph_composer.midi_reader import MidiReader

pitch_dict = {
        0: 'C',
        1: 'C#',
        2: 'D',
        3: 'D#',
        4: 'E',
        5: 'F',
        6: 'F#',
        7: 'G',
        8: 'G#',
        9: 'A',
        10: 'A#',
        11: 'B'
    }

if __name__ == '__main__':
    """
    """
    midi_seqs = MidiReader.read('data/raw/a_night_in_tunisia_2_jc.mid')
    graph_builder = MidiGraphBuilder()
    graph_builder.build(midi_seqs, is_per_instrument=False)
    g = graph_builder.sub_graphs[0]

    # Remap node labels to pitch names
    note_names = {hash_val: list(note_set)[0] for hash_val, note_set in nx.get_node_attributes(g, 'label').items()}
    remapped_labels = [note_names[hash_val] for hash_val in g.nodes()]

    #
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
    plt.savefig('graph.png', dpi=300)



