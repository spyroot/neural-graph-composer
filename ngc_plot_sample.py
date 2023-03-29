"""
Take midi file and construct graph and plot
in first version it uses matplotlib and second plot plotly.

Author Mus spyroot@gmail.com
           mbayramo@stanford.edu
"""
import argparse
from typing import Optional

import matplotlib.pyplot as plt

try:
    import networkx as nx
    from networkx import Graph
except ImportError:
    raise ImportError("networkx not installed. "
                      "Please run `pip install networkx` to install networkx.")

try:
    import plotly.graph_objects as go
except ImportError:
    raise ImportError("Plotly not installed. "
                      "Please run `pip install plotly` to install Plotly.")

from neural_graph_composer.midi_graph_builder import MidiGraphBuilder
import pathlib


def pitch_mapping():
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
    return pitch_dict


def create_graph_from_file(file_path) -> nx.Graph:
    """Constructs a networkx.Graph object from a MIDI file
    :return: A NetworkX graph representing the MIDI data in the specified file.
    """
    p = pathlib.Path(file_path)
    if not p.is_file():
        raise ValueError(f"Invalid file path: {file_path}")

    try:
        graph_builder = MidiGraphBuilder.from_file(
            file_path=file_path,
            per_instrument=False)
        graph_builder.build(is_per_instrument=False)
        g = graph_builder.sub_graphs[0]
        return g
    except Exception as e:
        raise ValueError(f"Failed to create graph from file {file_path}: {str(e)}")


def draw_midi_note_graph_matplotlib(
        file: Optional[str] = 'neural_graph_composer/dataset/data/nardis.mid',
        save_path: Optional[str] = 'graph.png') -> tuple[Graph, dict]:
    """
    :param save_path:
    :param file:
    :return:
    """
    g = create_graph_from_file(file)

    # remap node labels to pitch names
    note_names = {hash_val: list(note_set)[0] for
                  hash_val, note_set in nx.get_node_attributes(g, 'label').items()}
    remapped_labels = [note_names[hash_val] for hash_val in g.nodes()]
    #
    pos = nx.spring_layout(g, seed=42)

    # edges
    fig, ax = plt.subplots(figsize=(12, 12))
    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=150, node_color='lightblue')
    nx.draw_networkx_edges(g, pos, ax=ax, edge_color='gray', alpha=0.5)
    # labels nodes
    labels = nx.get_node_attributes(g, 'label')
    label_pos = {k: (v[0], v[1] + 0.05) for k, v in pos.items()}
    nx.draw_networkx_labels(g, label_pos, labels, font_size=8, font_family='sans-serif')
    # axis and title
    ax.set_axis_off()
    ax.set_title('MIDI Note Graph', fontweight='bold', fontsize=16)

    # Save figure
    plt.savefig(save_path, dpi=300)

    return g, pos


def draw_midi_note_graph_plotly(
        file: Optional[str] = 'neural_graph_composer/dataset/data/nardis.mid',
        save_path: Optional[str] = 'graph.png'):
    """Generate plot for MIDI graph and save and show final graph
    :param file:
    :param save_path:
    :return:
    """
    file_path = pathlib.Path(file)

    g = create_graph_from_file(file)
    note_names = {
        hash_val: list(note_set)[0] for
        hash_val, note_set in
        nx.get_node_attributes(g, 'label').items()
    }

    pos = nx.spring_layout(g, seed=42)

    edge_traces = []
    for edge in g.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]

        weight = g.get_edge_data(edge[0], edge[1])['weight']
        if weight > 1:
            edge_traces.append(
                go.Scatter(
                    x=[x0, x1, None], y=[y0, y1, None],
                    line=dict(width=2, color='red'),
                    hoverinfo='none', mode='lines')
            )
        else:
            edge_traces.append(
                go.Scatter(
                    x=[x0, x1, None], y=[y0, y1, None],
                    line=dict(width=0.5, color='#888'),
                    hoverinfo='none', mode='lines')
            )
    # node trace
    node_trace = go.Scatter(
        x=[], y=[], text=[],
        mode='markers+text', hoverinfo='text',
        marker=dict(color='lightblue', size=10)
    )

    for node in g.nodes():
        x, y = pos[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        label = ' '.join(list(g.nodes[node]['label']))
        print(label)
        node_trace['text'] += tuple([label])

    # Create figure
    fig = go.Figure(
        data=edge_traces + [node_trace],
        layout=go.Layout(
            title=file_path.name,
            width=800 * 4,
            height=600 * 4,
            titlefont=dict(size=16),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(showarrow=False,
                     x=pos[k][0],
                     y=pos[k][1],
                     text=note_names[k]
                     )
                for k in g.nodes()],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
    )
    fig.show()
    fig.write_image(save_path, width=800, height=600, scale=2)


def main():
    """
    :return:
    """
    parser = argparse.ArgumentParser("Plot MIDI Graph")
    parser.add_argument('file_name', type=str, help='MIDI file to process')
    parser.add_argument('--plot_file_name',
                        type=str, help='Name of the file to save the plot', default='graph.png')
    args = parser.parse_args()

    file_name = args.file_name
    file_path = pathlib.Path(file_name)

    if file_path.suffix not in ['.mid', '.midi']:
        raise ValueError(f"File {file_path} is not a MIDI file.")

    if not file_path.exists():
        raise ValueError(f"File {file_path} does not exist.")

    draw_midi_note_graph_matplotlib(file_name, save_path=args.plot_file_name)
    draw_midi_note_graph_plotly(file_name, save_path=args.plot_file_name)


if __name__ == '__main__':
    """
    """
    main()
