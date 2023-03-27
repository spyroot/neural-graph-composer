import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go

from neural_graph_composer.midi_graph_builder import MidiGraphBuilder
from neural_graph_composer.midi_reader import MidiReader
import pathlib

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
    # file = 'data/raw/a_night_in_tunisia_2_jc.mid'
    #
    file = 'neural_graph_composer/dataset/data/nardis.mid'
    p = pathlib.Path(file)
    print("Plot name ", p.name)

    midi_seqs = MidiReader.read(file)
    graph_builder = MidiGraphBuilder()
    graph_builder.build(midi_seqs, is_per_instrument=False)
    g = graph_builder.sub_graphs[0]

    print(g)

    # remap node labels to pitch names
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

    edge_traces = []
    for edge in g.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]

        weight = g.get_edge_data(edge[0], edge[1])['weight']
        if weight > 1:
            edge_traces.append(go.Scatter(x=[x0, x1, None], y=[y0, y1, None],
                                          line=dict(width=2, color='red'), hoverinfo='none', mode='lines'))
        else:
            edge_traces.append(go.Scatter(x=[x0, x1, None], y=[y0, y1, None],
                                          line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines'))


    #node trace
    node_trace = go.Scatter(x=[], y=[], text=[],
                            mode='markers+text', hoverinfo='text',
                            marker=dict(color='lightblue', size=10))

    for node in g.nodes():
        x, y = pos[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        label = ' '.join(list(g.nodes[node]['label']))
        print(label)
        node_trace['text'] += tuple([label])

    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace],

                    layout=go.Layout(

                        title=p.name,
                        width=800 * 4,
                        height=600 * 4,
                        titlefont=dict(size=16),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(showarrow=False, x=pos[k][0], y=pos[k][1], text=note_names[k])
                                     for k in
                g.nodes()],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    fig.show()
