import argparse
import os.path as osp

import torch
from torch_geometric import loader
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import download_url

from neural_graph_composer.midi_tensor import midi_to_tensor
from midi_graph_builder import construct_graph_from_seq, from_midi_networkx


class MidiDataset(InMemoryDataset):
    """Create dataset from list of MIDI files
    """

    def __init__(self, root, transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 default_node_attr='attr'):
        self.url = "http://localhost:9000"
        self.node_attr_name = default_node_attr
        print("called")
        super().__init__(root, transform, pre_transform, pre_filter)
        self.node_attr_name = default_node_attr

    @property
    def raw_file_names(self):
        return ['midi_test01.mid', 'midi_test02.mid', 'midi_test03.mid']

    @property
    def processed_file_names(self):
        return ['midi_test01.mid.pt', 'midi_test02.mid.pt', 'midi_test03.mid.pt']

    def download(self):
        print(f"raw_file_names {self.raw_file_names}")
        for raw_file in self.raw_file_names:
            print(f"Downloading {raw_file}")
            download_url(f"{self.url}/{raw_file}", self.raw_dir)

    def process(self):
        """
        :return:
        """
        idx = 0
        for raw_path in self.raw_paths:
            print(f"Downloading {raw_path}")
            # Read data from `raw_path`.
            try:
                midi_seq = midi_to_tensor(raw_path, is_debug=False)
                midi_graph = construct_graph_from_seq(midi_seq[0], is_debug=True)
                data = from_midi_networkx(midi_graph, group_node_attrs=[self.node_attr_name])
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
            except KeyError as ker_err:
                print(f"Error in file {raw_path} error: {ker_err}")
        idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()

    dataset = MidiDataset(root="./data")
    print("Dataset size", len(dataset))
    # print("Dataset 01", dataset[1])
    # print("Connected ", dataset[0].edge_index.T)

    # print("Dataset 02", dataset[1])
    # print("Dataset 03", dataset[2])
    # train_dataset = dataset[len(dataset) // 10:]
    # train_loader = loader.DataLoader(train_dataset, args.batch_size, shuffle=True)
    #
    # test_dataset = dataset[:len(dataset) // 10]
    # test_loader = loader.DataLoader(test_dataset, args.batch_size)

    # print(dataset[0])
    # # read midi to internal representation
    midi_seq = midi_to_tensor("dataset/midi_test01.mid")
    midi_graph = construct_graph_from_seq(midi_seq[0])
    pyg_data = from_midi_networkx(midi_graph)

    # # nx.draw(midi_graph, with_labels=True)
    # # plt.show()
    #
    # print(midi_graph.nodes(data=True))
    # print("Pyg data", pyg_data)

    # nx.draw(midi_graph_from_pyg, with_labels=True)
    # plt.show()
    # nx.draw(midi_graph, with_labels=True)
    # plt.show()

# loader = NeighborLoader(
#     data,
#     # Sample 30 neighbors for each node for 2 iterations
#     num_neighbors=[30] * 2,
#     # Use a batch size of 128 for sampling training nodes
#     batch_size=128,
#     input_nodes=data.train_mask,
# )


# for n in midi_graph:
#     print(n.notes)
#
# [ (frozenset({'C♯4', 'G3', 'A♯3', 'D♯4'}),
#    {'attr': {'notes': array([58, 61, 63, 55])},
#     'label': frozenset({'C♯4', 'G3', 'A♯3', 'D♯4'})}),
#   (frozenset({'E4', 'C♯4', 'A3', 'F3'}),
#    {'attr': {'notes': array([64, 57, 61, 53])},
#     'label': frozenset({'E4', 'C♯4', 'A3', 'F3'})}),
#   (frozenset({58, 63, 61, 55}), {}),

#   (frozenset({'G3', 'A♯3', 'D♯4'}),
#    {'attr': {'notes': array([58, 63, 55])}, 'label': frozenset({'G3', 'A♯3', 'D♯4'})}),
#   (frozenset({64, 57, 61, 53}), {}), (frozenset({58, 63, 55}), {}),
#   (frozenset({'A4', 'D4', 'E4', 'A♯3'}),
#    {'attr': {'notes': array([64, 58, 69, 62])},
#     'label': frozenset({'A4', 'D4', 'E4', 'A♯3'})}),
#   (frozenset({64, 58, 69, 62}), {}),
#   (frozenset({'C♯4', 'A3', 'E4'}),
#    {'attr': {'notes': array([64, 57, 61])},
#     'label': frozenset({'C♯4', 'A3', 'E4'})}),
#   (frozenset({'A4', 'E5', 'F4', 'B4'}),
#    {'attr': {'notes': array([65, 76, 69, 71])},
#     'label': frozenset({'A4', 'E5', 'F4', 'B4'})}),
#   (frozenset({64, 57, 61}), {}),
#   (frozenset({'A4', 'F4', 'B4'}),
#    {'attr': {'notes': array([65, 69, 71])},
#     'label': frozenset({'A4', 'F4', 'B4'})}),
#   (frozenset({65, 76, 69, 71}), {}),
#   (frozenset({'G4', 'C♯4', 'A♯3', 'D♯4'}),
#    {'attr': {'notes': array([58, 67, 61, 63])},
#     'label': frozenset({'G4', 'C♯4', 'A♯3', 'D♯4'})}),
#   (frozenset({65, 69, 71}), {})]

# for i, (_, feat_dict) in enumerate(midi_graph.nodes(data=True)):
#     print(f"comparing {set(feat_dict.keys())}")

# data = from_networkx(midi_graph)
# print(data)
#
# node_attrs = list(next(iter(midi_graph.nodes(data=True)))[-1].keys())
# print(node_attrs)
