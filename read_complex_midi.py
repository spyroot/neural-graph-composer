"""
This data set checker.

Check all edge , mask , hash
to index , index to hash etc.

Author Mus spyroot@gmail.com
"""
import glob
import torch
from torch_geometric.loader import DataLoader

from neural_graph_composer.midi_dataset import MidiDataset
from neural_graph_composer.midi_reader import MidiReader
from neural_graph_composer.midi_graph_builder import MidiGraphBuilder


def dataset_checker(graph_builder, dataset):
    """
    :return:
    """
    print(len(graph_builder.hash_to_notes))
    print(len(graph_builder.notes_to_hash))
    print(len(graph_builder.index_to_hash))
    print(len(graph_builder.hash_to_index))

    print("Dataset size", len(dataset))
    print("Labels shape", dataset[0].y)
    # print(dataset[0].y)
    print("Number of classes", dataset.num_classes)
    print(len(dataset.hash_to_notes))
    print(len(dataset.notes_to_hash))
    print(len(dataset.index_to_hash))
    print(len(dataset.hash_to_index))
    print(dataset[0].x)


def read_all_midi():
    """This a main loop to read all midi good as checker.
    :return:
    """
    # dataset = MidiDataset(root="./data")
    graph_builder = None
    midi_dir = "neural_graph_composer/dataset/"
    midi_files = glob.glob(midi_dir + "*.mid")
    for raw_path in midi_files:
        print(f"Reading {raw_path}")
        # read file and construct graph
        midi_seqs = MidiReader.read(raw_path)
        print(f"midi seq number of seq {midi_seqs.num_instruments()}")
        # we build per instrument
        if graph_builder is None:
            graph_builder = MidiGraphBuilder(
                None, is_instrument_graph=True)

        graph_builder.build(midi_seqs)

        # graph_builder output iterator
        for midi_data in graph_builder.graphs():
            print(f"midi_data {midi_data}")


def check_masking(batch):
    """
    Checks that the masking of the batch is correct.
    """
    # train_mask = batch.train_mask
    # val_mask = batch.val_mask
    # test_mask = batch.test_mask

    train_labels = batch.y[batch.train_mask]
    val_labels = batch.y[batch.val_mask]
    test_labels = batch.y[batch.test_mask]

    assert torch.all(train_labels == batch.label[batch.train_mask]), "Error in masking: Train labels don't match."
    assert torch.all(val_labels == batch.label[batch.val_mask]), "Error in masking: Val labels don't match."
    assert torch.all(test_labels == batch.label[batch.test_mask]), "Error in masking: Test labels don't match."

    # # Extract the masked nodes
    # train_idx = torch.where(train_mask)[0]
    # val_idx = torch.where(val_mask)[0]
    # test_idx = torch.where(test_mask)[0]
    #
    # train_nodes = batch.index_select(0, train_idx)
    # val_nodes = batch.index_select(0, val_idx)
    # test_nodes = batch.index_select(0, test_idx)
    #
    #
    # # Check that there's no overlap between the masks
    # overlap = torch.logical_and(train_mask, torch.logical_or(val_mask, test_mask)).sum()
    # assert overlap == 0, "Error in masking: Train and Val/Test masks overlap."
    #
    # # Check that the masked nodes have the expected labels
    # expected_labels = torch.cat([batch.y[train_mask], batch.y[val_mask], batch.y[test_mask]])
    # actual_labels = torch.cat([train_nodes.y, val_nodes.y, test_nodes.y])
    # assert torch.equal(actual_labels, expected_labels), "Error in masking: Labels don't match."

    print("Masking check passed.")


def check_edge_index(batch):
    """

    :param batch:
    :return:
    """
    train_mask = batch.train_mask
    val_mask = batch.val_mask
    test_mask = batch.test_mask

    train_edges = batch.edge_index[:, (train_mask * train_mask).nonzero(as_tuple=True)[0]]
    val_edges = batch.edge_index[:, (val_mask * val_mask).nonzero(as_tuple=True)[0]]
    test_edges = batch.edge_index[:, (test_mask * test_mask).nonzero(as_tuple=True)[0]]

    print(f"batch train_edges from mask  {train_edges.shape}")
    print(f"batch val_edges from mask    {val_edges.shape}")
    print(f"batch test_edges from mask   {test_edges.shape}")

    # Check that there's no overlap between the masks
    train_val_overlap = torch.logical_and(train_mask, val_mask).sum().item()
    train_test_overlap = torch.logical_and(train_mask, test_mask).sum().item()
    val_test_overlap = torch.logical_and(val_mask, test_mask).sum().item()

    assert train_val_overlap == 0, "Error in edge indexing: Train and Val masks overlap."
    assert train_test_overlap == 0, "Error in edge indexing: Train and Test masks overlap."
    assert val_test_overlap == 0, "Error in edge indexing: Val and Test masks overlap."

    # Check that the edge indices are within the expected range
    num_nodes = batch.num_nodes
    num_edges = batch.edge_index.shape[1]

    print(f"batch {batch}")

    max_src_node_idx = num_nodes - 1
    max_dst_node_idx = num_nodes - 1

    assert (train_edges[0] <= max_src_node_idx).all(), f"Error in edge indexing: Train edge src indices out of range."
    assert (train_edges[1] <= max_dst_node_idx).all(), f"Error in edge indexing: Train edge dst indices out of range."
    assert (val_edges[0] <= max_src_node_idx).all(), f"Error in edge indexing: Val edge src indices out of range."
    assert (val_edges[1] <= max_dst_node_idx).all(), f"Error in edge indexing: Val edge dst indices out of range."
    assert (test_edges[0] <= max_src_node_idx).all(), f"Error in edge indexing: Test edge src indices out of range."
    assert (test_edges[1] <= max_dst_node_idx).all(), f"Error in edge indexing: Test edge dst indices out of range."

    # Check that the edge indices are not negative
    assert (train_edges >= 0).all(), f"Error in edge indexing: Train edge indices are negative."
    assert (val_edges >= 0).all(), f"Error in edge indexing: Val edge indices are negative."
    assert (test_edges >= 0).all(), f"Error in edge indexing: Test edge indices are negative."

    # check that the total number of edges is correct
    expected_num_edges = (train_mask.sum() + val_mask.sum() + test_mask.sum())

    expected_num_edges2 = train_edges.shape[1] + val_edges.shape[1] + test_edges.shape[1]

    print(f"Number of train edges: {train_edges.shape[1]}")
    print(f"Number of val edges: {val_edges.shape[1]}")
    print(f"Number of test edges: {test_edges.shape[1]}")
    print(f"Expected number of edges: {expected_num_edges}")
    print(f"Expected number of edges: {expected_num_edges2}")

    assert num_edges == expected_num_edges, f"Error in edge indexing:" \
                                            f" Incorrect number of edges. Got {num_edges}, " \
                                            f"expected {expected_num_edges}."

    print("Edge indexing check passed.")


if __name__ == '__main__':
    read_all_midi()

    ds = MidiDataset(root="./data", per_instrument_graph=False)

    loader = DataLoader(ds, batch_size=1, shuffle=False)
    for i, data in enumerate(loader):
        print(data)

    for batch_size in [1, 4, 8]:
        loader = DataLoader(ds, batch_size=1, shuffle=False)

        for i, data in enumerate(loader):
            train_mask = data.train_mask
            val_mask = data.val_mask
            test_mask = data.test_mask
            total_nodes = data.num_nodes
            mask_sum = train_mask.sum() + val_mask.sum() + test_mask.sum()
            assert total_nodes == mask_sum, f"Error in masking for graph {i}. " \
                                            f"Sum of all masks is {mask_sum} " \
                                            f"but should be {total_nodes}."

            # Verify that there's no overlap between the masks
            mask_sum = (train_mask + val_mask + test_mask).sum()
            assert mask_sum == total_nodes, f"Error in masking for graph {i}. Masks overlap."

            check_masking(data)
            check_edge_index(data)

        for data in ds:
            for i, (y, label) in enumerate(zip(data.y, data.label)):
                original_hash_from_y = ds.index_to_hash[y.item()]
                original_hash_from_label = ds.index_to_hash[label.item()]
                assert original_hash_from_y == original_hash_from_label, \
                    f"Error in hashing for graph {i}. y and label hashes don't match."
