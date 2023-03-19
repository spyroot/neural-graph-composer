from torch_geometric.nn import GraphConv
import torch


class ComposerGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ComposerGCN, self).__init__()
        print(f"Creating model input dim {input_dim} hidden_dim {hidden_dim} output_dim {output_dim}")
        self.conv1 = GraphConv(input_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, output_dim)

    def forward(self, data, edge_weight=None):
        """

        :param data:
        :param edge_weight:
        :return:
        """
        print("FOrward")
        x, edge_index = data.x, data.edge_index

        if edge_weight is None:
            x = self.conv1(x, edge_index)
        else:
            x = self.conv1(x, edge_index, edge_weight)
        x = torch.relu(x)

        if edge_weight is None:
            x = self.conv2(x, edge_index)
        else:
            x = self.conv2(x, edge_index, edge_weight)


        return x


