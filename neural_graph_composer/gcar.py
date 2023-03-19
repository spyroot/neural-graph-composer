import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch.nn import Linear, Sequential, ReLU

class GCAR(MessagePassing):
    def __init__(self, in_channels, out_channels, num_layers, hidden_channels):
        super(GCAR, self).__init__(aggr='add')

        self.num_layers = num_layers

        self.conv_start = Linear(in_channels, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 2):
            self.convs.append(Linear(hidden_channels, hidden_channels))

        self.conv_end = Linear(hidden_channels, out_channels)

        self.activation = ReLU()

        self.reset_parameters()

    def reset_parameters(self):
        self.conv_start.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.conv_end.reset_parameters()

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        x = self.conv_start(x)
        x = self.activation(x)

        h = x
        for i in range(self.num_layers - 2):
            deg = degree(edge_index[0], x.size(0), dtype=x.dtype)
            h = h * deg.view(-1, 1).to(x.dtype)
            h = self.propagate(edge_index, x=h)
            h = self.convs[i](h)
            h = self.activation(h)

        deg = degree(edge_index[0], x.size(0), dtype=x.dtype)
        h = h * deg.view(-1, 1).to(x.dtype)
        h = self.propagate(edge_index, x=h)
        h = self.conv_end(h)

        return h

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return aggr_out