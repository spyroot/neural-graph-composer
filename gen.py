import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data


class GraphRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphRNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.rnn = nn.GRU(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)
        out, h = self.rnn(x, h)
        out = self.fc(out)
        out = out.view(-1, self.output_dim)
        return out


if __name__ == '__main__':
    # Generate toy dataset
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7, 0]])
    x = torch.randn(8, 5)
    y = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])

    # Mask some nodes for training
    mask = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.bool)
    mask_idx = torch.arange(8)[mask]
    mask_edge_index = edge_index[:,
                      ((edge_index[0] != mask_idx[:, None]) & (edge_index[1] != mask_idx[:, None])).all(0)]
    mask_x = x[mask]
    mask_y = y[mask]

    # Train model to predict missing nodes
    model = GraphRNN(x.size(-1), 64, y.max().item() + 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    epochs = 100
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model.forward(mask_x)
        loss = F.cross_entropy(out, mask_y)
        loss.backward()
        optimizer.step()

    # Generate missing nodes
    missing_mask = ~mask
    missing_x = x[missing_mask]
    missing_idx = torch.arange(8)[missing_mask]
    with torch.no_grad():
        out = model.forward(x)
        missing_y = out.argmax(dim=-1)[missing_idx]
    predicted_y = torch.cat((mask_y, missing_y))

    # Construct final graph
    final_edge_index = edge_index.clone()
    final_x = x.clone()
    final_y = predicted_y.clone()
    final_edge_index = torch.cat((final_edge_index, mask_edge_index), dim=1)
    final_x = torch.cat((final_x, missing_x), dim=0)
    final_y = torch.cat((final_y, torch.zeros(len(missing_idx), dtype=torch.long)), dim=0)
    final_graph = Data(edge_index=final_edge_index, x=final_x, y=final_y)

    print(final_graph)