import torch
from torch_geometric.nn import GCNConv


x = torch.cat((torch.randn(size=(500, 1)), torch.randn(size=(500, 1)) * .0001))
A = torch.cat((torch.ones(size=(500, 500)), torch.zeros(size=(500, 500))))
A = torch.cat((A, torch.flip(A, dims=(0, 1))), axis=1).long()

gcn_conv = GCNConv(in_channels=1, out_channels=3)
gcn_conv(x, A)