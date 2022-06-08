# from torch_geometric.data import Data
from torch.utils.data import Dataset
import torch


class GraphDS(Dataset):
    def __init__(self, nodes_num, device=None, test=False):
        self.device = device
        self.nodes_num = nodes_num
        self.labels = torch.Tensor().to(device)
        self.x = torch.Tensor().to(device)
        self.adj_mats = torch.Tensor().to(device)
        self.test = test

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        if self.test:
            return self.x[idx], self.adj_mats[idx]
        else:
            return self.x[idx], self.adj_mats[idx], self.labels[idx]

    def add_graph(self, x, adj_mat, label=None):
        self.x = torch.cat((self.x, x.to(self.device).view(1, -1)), dim=0)
        self.adj_mats = torch.cat((self.adj_mats, adj_mat.to(self.device).view(1, *adj_mat.shape)), dim=0)
        if not self.test:
            if not isinstance(label, torch.Tensor):
                label = torch.tensor(label).to(self.device)
            self.labels = torch.cat((self.labels, label.view(-1)))

"""
class GraphsDataset(Dataset):
    def __init__(self, test=False, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        else:
            self.device = device

        # self.transform = transforms.Compose([transforms.ToTensor()])

        self.x = torch.Tensor().long().to(self.device)
        self.edge_index = torch.Tensor().long().to(self.device)
        self.node_attr = torch.Tensor().to(self.device)
        self.edge_attr = torch.Tensor().to(self.device)
        self.node_slice = torch.Tensor([0]).long().to(self.device)
        self.edge_slice = torch.Tensor([0]).long().to(self.device)

        self.num_nodes = torch.Tensor().long().to(self.device)
        self.num_edges = torch.Tensor().long().to(self.device)

        self.last_node_idx = 0
        self.last_edge_idx = 0
        self.num_graphs = 0

        self.test = test

        if not test:
            self.labels = torch.Tensor().long().to(self.device)

    def __len__(self):
        return self.num_graphs

    def __getitem__(self, idx):
        assert idx < self.num_graphs, 'Index out of range'
        x = self.x[self.node_slice[idx]:self.node_slice[idx + 1]]
        edge_index = self.edge_index[:, self.edge_slice[idx]:self.edge_slice[idx + 1]]
        edge_features = self.edge_attr[self.edge_slice[idx]:self.edge_slice[idx + 1]]
        node_features = self.node_attr[self.node_slice[idx]:self.node_slice[idx + 1]]

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_features, node_attr=node_features)

        if self.test:
            return data

        return [data, self.labels[idx].item()]
        # return [self.transform(data), self.transform(self.labels[idx].item())]

    def add_graph(self, kwargs, label=None):
        if self.test:
            assert label is not None, "Label can't be provided for test set"

        x = kwargs['x'].long()
        edge_index = kwargs['edge_index'].long()
        edge_attr = kwargs['edge_attr']
        node_attr = kwargs['node_attr']

        num_nodes = int(x.shape[0])
        num_edges = int(edge_index.shape[1])

        self.x = torch.cat((self.x, x), dim=0)
        self.edge_index = torch.cat((self.edge_index, edge_index), dim=1)
        self.edge_attr = torch.cat((self.edge_attr, edge_attr), dim=0)
        self.node_attr = torch.cat((self.node_attr, node_attr), dim=0)

        self.node_slice = torch.cat(
            (self.node_slice, torch.Tensor([self.last_node_idx + num_nodes]).long().to(self.device)), dim=0)
        self.edge_slice = torch.cat(
            (self.edge_slice, torch.Tensor([self.last_edge_idx + num_edges]).long().to(self.device)), dim=0)

        self.num_nodes = torch.cat((self.num_nodes, torch.Tensor([num_nodes]).long().to(self.device)), dim=0)
        self.num_edges = torch.cat((self.num_edges, torch.Tensor([num_edges]).long().to(self.device)), dim=0)

        self.last_node_idx += num_nodes
        self.last_edge_idx += num_edges

        self.num_graphs += 1

        if not self.test:
            self.labels = torch.cat((self.labels, torch.Tensor([label]).long().to(self.device)), dim=0)

    def get_num_nodes(self, idx):
        return self.num_nodes[idx].item()

    def get_num_edges(self, idx):
        return self.num_edges[idx].item()

    def get_num_graphs(self):
        return self.num_graphs

    def get_node_attr(self, idx):
        return self.node_attr[self.node_slice[idx]:self.node_slice[idx + 1]]

    def get_edge_attr(self, idx):
        return self.edge_attr[self.edge_slice[idx]:self.edge_slice[idx + 1]]

    def get_edge_index(self, idx):
        return self.edge_index[self.edge_slice[idx]:self.edge_slice[idx + 1]]

    def get_node_slice(self, idx):
        return self.node_slice[idx], self.node_slice[idx + 1]

    def get_edge_slice(self, idx):
        return self.edge_slice[idx], self.edge_slice[idx + 1]

    def get_nun_node_features(self):
        return self.node_attr.shape[1]

    def get_num_edge_features(self):
        return self.edge_attr.shape[1]

    def get_num_classes(self):
        return self.labels.max().item() + 1
"""