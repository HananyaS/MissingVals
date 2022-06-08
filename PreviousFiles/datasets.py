import numpy as np
import torch
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, df, labels):
        self.df = torch.from_numpy(df.values)
        self.labels = torch.from_numpy(labels.values).long()
        # self.labels = torch.from_numpy(np.array([[1, 0] if x == 0 else [0, 1] for x in labels])).long()

    def __getitem__(self, idx):
        # return torch.Tensor(self.df.iloc[idx].values)
        return [self.df[idx], self.labels[idx]]

    def __len__(self):
        return self.df.shape[0]


class GraphDataset(Dataset):
    def __init__(self, feats, adj_mats, labels=None):
        assert feats.shape[0] == adj_mats.shape[0]
        assert feats.shape[0] == labels.shape[0] if labels is not None else True

        self.feats = torch.from_numpy(feats)
        self.adj_mats = torch.from_numpy(adj_mats)
        self.labels = torch.from_numpy(labels.values).long()

    def __getitem__(self, idx):
        if self.labels[idx] is None:
            return [self.feats[idx], self.adj_mats[idx]]

        return [self.feats[idx], self.adj_mats[idx], self.labels[idx]]

    def __len__(self):
        return self.feats.shape[0]

