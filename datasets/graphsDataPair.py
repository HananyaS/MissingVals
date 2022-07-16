import torch
from datasets.graphDataPair import GraphDataPair

import numpy as np
from typing import List, Union, Tuple

from tools.data.tab2graph import tab2graphs
from torch.utils.data import Dataset, DataLoader


class GraphsDataPair:
    _input_types = Union[torch.Tensor, np.ndarray]

    def __init__(
        self,
        X_list: List[_input_types],
        edges_list: List[_input_types],
        Y_list: List[_input_types] = None,
        given_as_adj: bool = False,
        store_as_adj: bool = False,
        include_edge_weights: bool = False,
        name: str = "",
        normalize: bool = False,
        normalization_params: Tuple[List, List] = None,
        shuffle: bool = False,
        add_existence_cols: bool = False,
    ):
        self.name = name
        self.normalized = False
        self.norm_params = None

        self.add_existence_cols = add_existence_cols

        self._create_graph_list(
            X_attr_list=X_list,
            edges_list=edges_list,
            Y_list=Y_list,
            given_as_adj=given_as_adj,
            store_as_adj=store_as_adj,
            add_existence_cols=add_existence_cols,
            include_edge_weights=include_edge_weights,
        )

        if normalize:
            self.zscore(normalization_params=normalization_params, inplace=True)
            self.normalized = True

        if shuffle:
            self._shuffle()

    @classmethod
    def from_tab(cls, **kwargs):
        graph_kwargs, normalized, norm_params = tab2graphs(**kwargs)
        gd = cls(**graph_kwargs)
        gd.normalized = normalized
        gd.norm_params = norm_params

        return gd

    def _create_graph_list(
        self,
        X_attr_list: List[_input_types],
        edges_list: List[_input_types],
        Y_list: List[_input_types] = None,
        **kwargs,
    ):
        graphs_lst = []
        Y_exist = Y_list is not None

        to_iterate = (
            zip(X_attr_list, edges_list, Y_list)
            if Y_exist
            else zip(X_attr_list, edges_list)
        )

        first_graph = True

        for i, params in enumerate(to_iterate):
            if Y_exist:
                X_attr, edges, Y = params
            else:
                X_attr, edges = params
                Y = None

            g = GraphDataPair(
                X=X_attr.view(-1, 1),
                edges=edges,
                Y=Y.view(1, 1),
                normalize=False,
                shuffle=False,
                **kwargs,
            )

            graphs_lst.append(g)

            if first_graph:
                all_X = g.X.unsqueeze(0)
                all_edges = g.edges_to_adj(inplace=False).unsqueeze(0)
                all_Y = g.Y.unsqueeze(0) if Y_exist else None

                first_graph = False

            else:
                all_X = torch.cat((all_X, g.X.unsqueeze(0)), dim=0)

                all_edges = torch.cat(
                    (all_edges, g.edges_to_adj(inplace=False).unsqueeze(0)), dim=0
                )
                if Y_exist:
                    all_Y = torch.cat((all_Y, g.Y.unsqueeze(0)), dim=0)

        self.graph_list = graphs_lst
        self.X = torch.squeeze(all_X, 1)
        self.Y = None if not Y_exist else torch.squeeze(all_Y, 1)
        self.edges = all_edges

        return graphs_lst

    def zscore(
        self,
        normalization_params: Tuple[List, List] = None,
        inplace: bool = False,
        return_params: bool = False,
    ):
        if self.normalized:
            print("Data is already normalize!")
            return self if inplace else self.X

        if normalization_params is None:
            normalization_params = (
                self.X.mean(axis=1),
                self.X.std(axis=1),
            )

        X_ = (self.X - normalization_params[0]) / normalization_params[1]

        if inplace:
            self.normalized = True
            self.norm_params = normalization_params

            self.X = X_

        if return_params:
            return normalization_params

        return X_ if not inplace else self

    def denormalize(self, inplace: bool = True):
        if not self.normalized:
            print("Data isn't normalize!")
            return self.X if not inplace else self

        mu, sigma = self.norm_params
        denorm_ = self.X * sigma.view(*self.X.shape[1:]) + mu.view(*self.X.shape[1:])

        if inplace:
            self.normalized = False
            self.norm_params = None

            self.X = denorm_
            return self

        return denorm_

    def _shuffle(self):
        indices = torch.randperm(len(self.X))
        self.X = self.X[indices]
        self.edges = self.edges[indices]
        if self.Y is not None:
            self.Y = self.Y[indices]

        self.graph_list = self.graph_list[indices]

        return self

    def __getitem__(self, index):
        return self.graph_list[index]

    def to_loader(self, **kwargs):
        class GraphsDataset(Dataset):
            def __init__(self, gdp: GraphsDataPair):
                self.gdp = gdp

            def __getitem__(self, idx):
                if self.gdp.Y is not None:
                    return self.gdp.X[idx], self.gdp.edges[idx], self.gdp.Y[idx]

                return self.gdp.X[idx], self.gdp.edges[idx]

            def __len__(self):
                return len(self.gdp)

        ds = GraphsDataset(self)
        return DataLoader(ds, **kwargs)

    def __len__(self):
        return len(self.graph_list)

    def __repr__(self):
        return f"Graph Dataset ({self.name})"

    def __str__(self):
        return self.__repr__()

    def get_X(self):
        return self.X

    def get_Y(self):
        return self.Y

    def get_edges(self):
        return self.edges

    @property
    def num_graphs(self):
        return len(self.graph_list)

    @property
    def num_nodes(self):
        return self.X.shape[0]

    @property
    def num_classes(self):
        if self.Y is None:
            print("No classes in dataset")
            return None

        return len(np.unique(self.Y))

    @property
    def num_features(self):
        return self.X.shape[1]
