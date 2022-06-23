import torch
from graphDataPair import GraphDataPair

import numpy as np
from typing import List, Union, Tuple


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
            self.zcore(normalization_params=normalization_params, inplace=True)
            self.normalized = True

        if shuffle:
            self._shuffle()

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

        for i, *params in to_iterate:
            if Y_exist:
                X_attr, edges, Y = params
            else:
                X_attr, edges = params
                Y = None

            g = GraphDataPair(
                X=X_attr,
                edges=edges,
                Y=Y,
                normalize=False,
                shuffle=False,
                **kwargs,
            )

            graphs_lst.append(g)

            if first_graph:
                all_X = g.X.unsqueeze(0)
                all_edges = g.edges.unsqueeze(0)
                all_Y = g.Y.unsqueeze(0) if Y_exist else None

                first_graph = False

            else:
                all_X = torch.cat((all_X, g.X.unsqueeze(0)), dim=0)
                all_edges = torch.cat((all_edges, g.edges.unsqueeze(0)), dim=0)
                if Y_exist:
                    all_Y = torch.cat((all_Y, g.Y.unsqueeze(0)), dim=0)

        self.graph_list = graphs_lst
        self.X = all_X
        self.edges = all_edges
        self.Y = all_Y

        return graphs_lst

    def zcore(
        self,
        normalization_params: Tuple[List, List] = None,
        inplace: bool = False,
        return_params: bool = False,
    ):
        if self.normalized:
            print("Dataset already normalized")
            return self if inplace else self.X

        if normalization_params is None:
            normalization_params = (
                self.X.mean(axis=1),
                self.X.std(axis=1),
            )

        X_ = (self.X - normalization_params[0]) / normalization_params[1]

        if inplace:
            self.X = X_

        if return_params:
            return normalization_params

        return X_ if not inplace else self

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

    def __len__(self):
        return len(self.graph_list)

    def __repr__(self):
        return f"Graph Dataset ({self.name})"

    def __str__(self):
        return self.__repr__()

    @property
    def num_graphs(self):
        return len(self.graph_list)

    @property
    def num_nodes(self):
        return self.X.shape[1]

    @property
    def num_edges(self):
        return self.edges.shape[1]

    @property
    def num_classes(self):
        if self.Y is None:
            print("No classes in dataset")
            return None

        return self.Y.shape[1]

    @property
    def num_features(self):
        return self.X.shape[2]

