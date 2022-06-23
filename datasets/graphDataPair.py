import torch
from tabDataPair import TabDataPair
from typing import Type


class GraphDataPair(TabDataPair):
    def __init__(
        self,
        edges: super()._input_types,
        given_as_adj: bool = False,
        store_as_adj: bool = False,
        include_edge_weights: bool = False,
        **kwargs,
    ):
        super(GraphDataPair, self).__init__(**kwargs)
        self.edges = self._transform_types(
            edges, Type[float] if include_edge_weights else Type[int]
        )
        self.edges = self._transform_edge_format(
            self.edges,
            given_as_adj=given_as_adj,
            store_as_adj=store_as_adj,
            weights=include_edge_weights,
        )

        self.include_edge_weights = include_edge_weights
        self.as_adj = store_as_adj

    @staticmethod
    def _transform_edge_format(
        edges: super()._input_types,
        given_as_adj: bool,
        store_as_adj: bool = False,
        weights: bool = False,
    ) -> torch.Tensor:
        required_edge_dim = 2 if not weights else 3
        assert len(edges.shape) == 2, "edges_list must be a 2D array"
        assert given_as_adj or required_edge_dim in edges.shape
        assert not given_as_adj or edges.shape[0] == edges.shape[1]

        if not given_as_adj:
            if edges.shape[0] == required_edge_dim:
                edges = edges.T

        if given_as_adj == store_as_adj:
            return edges

        if not store_as_adj:
            edge_list = torch.nonzero(edges).view(-1, 2)

            if not weights:
                return edge_list.long()

            return torch.cat(
                (edge_list, edges[edge_list[:, 0], edge_list[:, 1]].view(-1, 1)), dim=1
            )

        adj = torch.zeros(edges.shape[0], edges.shape[0])

        if weights:
            adj[edges[:, 0], edges[:, 1]] = edges[:, 2]

        else:
            adj[edges[:, 0], edges[:, 1]] = 1

        return adj

    @property
    def num_nodes(self):
        return self.X.shape[0]

    @property
    def num_edges(self):
        return self.edges.shape[0]

    @property
    def get_nodes(self):
        return self.X

    @property
    def get_edges(self):
        return self.edges

    def edges_to_adj(self, inplace: bool = True):
        if self.as_adj:
            return self if inplace else self.edges

        adj = torch.zeros(self.X.shape[0], self.X.shape[0])
        adj[self.edges[:, 0], self.edges[:, 1]] = (
            (self.edges[:, 2]) if self.include_edge_weights else 1
        )
        if inplace:
            self.edges = adj
            return self

        return adj

    def edges_to_lst(self, inplace: bool = True):
        if not self.as_adj:
            return self if inplace else self.edges

        edges = torch.nonzero(self.edges).view(-1, 2)
        if self.include_edge_weights:
            edges = torch.cat(
                (edges, self.edges[edges[:, 0], edges[:, 1]].view(-1, 1)), dim=1
            )

        if inplace:
            self.edges = edges
            return self

        return edges

    def _shuffle(self, inplace: bool = True):
        idx = torch.randperm(self.X.shape[0])

        convertor = self.edges_to_adj if self.as_adj else self.edges_to_lst
        edges = self.edges_to_adj(inplace=False)[idx][:, idx]

        if inplace:
            if self.Y is not None:
                self.X, self.Y, self.edges = self.X[idx], self.Y[idx], convertor(edges)

            else:
                self.X = self.X[idx]

            return self

        else:
            if self.Y is not None:
                return self.X[idx], edges, self.Y[idx]

            return self.X[idx], edges

    @classmethod
    def load(
        cls,
        data_dir: str,
        data_file_name: str,
        include_config: bool = True,
        name: str = None,
        target_col: str = None,
        **kwargs,
    ):
        raise Exception("Cannot load graph data from a file")

