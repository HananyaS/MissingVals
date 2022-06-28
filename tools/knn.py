from tools.distance.heur_euc import HeurEuc
from tools.distance.euclidian import EuclidianDist

import numpy as np
import torch
from datasets.tabDataPair import TabDataPair

from copy import deepcopy
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csr import csr_matrix

from typing import Type


class KNN:
    _dist_dict = {"euclidian": EuclidianDist, "heur_dist": HeurEuc}
    _input_types = [np.ndarray, torch.Tensor, TabDataPair]

    def __init__(self, distance: str, dist_params: dict = {}, k: int = 30):
        self.dist_name = distance
        self.dist = self._dist_dict[distance](**dist_params)
        self.k = k

    def get_edges(
        self,
        data: _input_types,
        return_type: [Type[torch.LongTensor], Type[np.ndarray]] = torch.LongTensor,
        as_adj: bool = True,
    ):
        data_ = deepcopy(data)

        if not isinstance(data_, np.ndarray):
            if isinstance(data_, TabDataPair):
                data_ = data_.X

            data_ = data_.cpu().detach().numpy()

        knn = self._calc_knn_obj(data_)
        return KNN._parse_knn(knn, return_type=return_type, as_adj=as_adj)

    def _calc_knn_obj(self, data: np.ndarray):
        if self.dist_name == "euclidian":
            data_ = np.nan_to_num(data)
            nearest_neighbors = NearestNeighbors(n_neighbors=self.k).fit(data_)
            return nearest_neighbors.kneighbors_graph(data_)

        if self.dist_name == "heur_dist":
            all_dists = np.zeros((data.shape[0], data.shape[0]))
            for i, s1 in enumerate(data):
                for j, s2 in enumerate(data[i:, :]):
                    d_ij = self.dist(s1, s2)
                    all_dists[i, j] = d_ij
                    all_dists[j, i] = d_ij

            nearest_neighbors = NearestNeighbors(
                n_neighbors=self.k, metric="precomputed"
            ).fit(all_dists)
            return nearest_neighbors.kneighbors_graph(all_dists)

        raise NotImplementedError(
            f"Distance metric isn't supported. Available metrics: {','.join(list(self._dist_dict.keys()))}"
        )

    @staticmethod
    def _parse_knn(
        knn_obj: csr_matrix,
        return_type: [Type[torch.LongTensor], Type[np.ndarray]] = torch.LongTensor,
        as_adj: bool = True,
    ):
        adj = knn_obj.asformat(format="array")
        edges = (
            adj.astype(np.int)
            if as_adj
            else np.transpose(np.nonzero(adj)).astype(np.int)
        )

        if return_type == np.ndarray:
            edges = np.array(edges).astype(int)
            return edges

        if return_type in [torch.LongTensor, torch.FloatTensor]:
            return torch.from_numpy(edges).long()

        raise NotImplementedError(
            "Return type is not supported. Available types: torch.LongTensor, np.ndarray"
        )


if __name__ == "__main__":
    from datasets.tabDataset import TabDataset

    data_dir = "../data/Banknote/processed/50"

    td = TabDataset.load(
        data_dir=data_dir,
        normalize=False,
        shuffle=True,
        add_existence_cols=True,
    )

    data_ = td.train

    knn = KNN(distance="heur_dist")
    adj = knn.get_edges(data_, return_type=torch.LongTensor, as_adj=False)
    print(type(adj))
    print(adj.shape)
    print(adj.sum(axis=1))
