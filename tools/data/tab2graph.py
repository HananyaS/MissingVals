import torch
import numpy as np

from datasets.tabDataPair import TabDataPair

# from datasets.graphsDataPair import GraphsDataPair

from tools.gfp import GFP
from tools.knn import KNN

from copy import deepcopy
from typing import Union, Type

from itertools import combinations


def tab2graphs(
    tab_data: TabDataPair,
    store_as_adj: bool = True,
    name: str = None,
    include_edge_weights: bool = False,
    edge_weights_method: str = "corr",
    fill_data_method: str = "gfp",
    knn_kwargs: dict = {"distance": "euclidian"},
    gfp_kwargs: dict = {},
):
    assert not include_edge_weights or edge_weights_method in ["corr", None]
    assert fill_data_method in ["gfp", "zeros"]

    tab_data_ = deepcopy(tab_data)

    if fill_data_method == "gfp":
        inter_samples_edges = get_knn_adj(tab_data_, **knn_kwargs)
        gfp = GFP(**gfp_kwargs)
        imputed_data_ = gfp.prop(tab_data_.X, inter_samples_edges)

    elif fill_data_method == "zeros":
        imputed_data_ = tab_data_.fill_na(inplace=False)

    else:
        raise NotImplementedError(f"Method {fill_data_method} is not supported!")

    if include_edge_weights:
        if edge_weights_method == "corr":
            edge_weights = tab_data_.get_feat_corr(abs_=True, fill_na_first=False)
        else:
            raise NotImplementedError

    X_list, edge_list = [], []

    if tab_data_.Y is not None:
        Y_list = []

    for i in range(tab_data_.X.shape[0]):
        X_list.append(imputed_data_[i])
        edge_list.append(
            edges_from_sample(sample=tab_data_.X[i], edge_weights=edge_weights)
        )

        if tab_data_.Y is not None:
            Y_list.append(tab_data_.Y[i])

    if name is None:
        name = f"{tab_data_.name} - Graph"

    # gdp = GraphsDataPair(
    #     X_list=X_list,
    #     edges_list=edge_list,
    #     Y_list=Y_list,
    #     name=name,
    #     include_edge_weights=include_edge_weights,
    #     given_as_adj=True,
    #     store_as_adj=store_as_adj,
    #     normalize=False,
    #     normalization_params=None,
    #     shuffle=False,
    #     add_existence_cols=False,
    # )

    # if tab_data_.normalized:
    #     gdp.zscore(normalization_params=tab_data_.norm_params, inplace=True)
    #
    # return gdp

    kwargs = {
        "X_list": X_list,
        "edges_list": edge_list,
        "Y_list": Y_list,
        "name": name,
        "include_edge_weights": include_edge_weights,
        "given_as_adj": True,
        "store_as_adj": store_as_adj,
        "normalize": False,
        "normalization_params": None,
        "shuffle": False,
        "add_existence_cols": False,
    }

    return kwargs, tab_data_.normalized, tab_data_.norm_params


def get_knn_adj(tab_data: TabDataPair, **kwargs):
    knn = KNN(**kwargs)
    edges = knn.get_edges(tab_data.X, as_adj=True)
    return edges


def edges_from_sample(
    sample: Union[Type[np.ndarray], Type[torch.Tensor]],
    edge_weights: Union[Type[torch.Tensor], Type[np.ndarray]] = None,
):
    edge_weights = (
        edge_weights
        if edge_weights is not None
        else np.ones((sample.shape[1], sample.shape[1]))
    )
    adj = torch.zeros(len(sample), len(sample), dtype=torch.float)

    features_existence = np.where(sample == sample)[0]

    for i, j in combinations(features_existence, r=2):
        adj[i, j] = edge_weights[i, j]
        adj[j, i] = edge_weights[i, j]

    return adj
