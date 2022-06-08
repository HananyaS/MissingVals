import json
import pandas as pd
from gfp import GFP
import numpy as np
from utils import timer
from sklearn.metrics.pairwise import euclidean_distances
from itertools import combinations

import torch
from torch.utils.data.dataloader import DataLoader
from datasets import GraphDS

from model import ValuesAndGraphStructure as VGS

from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier as XGB
from sklearn.metrics import roc_auc_score
from torch.nn.functional import one_hot

from distances import hur_dist


def add_existence_columns(data):
    return pd.concat(
        (
            data,
            pd.DataFrame(
                (1 - data.isna()).values,
                columns=[f"{c}_Existence" for c in data.columns],
            ),
        ),
        axis=1,
    )


def split_features_label(all_data, target_col):
    Y = all_data[target_col].values
    X = all_data.drop(columns=[target_col])
    return X, Y


def get_edges(sparse_data):
    dists = euclidean_distances(sparse_data, sparse_data)
    mean_dist = np.mean(dists)
    ind = np.where(dists < mean_dist)
    edges = np.array(list(zip(ind[0], ind[1])))
    return edges


# New distance function
def knn(data, k=30, metric="euclidian", dims_to_ignore=[], metric_params={}):
    if isinstance(data, pd.DataFrame):
        data = data.values

    if metric == "euclidian":
        data = np.nan_to_num(data)
        nearest_neighbors = NearestNeighbors(n_neighbors=k).fit(data)
        return nearest_neighbors.kneighbors_graph(data)

    if metric == "hur_dist":
        all_dists = np.zeros((data.shape[0], data.shape[0]))

        for i, s1 in enumerate(data):
            for j, s2 in enumerate(data[i:, :]):
                # d_ij = hur_dist(s1, s2)
                d_ij = hur_dist(
                    i, j, data, dims_to_ignore=dims_to_ignore, **metric_params
                )
                all_dists[i, j] = d_ij
                all_dists[j, i] = d_ij

        nearest_neighbors = NearestNeighbors(n_neighbors=k, metric="precomputed").fit(
            all_dists
        )
        return nearest_neighbors.kneighbors_graph(all_dists)

    raise NotImplementedError


def get_knn_adj(data, k=30, distance="euclidian", metric_params={}):
    if isinstance(data, pd.DataFrame):
        data = data.values

    # data = np.nan_to_num(data)
    # A = kneighbors_graph(data, k, include_self=False)  #  for euclidian distance
    # dims_to_ignore = range(data.shape[1] // 2, data.shape[1])
    dims_to_ignore = []
    A = knn(
        data,
        k,
        metric=distance,
        dims_to_ignore=dims_to_ignore,
        metric_params=metric_params,
    )

    edges = []

    for i in range(A.shape[0]):
        for j in A.indices[A.indptr[i] : A.indptr[i + 1]]:
            edges.append([i, j])

    edges = np.array(edges)
    return edges


def fill_data_gfp(
    data,
    normalize=True,
    n_iters=50,
    edges="knn",
    distance="euclidian",
    metric_params={},
):
    if "tags" in data.columns:
        data = data.drop(columns=["tags"])

    data_w_existence = add_existence_columns(data)
    filled_data = data.copy(deep=True).fillna(0)

    if edges is "knn":
        # edges = get_knn_adj(data_w_existence.fillna(0))
        edges = get_knn_adj(
            data_w_existence, distance=distance, metric_params=metric_params
        )

    gfp = GFP(iters=n_iters)
    imputed_data = gfp.prop(filled_data.values, edges)

    # z score normalization
    if normalize:
        imputed_data = (imputed_data - np.mean(imputed_data, axis=0)) / np.std(
            imputed_data, axis=0
        )

    edges = np.concatenate((edges, edges[edges[:, 0] != edges[:, 1]][:, ::-1]), axis=0)
    return pd.DataFrame(imputed_data, columns=data.columns), edges


def graph_from_sample(
    sample,
    features_existence,
    weights,
    device,
):
    x = torch.tensor(sample, dtype=torch.float, device=device).view(-1, 1)

    adj = torch.zeros(len(sample), len(sample), dtype=torch.float, device=device)

    features_existence = np.where(features_existence == 1)[0]
    #
    for i, j in combinations(features_existence, r=2):
        adj[i, j] = weights[i, j]
        adj[j, i] = weights[i, j]

    # for i in range(len(sample)):
    #     for j in range(i + 1, len(sample)):
    #         if features_existence[i] == 1 and features_existence[j] == 1:
    #             adj[i, j] = weights[i, j]
    #             adj[j, i] = weights[i, j]

    return x, adj


def graphs_from_samples(
    samples,
    weights,
    features_existence,
    device,
    labels=None,
    test=True,
    graph_from_sample=graph_from_sample,
):
    assert (test and labels is None) or (not test and labels is not None)
    nodes_num = samples.shape[1]
    graphs_ds = GraphDS(nodes_num=nodes_num, test=test, device=device)

    for i in range(len(samples)):
        x, adj = graph_from_sample(
            samples[i],
            features_existence[i],
            weights=weights,
            device=device,
        )
        if test:
            graphs_ds.add_graph(x, adj)
        else:
            graphs_ds.add_graph(x, adj, labels[i])

    return graphs_ds


def run_naive(train_X, train_Y, test_X, test_Y, use_existence_cols=False, params=None):
    if use_existence_cols:
        train_X = add_existence_columns(train_X)
        test_X = add_existence_columns(test_X)

    filled_train = train_X.copy(deep=True).fillna(0)
    filled_test = test_X.copy(deep=True).fillna(0)

    if params is not None:
        xgb = XGB(**params)

    else:
        xgb = XGB(
            n_estimators=25,
            objective="binary:logistic",
            learning_rate=0.03,
            max_depth=8,
            gamma=0.6,
            colsample_bytree=1,
            n_jobs=-1,
        )

    xgb.fit(filled_train, train_Y)
    preds = xgb.predict_proba(filled_test)

    # define the labels as a numpy one hot vector
    labels = np.zeros((len(test_Y), 2), dtype=int)

    for i in range(len(test_Y)):
        labels[i, int(test_Y[i])] = 1

    auc = roc_auc_score(labels, preds)

    return auc


def run_method_2(
    train_X,
    train_Y,
    test_X,
    test_Y,
    distance,
    n_iters=50,
    use_existence_cols=True,
    params=None,
    metric_params={},
):
    existing_features_train = (1 - train_X.isna()).values
    existing_features_test = (1 - test_X.isna()).values

    imputed_train_X, _ = fill_data_gfp(
        train_X, n_iters=n_iters, distance=distance, metric_params=metric_params
    )
    imputed_test_X, _ = fill_data_gfp(
        test_X, n_iters=n_iters, distance=distance, metric_params=metric_params
    )

    if use_existence_cols:
        imputed_train_X = pd.concat(
            [
                imputed_train_X,
                pd.DataFrame(
                    existing_features_train,
                    columns=[f"{c}_Existence" for c in imputed_train_X.columns],
                ),
            ],
            axis=1,
        )
        imputed_test_X = pd.concat(
            [
                imputed_test_X,
                pd.DataFrame(
                    existing_features_test,
                    columns=[f"{c}_Existence" for c in imputed_test_X.columns],
                ),
            ],
            axis=1,
        )

    return run_naive(
        imputed_train_X,
        train_Y,
        imputed_test_X,
        test_Y,
        use_existence_cols=False,
        params=params,
    )


def methods_3_4(
    train_X,
    train_Y,
    test_X,
    test_Y,
    edge_weights,
    distance,
    lr,
    batch_size,
    n_epochs,
    device,
    verbose=True,
    add_isolated_nodes=False,
    params=None,
    metric_params={},
):
    existing_features_train = (1 - train_X.isna()).values
    existing_features_test = (1 - test_X.isna()).values

    if add_isolated_nodes:
        imputed_train_X, _ = fill_data_gfp(
            train_X, normalize=False, distance=distance, metric_params=metric_params
        )
        imputed_test_X, _ = fill_data_gfp(
            test_X, normalize=False, distance=distance, metric_params=metric_params
        )

    else:
        imputed_train_X = train_X.fillna(0)
        imputed_test_X = test_X.fillna(0)

    imputed_train_X = imputed_train_X.values
    imputed_test_X = imputed_test_X.values

    train_graphs_ds = graphs_from_samples(
        imputed_train_X,
        edge_weights,
        existing_features_train,
        device=device,
        labels=train_Y,
        test=False,
    )

    test_graphs_ds = graphs_from_samples(
        imputed_test_X,
        edge_weights,
        existing_features_test,
        device=device,
        labels=None,
        test=True,
    )

    if params is not None:
        ### Replace this script by more elegant one using inspect library
        batch_size = params.get("batch_size", batch_size)
        lr = params.get("lr", lr)
        preweight = params.get("preweight", 4)
        layer_1 = params.get("layer_1", 3)
        layer_2 = params.get("layer_2", 3)
        activation = params.get("activation", "relu")
        dropout = params.get("dropout", 0)

        RECEIVED_PARAMS = {
            "preweight": preweight,
            "layer_1": layer_1,
            "layer_2": layer_2,
            "activation": activation,
            "dropout": dropout,
        }

    else:
        params_file = open("recieved_params.json", "r")
        RECEIVED_PARAMS = json.load(params_file)
        params_file.close()

    train_loader = DataLoader(train_graphs_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs_ds, batch_size=batch_size, shuffle=False)

    model = VGS(
        nodes_number=imputed_train_X.shape[1],
        feature_size=1,
        device=device,
        num_classes=1,
        RECEIVED_PARAMS=RECEIVED_PARAMS,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.fit(
        train_loader=train_loader,
        val_loader=test_loader,
        optimizer=optimizer,
        epochs=n_epochs,
        verbose=verbose,
        plot_loss=True,
        val_labels=test_Y,
    )

    auc = model.evaluate(
        test_loader,
        one_hot(torch.Tensor(test_Y).long(), num_classes=2).float(),
        metric="AUC",
    )

    return auc


@timer
def run_method_3(
    train_X,
    train_Y,
    test_X,
    test_Y,
    edge_weights,
    distance,
    lr,
    batch_size,
    n_epochs,
    device,
    verbose=True,
    params=None,
    metric_params={},
):
    return methods_3_4(
        train_X,
        train_Y,
        test_X,
        test_Y,
        edge_weights,
        distance,
        lr,
        batch_size,
        n_epochs,
        device,
        verbose=verbose,
        add_isolated_nodes=True,
        params=params,
        metric_params=metric_params,
    )


@timer
def run_method_4(
    train_X,
    train_Y,
    test_X,
    test_Y,
    edge_weights,
    distance,
    lr,
    batch_size,
    n_epochs,
    device,
    verbose=True,
    params=None,
    metric_params={},
):
    return methods_3_4(
        train_X,
        train_Y,
        test_X,
        test_Y,
        edge_weights,
        distance,
        lr,
        batch_size,
        n_epochs,
        device,
        verbose=verbose,
        add_isolated_nodes=False,
        params=params,
        metric_params=metric_params,
    )


def methods_5_6(
    train_X,
    train_Y,
    test_X,
    test_Y,
    edge_weights,
    lr,
    batch_size,
    n_epochs,
    device,
    verbose=True,
    params=None,
    return_knn_all_data=False,
):
    existing_features_train = (1 - train_X.isna()).values
    existing_features_test = (1 - test_X.isna()).values

    imputed_train_X = train_X.fillna(0).values
    imputed_test_X = test_X.fillna(0).values

    train_graphs_ds = graphs_from_samples(
        imputed_train_X,
        edge_weights,
        existing_features_train,
        device=device,
        labels=train_Y,
        test=False,
        add_isolated_nodes=False,
    )
    test_graphs_ds = graphs_from_samples(
        imputed_test_X,
        edge_weights,
        existing_features_test,
        device=device,
        labels=None,
        test=True,
        add_isolated_nodes=False,
    )

    if params is not None:
        batch_size = params.get("batch_size", batch_size)
        lr = params.get("lr", lr)
        preweight = params.get("preweight", 4)
        layer_1 = params.get("layer_1", 3)
        layer_2 = params.get("layer_2", 3)
        activation = params.get("activation", "relu")
        dropout = params.get("dropout", 0)

        RECEIVED_PARAMS = {
            "preweight": preweight,
            "layer_1": layer_1,
            "layer_2": layer_2,
            "activation": activation,
            "dropout": dropout,
        }

    else:
        params_file = open("recieved_params.json", "r")
        RECEIVED_PARAMS = json.load(params_file)
        params_file.close()

    train_loader = DataLoader(train_graphs_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs_ds, batch_size=batch_size, shuffle=False)

    model = VGS(
        nodes_number=imputed_train_X.shape[1],
        feature_size=1,
        device=device,
        num_classes=1,
        RECEIVED_PARAMS=RECEIVED_PARAMS,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.fit(
        train_loader=train_loader,
        val_loader=test_loader,
        optimizer=optimizer,
        epochs=n_epochs,
        verbose=verbose,
        plot_loss=True,
        val_labels=test_Y,
    )

    one_before_last_layer_train = torch.Tensor().to(device)
    one_before_last_layer_test = torch.Tensor().to(device)

    with torch.no_grad():
        for batch in train_loader:
            xs, adjs, _ = batch

            xs = xs.to(device)
            adjs = adjs.to(device)

            one_before_last_layer_train = torch.cat(
                (
                    one_before_last_layer_train,
                    model.extract_one_before_last_layer(xs, adjs),
                ),
                dim=0,
            )

        for batch in test_loader:
            xs, adjs = batch

            xs = xs.to(device)
            adjs = adjs.to(device)

            one_before_last_layer_test = torch.cat(
                (
                    one_before_last_layer_test,
                    model.extract_one_before_last_layer(xs, adjs),
                ),
                dim=0,
            )

    one_before_last_layer_train = pd.DataFrame(
        one_before_last_layer_train.detach().cpu().numpy()
    )
    one_before_last_layer_test = pd.DataFrame(
        one_before_last_layer_test.detach().cpu().numpy()
    )

    train_labels = train_graphs_ds.labels.detach().cpu().numpy()

    test_labels = (
        one_hot(torch.Tensor(test_Y).long(), num_classes=2)
        .float()
        .detach()
        .cpu()
        .numpy()
        .argmax(axis=1)
    )

    if return_knn_all_data:
        if params is not None:
            k = params.get("k", 30)
        knn_edges = get_knn_adj(
            np.concatenate((imputed_train_X, imputed_test_X), axis=0),
            k=k,
        )
        return (
            pd.concat((one_before_last_layer_train, one_before_last_layer_test)),
            np.concatenate(train_labels, test_labels),
            list(range(len(train_labels))),
            knn_edges,
        )

    return (
        one_before_last_layer_train,
        train_labels,
        one_before_last_layer_test,
        test_labels,
    )


def run_method_5(
    train_X,
    train_Y,
    test_X,
    test_Y,
    edge_weights,
    lr,
    batch_size,
    n_epochs,
    device,
    verbose=True,
    params=None,
):
    one_before_last_layer, labels, train_indices, knn_edges = methods_5_6(
        train_X,
        train_Y,
        test_X,
        test_Y,
        edge_weights,
        lr,
        batch_size,
        n_epochs,
        device,
        verbose=True,
        params=None,
        return_knn_all_data=True,
    )

    # implement node classification
    raise NotImplementedError


def run_method_6(
    train_X,
    train_Y,
    test_X,
    test_Y,
    edge_weights,
    lr,
    batch_size,
    n_epochs,
    device,
    verbose=True,
    params=None,
):
    (
        one_before_last_layer_train,
        train_labels,
        one_before_last_layer_test,
        test_labels,
    ) = methods_5_6(
        train_X,
        train_Y,
        test_X,
        test_Y,
        edge_weights,
        lr,
        batch_size,
        n_epochs,
        device,
        verbose,
        params,
        return_knn_all_data=False,
    )

    xgb_params = None

    if params is not None:
        objective = params.get("objective", "binary:logistic")
        learning_rate = params.get("learning_rate", 0.3)
        max_depth = params.get("max_depth", 8)
        colsample_bytree = params.get("colsample_bytree", 1)
        gamma = params.get("gamma", 0.6)
        n_estimators = params.get("n_estimators", 25)
        xgb_params = {
            "objective": objective,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "colsample_bytree": colsample_bytree,
            "gamma": gamma,
            "n_estimators": n_estimators,
        }

    return run_naive(
        one_before_last_layer_train,
        train_labels,
        one_before_last_layer_test,
        test_labels,
        params=xgb_params,
    )
