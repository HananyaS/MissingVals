import torch
from datasets.graphsDataPair import GraphsDataPair
from datasets.tabDataset import TabDataset

import numpy as np

from typing import List, Union


class GraphsDataset:
    _input_types = Union[torch.Tensor, np.ndarray]

    def __init__(
        self,
        name: str,
        train: GraphsDataPair,
        test: GraphsDataPair = None,
        val: GraphsDataPair = None,
        normalize: bool = True,
    ):
        self.name = name
        self.normalized = False

        self.train = train
        self.train.denormalize(inplace=True)

        self.test_exists = test is not None
        self.val_exists = val is not None

        if self.test_exists:
            assert (
                test.num_features == train.num_features
            ), "Test doesn't have the same number of features as in train"
            self.test = test
            self.test.denormalize(inplace=True)

        if self.val_exists:
            assert (
                val.num_features == train.num_features
            ), "Validation doesn't have the same number of features as in train"
            self.val = val
            self.val.denormalize(inplace=True)

        if normalize:
            self.zscore()

    @classmethod
    def from_attributes(
        cls,
        train_X_attributes: List[_input_types],
        train_edges: List[_input_types],
        train_Y: List[_input_types] = None,
        test_X_attributes: List[_input_types] = None,
        test_edges: List[_input_types] = None,
        test_Y: List[_input_types] = None,
        val_X_attributes: List[_input_types] = None,
        val_edges: List[_input_types] = None,
        val_Y: List[_input_types] = None,
        name: str = "",
        normalize: bool = False,
        **kwargs,
    ):
        test_exists = test_X_attributes is not None
        val_exists = val_X_attributes is not None

        train = GraphsDataPair(
            X_list=train_X_attributes,
            edges_list=train_edges,
            Y_list=train_Y,
            name=f"{name} - train",
            normalize=False,
            **kwargs,
        )

        if test_exists:
            test = GraphsDataPair(
                X_list=test_X_attributes,
                edges_list=test_edges,
                Y_list=test_Y,
                name=f"{name} - test",
                normalize=False,
                **kwargs,
            )
        else:
            test = None

        if val_exists:
            val = GraphsDataPair(
                X_list=val_X_attributes,
                edges_list=val_edges,
                Y_list=val_Y,
                name=f"{name} - val",
                normalize=False,
                **kwargs,
            )
        else:
            val = None

        return cls(name=name, train=train, val=val, test=test, normalize=normalize)

    @classmethod
    def from_tab(cls, tab_data: TabDataset, **kwargs):
        train = GraphsDataPair.from_tab(
            tab_data=tab_data.train, include_edge_weights=True, **kwargs
        )
        val = (
            None
            if not tab_data.val_exists
            else GraphsDataPair.from_tab(
                tab_data=tab_data.val, include_edge_weights=True, **kwargs
            )
        )
        test = (
            None
            if not tab_data.test_exists
            else GraphsDataPair.from_tab(
                tab_data=tab_data.test, include_edge_weights=True, **kwargs
            )
        )

        graphs_dataset = cls(
            train=train,
            val=val,
            test=test,
            normalize=False,
            name=f"{tab_data.name} - graph",
        )

        graphs_dataset.normalized = tab_data.normalized

        return graphs_dataset

    def zscore(self):
        _, mu, sigma = self.train.zscore(return_params=True, inplace=True)
        if self.test_exists:
            self.test.zscore(normalization_params=(mu, sigma), inplace=True)
        if self.val_exists:
            self.val.zscore(normalization_params=(mu, sigma), inplace=True)

        self.normalized = True

        return self

    def denormalize(self):
        if not self.normalized:
            print("Data isn't normalize!")
            return self

        self.train.denormalize(inplace=True)

        if self.test_exists:
            self.test.denormalize(inplace=True)

        if self.val_exists:
            self.val.denormalize(inplace=True)

        self.normalized = False

        return self

    def get_train_data(self, as_loader: bool = False, **kwargs):
        train = self.train
        if as_loader:
            train = train.to_loader(**kwargs)

        return train

    def get_test_data(self, as_loader: bool = False, **kwargs):
        assert self.test_exists, "Test data is not available"

        test = self.test
        if as_loader:
            test = test.to_loader()

        return test

    def get_val_data(self, as_loader: bool = False, **kwargs):
        assert self.val_exists, "Validation data is not available"

        val = self.val

        if as_loader:
            val = val.to_loader()

        return val

    def __str__(self):
        return f'Dataset "{self.name}" contains {self.train.X.shape[1]} features, including train {f", test" if self.test_exists else ""}, {f"val" if self.val_exists else ""}'

    def __repr__(self):
        return self.__str__()

    @property
    def num_features(self):
        return self.train.num_features

    @property
    def num_classes(self):
        return self.train.num_classes

    """
    @property
    def num_train_nodes(self):
        return self.train_graph.num_nodes

    @property
    def num_test_nodes(self):
        if not self.test_exists:
            print("Test data is not available")
            return None

        return self.test_graph.num_nodes

    @property
    def num_val_nodes(self):
        if not self.val_exists:
            print("Validation data is not available")
            return None

        return self.val_graph.num_nodes
    
    @property
    def num_train_edges(self):
        return self.train_graph.num_edges

    @property
    def num_test_edges(self):
        if not self.test_exists:
            print("Test data is not available")
            return None

        return self.test_graph.num_edges

    @property
    def num_val_edges(self):
        if not self.val_exists:
            print("Validation data is not available")
            return None

        return self.val_graph.num_edges
    
    """

    @property
    def train_len(self):
        return len(self.train)

    @property
    def test_len(self):
        if not self.test_exists:
            print("Test data is not available")
            return None

        return len(self.test)

    @property
    def val_len(self):
        if not self.val_exists:
            print("Validation data is not available")
            return None

        return len(self.val)


if __name__ == "__main__":
    from time import time
    from models.graphClassification import ValuesAndGraphStructure

    st = time()

    data_dir = "../data/Banknote/processed/90"

    td = TabDataset.load(
        data_dir=data_dir,
        normalize=True,
        shuffle=True,
        add_existence_cols=False,
    )

    graphs_dataset = GraphsDataset.from_tab(
        td,
        fill_data_method="gfp",
        store_as_adj=True,
        # knn_kwargs={"distance": "heur_dist", "dist_params": {"alpha": .5, "beta": .5}},
    )

    train_graphs = graphs_dataset.get_train_data(as_loader=True, batch_size=32)
    val_graphs = graphs_dataset.get_val_data(as_loader=True, batch_size=32)
    test_graphs = graphs_dataset.get_test_data(as_loader=True, batch_size=32)

    params = {
        "preweight": 5,
        "layer_1": 12,
        "layer_2": 7,
        "activation": "elu",
        "dropout": 0.3,
    }

    model = ValuesAndGraphStructure(input_example=train_graphs, RECEIVED_PARAMS=params)

    forward_args = model.transform_input(
        [train_graphs.dataset.gdp.get_X(), train_graphs.dataset.gdp.get_edges(), None]
    )
    graph_embeddings = model.forward_one_before_last_layer(*forward_args[0])
    # print(len(graph_embeddings))
    # print(len(train_graphs.dataset))
    model.fit(
        train_loader=train_graphs,
        val_loader=val_graphs,
        auc_plot_path="auc.png",
        loss_plot_path="loss.png",
        lr=0.01,
        n_epochs=30,
        verbose=True,
    )
    #
    # test_auc = model.evaluate(loader=test_graph)
    # print(f"Test AUC: {test_auc:.4f}")

    print(f"Time elapsed: {time() - st} seconds")
