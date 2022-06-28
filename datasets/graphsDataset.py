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
            train=train, val=val, test=test, normalize=False, name=f'{tab_data.name} - graph'
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

    def get_train_data(self):
        return self.train

    def get_test_data(self):
        assert self.test_exists, "Test data is not available"
        return self.test

    def get_val_data(self):
        assert self.val_exists, "Validation data is not available"
        return self.val

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

    @property
    def num_train_nodes(self):
        return self.train.num_nodes

    @property
    def num_test_nodes(self):
        if not self.test_exists:
            print("Test data is not available")
            return None

        return self.test.num_nodes

    @property
    def num_val_nodes(self):
        if not self.val_exists:
            print("Validation data is not available")
            return None

        return self.val.num_nodes

    @property
    def num_train_edges(self):
        return self.train.num_edges

    @property
    def num_test_edges(self):
        if not self.test_exists:
            print("Test data is not available")
            return None

        return self.test.num_edges

    @property
    def num_val_edges(self):
        if not self.val_exists:
            print("Validation data is not available")
            return None

        return self.val.num_edges

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
