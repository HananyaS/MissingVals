import torch
from graphsDataPair import GraphsDataPair

import numpy as np

from typing import List, Union


class GraphDataset:
    _input_types = Union[torch.Tensor, np.ndarray]

    def __init__(
        self,
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
        self.name = name
        self.test_exist = test_X_attributes is not None
        self.val_exist = val_X_attributes is not None

        self.train = GraphsDataPair(
            X_list=train_X_attributes,
            edges_list=train_edges,
            Y_list=train_Y,
            name=f"{name} - train",
            normalize=False,
            **kwargs,
        )

        if self.test_exist:
            self.test = GraphsDataPair(
                X_list=test_X_attributes,
                edges_list=test_edges,
                Y_list=test_Y,
                name=f"{self.name} - test",
                normalize=False,
                **kwargs,
            )

        if self.val_exist:
            self.val = GraphsDataPair(
                X_list=val_X_attributes,
                edges_list=val_edges,
                Y_list=val_Y,
                name=f"{self.name} - val",
                normalize=False,
                **kwargs,
            )

        self.normalized = False

        if normalize:
            self.zcore()
            self.normalized = True

    def zcore(self):
        _, mu, sigma = self.train.zcore(return_params=True, inplace=True)
        if self.test_exist:
            self.test.zcore(normalization_params=(mu, sigma), inplace=True)
        if self.val_exist:
            self.val.zcore(normalization_params=(mu, sigma), inplace=True)

        return self

    def get_train_data(self):
        return self.train

    def get_test_data(self):
        assert self.test_exist, "Test data is not available"
        return self.test

    def get_val_data(self):
        assert self.val_exist, "Validation data is not available"
        return self.val

    def __str__(self):
        return f'Dataset "{self.name}" contains {self.train.X.shape[1]} features, including train {f", test" if self.test_exist else ""}, {f"val" if self.val_exist else ""}'

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
        if not self.test_exist:
            print("Test data is not available")
            return None

        return self.test.num_nodes

    @property
    def num_val_nodes(self):
        if not self.val_exist:
            print("Validation data is not available")
            return None

        return self.val.num_nodes

    @property
    def num_train_edges(self):
        return self.train.num_edges

    @property
    def num_test_edges(self):
        if not self.test_exist:
            print("Test data is not available")
            return None

        return self.test.num_edges

    @property
    def num_val_edges(self):
        if not self.val_exist:
            print("Validation data is not available")
            return None

        return self.val.num_edges

    @property
    def train_len(self):
        return len(self.train)

    @property
    def test_len(self):
        if not self.test_exist:
            print("Test data is not available")
            return None

        return len(self.test)

    @property
    def val_len(self):
        if not self.val_exist:
            print("Validation data is not available")
            return None

        return len(self.val)

