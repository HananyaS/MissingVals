import os
import json
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from typing import Union
from datasets.tabDataPair import TabDataPair as TDP


class TabDataset:
    _input_types = Union[torch.Tensor, np.ndarray]

    def __init__(
        self,
        train_X: _input_types,
        train_Y: _input_types = None,
        test_X: _input_types = None,
        test_Y: _input_types = None,
        val_X: _input_types = None,
        val_Y: _input_types = None,
        name: str = "",
        normalize: bool = False,
        shuffle: bool = False,
        add_existence_cols: bool = False,
    ):
        assert test_X is not None or test_Y is None
        assert val_X is not None or val_Y is None

        self.test_exists = test_X is not None
        self.val_exists = val_X is not None
        self.name = name
        self.normalized = normalize

        self.train = TDP(
            X=train_X,
            Y=train_Y,
            name=f"{name} - train",
            normalize=False,
            shuffle=shuffle,
            add_existence_cols=add_existence_cols,
        )

        # if normalize:
        #     _, mu, sigma = self.train.zscore(inplace=True)

        if self.test_exists:
            assert train_X.shape[1] == test_X.shape[1]
            self.test = TDP(
                X=test_X,
                Y=test_Y,
                name=f"{name} - test",
                normalize=False,
                # normalize=normalize,
                # normalization_params=None if not normalize else (mu, sigma),
                shuffle=shuffle,
                add_existence_cols=add_existence_cols,
            )

        if self.val_exists:
            assert train_X.shape[1] == val_X.shape[1]
            self.val = TDP(
                X=val_X,
                Y=val_Y,
                name=f"{name} - val",
                normalize=False,
                # normalize=normalize,
                # normalization_params=None if not normalize else (mu, sigma),
                shuffle=shuffle,
                add_existence_cols=add_existence_cols,
            )

        if normalize:
            self.zscore()

    def __str__(self):
        return f'Dataset "{self.name}" contains {self.train.X.shape[1]} features, including train {f", test" if self.test_exists else ""}, {f"val" if self.val_exists else ""}'

    def zscore(self):
        if not self.normalized:
            _, mu, sigma = self.train.zscore(inplace=True, return_params=True)

            if self.test_exists:
                self.test.zscore(inplace=True, params=(mu, sigma))

            if self.val_exists:
                self.val.zscore(inplace=True, params=(mu, sigma))

            self.normalized = True

        return self

    def get_train_data(self, as_loader=False, **kwargs):
        if as_loader:
            return DataLoader(self.train, **kwargs)

        return self.train

    def get_test_data(self, as_loader=False, **kwargs):
        if self.test_exists:
            if as_loader:
                return DataLoader(self.test, **kwargs)

            return self.test

        raise ValueError("No test data available")

    def get_val_data(self, as_loader=False, **kwargs):
        if self.val_exists:
            if as_loader:
                return DataLoader(self.val, **kwargs)

            return self.val

        raise ValueError("No val data available")

    def get_num_features(self):
        return self.train.X.shape[1]

    def get_train_corr(self, **kwargs):
        return self.train.get_feat_corr(**kwargs)

    def add_existence_cols(self):
        self.train.add_existence_cols(inplace=True)

        if self.test_exists:
            self.test.add_existence_cols(inplace=True)

        if self.val_exists:
            self.val.add_existence_cols(inplace=True)

    def drop_existence_cols(self):
        self.train.drop_existence_cols(inplace=True)

        if self.test_exists:
            self.test.drop_existence_cols(inplace=True)

        if self.val_exists:
            self.val.drop_existence_cols(inplace=True)

    @classmethod
    def load(
        cls,
        data_dir: str,
        train_file_name: str = "train.csv",
        test_file_name: str = "test.csv",
        val_file_name: str = "val.csv",
        include_config: bool = True,
        name: str = None,
        target_col: str = None,
        **kwargs,
    ):
        assert os.path.isdir(data_dir), "Directory doesn't exist!"
        assert os.path.isfile(
            os.path.join(data_dir, train_file_name)
        ), "Train file doesn't exist!"
        assert train_file_name.endswith(".csv"), "Only csv files are supported"
        assert test_file_name is None or os.path.isfile(
            os.path.join(data_dir, test_file_name)
        ), "Test file doesn't exist! If you don't want to use test data, set test_file_name to None"
        assert val_file_name is None or os.path.isfile(
            os.path.join(data_dir, val_file_name)
        ), "Val file doesn't exist! If you don't want to use val data, set val_file_name to None"

        assert include_config or target_col is not None

        if include_config:
            with open(os.path.join(data_dir, "config.json"), "r") as f:
                config = json.load(f)
                target_col = config["target_col"]
                name = config.get("name", "")

        train_data = pd.read_csv(os.path.join(data_dir, train_file_name), index_col=0)
        train_Y = train_data[target_col].values
        train_X = train_data.drop(target_col, axis=1).values

        if test_file_name is not None:
            assert test_file_name.endswith(".csv"), "Only csv files are supported"
            test_data = pd.read_csv(os.path.join(data_dir, test_file_name), index_col=0)
            test_Y = test_data[target_col].values
            test_X = test_data.drop(target_col, axis=1).values

        else:
            test_Y = None
            test_X = None

        if val_file_name is not None:
            assert val_file_name.endswith(".csv"), "Only csv files are supported"
            val_data = pd.read_csv(os.path.join(data_dir, val_file_name), index_col=0)
            val_Y = val_data[target_col].values
            val_X = val_data.drop(target_col, axis=1).values

        else:
            val_Y = None
            val_X = None

        return cls(
            train_X=train_X,
            train_Y=train_Y,
            test_X=test_X,
            test_Y=test_Y,
            val_X=val_X,
            val_Y=val_Y,
            name=name,
            **kwargs,
        )


def test():
    data_dir = "../data/Banknote/processed/50"

    td = TabDataset.load(
        data_dir=data_dir,
        normalize=False,
        shuffle=True,
        add_existence_cols=True,
    )

    td.drop_existence_cols()
    print(td)


if __name__ == "__main__":
    test()
