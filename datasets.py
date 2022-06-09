import os
import json
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from typing import Union, List, Tuple


class TabDataPair(Dataset):
    _input_types = Union[np.ndarray, torch.Tensor]

    def __init__(
        self,
        X: _input_types,
        name: str = "",
        Y: _input_types = None,
        normalize: bool = False,
        normalization_params: Tuple[List, List] = None,
        shuffle: bool = False,
        add_existence_cols: bool = False,
    ):
        assert Y is None or X.shape[0] == Y.shape[0]
        self.X = self._transform_types(X, float)
        self.Y = self._transform_types(Y, int) if Y is not None else None

        if normalize:
            self.zscore(normalization_params, inplace=True, return_params=False)

        if shuffle:
            self._shuffle()

        self.existence_cols = []

        if add_existence_cols:
            self.add_existence_cols(inplace=True)

        self.name = name
        self.train = self.Y is not None

    @staticmethod
    def _transform_types(
        data: _input_types, _naive_type: Union[int, float] = float
    ) -> torch.Tensor:
        if type(data) == np.ndarray:
            data = torch.from_numpy(data)

        data = data.long() if _naive_type == int else data.float()

        return data

    def _shuffle(self, inplace: bool = True):
        idx = torch.randperm(self.X.shape[0])

        if inplace:
            if self.Y is not None:
                self.X, self.Y = self.X[idx], self.Y[idx]

            else:
                self.X = self.X[idx]

            return self

        else:
            if self.Y is not None:
                return self.X[idx], self.Y[idx]

            return self.X[idx]

    def zscore(
        self,
        params: Tuple[List, List] = None,
        inplace: bool = True,
        return_params: bool = False,
    ):
        if params is not None:
            mu, sigma = params
            assert len(mu) == len(sigma) == self.X.shape[1]

        else:
            mu = torch.nanmean(self.X, axis=0)
            sigma = torch.from_numpy(np.nanstd(self.X.cpu().detach().numpy(), axis=0))

        if inplace:
            self.X = (self.X - mu) / sigma

            if return_params:
                return self, mu, sigma

            return self

        if return_params:
            return (self.X - mu) / sigma, mu, sigma

        return (self.X - mu) / sigma

    def __getitem__(self, idx):
        if self.train:
            return self.X[idx], self.Y[idx]

        return self.X[idx]

    def __len__(self):
        return self.X.shape[0]

    def __str__(self):
        return f'Dataset "{self.name}" contains {self.X.shape[0]} samples and {self.X.shape[1]} features, in {"train" if self.train else "test"} mode '

    def get_feat_corr(self, abs_: bool = False, fill_na_first: bool = False):
        df = pd.DataFrame(self.X.cpu().detach().numpy())

        if fill_na_first:
            df = df.fillna(df.mean())

        corr = df.corr()

        if abs_:
            corr = abs(corr)

        return corr

    def add_existence_cols(self, inplace: bool = True):
        assert len(self.existence_cols) == 0, "Existence columns were already added!"
        existence_cols = 1 - torch.isnan(self.X).float()

        if inplace:
            self.X = torch.cat((self.X, existence_cols), dim=1)
            self.existence_cols = list(range(int(self.X.shape[1] / 2), self.X.shape[1]))
            return self

        return torch.cat((self.X, existence_cols), dim=1)

    def drop_existence_cols(self, inplace: bool = True):
        assert len(self.existence_cols) > 0, "Existence columns were not added!"
        if inplace:
            self.X = self.X[:, : -len(self.existence_cols)]
            self.existence_cols = []
            return self

        return self.X[:, : -len(self.existence_cols)]

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
        assert os.path.isdir(data_dir), "Directory doesn't exist!"
        assert os.path.isfile(
            os.path.join(data_dir, data_file_name)
        ), "Data file doesn't exist!"
        assert data_file_name.endswith(".csv"), "Only csv files are supported"
        assert include_config or target_col is not None

        all_data = pd.read_csv(os.path.join(data_dir, data_file_name), index_col=0)

        if include_config:
            with open(os.path.join(data_dir, "config.json"), "r") as f:
                config = json.load(f)

            target_col = config["target_col"]
            name = config.get("name", "")

        Y = all_data[target_col].values
        X = all_data.drop(target_col, axis=1).values

        return cls(
            X=X,
            Y=Y,
            name=f'{name} - {"".join(data_file_name.split(".")[:-1])}',
            **kwargs,
        )


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

        self.train = TabDataPair(
            X=train_X,
            Y=train_Y,
            name=f"{name} - train",
            normalize=False,
            shuffle=shuffle,
            add_existence_cols=add_existence_cols,
        )

        if normalize:
            _, mu, sigma = self.train.zscore(return_params=True, inplace=True)

        if self.test_exists:
            assert train_X.shape[1] == test_X.shape[1]
            self.test = TabDataPair(
                X=test_X,
                Y=test_Y,
                name=f"{name} - test",
                normalize=normalize,
                normalization_params=None if not normalize else (mu, sigma),
                shuffle=shuffle,
                add_existence_cols=add_existence_cols,
            )

        if self.val_exists:
            assert train_X.shape[1] == val_X.shape[1]
            self.val = TabDataPair(
                X=val_X,
                Y=val_Y,
                name=f"{name} - val",
                normalize=normalize,
                normalization_params=None if not normalize else (mu, sigma),
                shuffle=shuffle,
                add_existence_cols=add_existence_cols,
            )

    def __str__(self):
        return f'Dataset "{self.name}" contains {self.train.X.shape[1]} features, including train {f", test" if self.test_exists else ""}, {f"val" if self.val_exists else ""}'

    def get_train_data(self):
        return self.train

    def get_test_data(self):
        if self.test_exists:
            return self.test

        raise ValueError("No test data available")

    def get_val_data(self):
        if self.val_exists:
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
        ), "Test file doesn't exist!"
        assert val_file_name is None or os.path.isfile(
            os.path.join(data_dir, val_file_name)
        ), "Val file doesn't exist!"

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
    data_dir = "Data/RonsData/processed"
    data_file_name = "train.csv"

    tdp = TabDataPair.load(
        data_dir=data_dir,
        data_file_name=data_file_name,
        include_config=True,
        normalize=True,
        add_existence_cols=False,
    )

    tdp.add_existence_cols(inplace=True)
    td = TabDataset.load(
        data_dir=data_dir, normalize=False, shuffle=True, add_existence_cols=True
    )
    td.drop_existence_cols()

    print(td.get_train_corr(abs_=True))
    print(td.get_train_corr(abs_=True, fill_na_first=True))


if __name__ == "__main__":
    test()
