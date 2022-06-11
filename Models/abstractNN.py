import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
from abc import abstractmethod
from torch.utils.data import DataLoader

from typing import Callable, List
from sklearn.metrics import roc_auc_score

from .abstractModel import AbstractModel


class AbstractNN(nn.Module, AbstractModel):
    def __init__(self, device: torch.device):
        super(AbstractNN, self).__init__()
        self.device = device

    @abstractmethod
    def forward_one_before_last_layer(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def forward_last_layer(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        x = self.forward_one_before_last_layer(*args, **kwargs)
        x = self.forward_last_layer(x)
        return x

    @abstractmethod
    def _transform_input(self, data: torch.Tensor):
        raise NotImplementedError  # TODO: specify device

    @abstractmethod
    def _eval_loss(
        self,
        output: torch.Tensor,
        labels: torch.Tensor,
        loss_func: torch.nn.modules.loss,
        n_classes: int = 2,
    ) -> torch.nn.modules.loss:
        raise NotImplementedError

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int,
        lr: float,
        weight_decay: float,
        optimizer: torch.optim = optim.Adam,
        verbose: bool = False,
        criterion: torch.nn.modules.loss = nn.CrossEntropyLoss(),
        labels_from_loader: Callable = lambda loader: loader.dataset.get_labels(),
        metric: str = "auc",
        plot_results: bool = True,
        save_results: bool = True,
        auc_plot_path: str = None,
        loss_plot_path: str = None,
        show_results: bool = True,
        save_model: bool = True,
    ):
        assert metric in ["auc", "accuracy"]
        assert not save_results or plot_results, "Plotting is required to save results"
        assert (
            not save_results or auc_plot_path is not None
        ), "Please provide a path to save the AUC plot"
        assert (
            not save_results or loss_plot_path is not None
        ), "Please provide a path to save the loss plot"

        self.train()
        train_losses, val_losses = [], []
        train_aucs, val_aucs = [], []

        optimizer = optimizer(self.parameters(), lr=lr, weight_decay=weight_decay)

        for epoch in range(1, n_epochs + 1):
            total_train_loss = 0

            for data in train_loader:
                optimizer.zero_grad()

                input_data, labels = self._transform_input(data)
                output = self(*input_data)

                loss = self._eval_loss(output, labels, criterion, n_classes=train_loader.dataset.get_num_classes())
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item() * (
                    input_data.size(0)
                    * (1 if criterion.reduction == "sum" else input_data.size(0))
                )

            total_train_loss = total_train_loss / len(train_loader.dataset)

            # calculate loss on validation set
            with torch.no_grad():
                self.eval()
                pred_val = self.predict(val_loader, probs=True)
                val_loss = self._eval_loss(
                    pred_val,
                    labels_from_loader(val_loader),
                    criterion,
                    n_classes=train_loader.dataset.get_num_classes(),
                )
            train_auc = self.evaluate(
                train_loader, metric=metric, labels_from_loader=labels_from_loader
            )
            val_auc = self.evaluate(
                val_loader, metric=metric, labels_from_loader=labels_from_loader
            )

            train_losses.append(total_train_loss)
            val_losses.append(val_loss)
            train_aucs.append(train_auc)
            val_aucs.append(val_auc)

            if verbose:
                print(
                    f"Epoch {epoch}/{n_epochs}:\n"
                    f"\tTrain loss: {total_train_loss:.4f}\n"
                    f"\tVal loss: {val_loss:.4f}\n"
                    f"\tTrain AUC: {train_auc:.4f}\n"
                    f"\tVal AUC: {val_auc:.4f}"
                )

            if plot_results:
                self._plot_results(
                    train_losses,
                    val_losses,
                    train_aucs,
                    val_aucs,
                    save_results,
                    auc_plot_path,
                    loss_plot_path,
                    show_results,
                )

            if save_model:
                self._save_model(epoch)

    @staticmethod
    def _plot_results(
        train_losses: List,
        val_losses: List,
        train_aucs: List,
        val_aucs: List,
        save_results: bool,
        auc_plot_path: str,
        loss_plot_path: str,
        show_results: bool,
    ):
        plt.clf()
        plt.plot(range(1, 1 + len(train_losses)), train_losses, label="Train")
        plt.plot(range(1, 1 + len(val_losses)), val_losses, label="Val")
        plt.legend()
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        if save_results:
            plt.savefig(loss_plot_path)

        if show_results:
            plt.show()

        plt.clf()
        plt.plot(range(1, 1 + len(train_aucs)), train_aucs, label="Train")
        plt.plot(range(1, 1 + len(val_aucs)), val_aucs, label="Val")
        plt.legend()
        plt.title("AUC")
        plt.xlabel("Epoch")
        plt.ylabel("AUC")
        if save_results:
            plt.savefig(auc_plot_path)

        if show_results:
            plt.show()

    def _save_model(self, path: str = "model.pt"):
        torch.save(self.state_dict(), path)
        return self

    def predict(
        self,
        loader: DataLoader,
        probs: bool = False,
        pred_from_output: Callable = lambda output: torch.argmax(output, 1),
    ) -> torch.Tensor:
        self.eval()
        preds = torch.Tensor([]).view(
            -1, loader.dataset.get_num_classes() if probs else 1
        )

        with torch.no_grad():
            for data in loader:
                input_data, _ = self._transform_input(data)
                output = self(*input_data)
                if probs:
                    preds = torch.cat((preds, output), 0)
                else:
                    preds = torch.cat(
                        (preds, torch.argmax(output, dim=1).view(-1, 1)), 0
                    )

        return preds

    def evaluate(self, metric: str = "auc", **kwargs) -> float:
        if metric == "auc":
            return self._eval_auc(**kwargs)

        raise NotImplementedError

    def _eval_auc(
        self,
        loader: DataLoader,
        labels_from_loader: Callable = lambda loader: loader.dataset.get_labels(),
    ) -> float:
        self.eval()
        preds = self.predict(loader, probs=True)
        labels = labels_from_loader(loader)

        return roc_auc_score(labels, preds)
