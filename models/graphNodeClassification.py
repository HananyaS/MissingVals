import argparse
import sys

sys.path.append("../")

import torch
from torch import nn
from torch import optim

from models.abstractModel import AbstractModel
from models.nodeClassification import NodeClassification
from models.graphClassification import ValuesAndGraphStructure

from datasets.tabDataset import TabDataset

from typing import Dict

from sklearn.metrics import roc_auc_score
from torch.nn.functional import one_hot


parser = argparse.ArgumentParser()
str2bool = lambda x: x.lower in ["true", "t", 1]
parser.add_argument("--nni", type=str2bool, default=False)
args = parser.parse_args()

if args.nni:
    import nni


class GraphNodeClassification(nn.Module, AbstractModel):
    def predict(self, *args, **kwargs):
        pass

    def evaluate(self, *args, **kwargs):
        pass

    def __init__(self, gc_kwargs: Dict = {}, nc_kwargs: Dict = {}):
        super(GraphNodeClassification, self).__init__()
        self.gc_model = ValuesAndGraphStructure(**gc_kwargs)
        self.nc_model = NodeClassification(**nc_kwargs)

    def __str__(self):
        return "Graph Node Classification"

    def forward(self, graphs_loader, inter_sample_edges):
        first_batch = True

        for graphs_data in graphs_loader:
            input_data, _ = self.gc_model.transform_input(graphs_data)
            output_ = self.gc_model.forward_one_before_last_layer(*input_data)

            if first_batch:
                attr_embeddings = output_
                gc_output_ = self.gc_model.forward_last_layer(output_)
                gc_output = gc_output_
                first_batch = False

            else:
                attr_embeddings = torch.cat((attr_embeddings, output_), 0)
                gc_output_ = self.gc_model.forward_last_layer(output_)
                gc_output = torch.cat((gc_output, gc_output_), 0)

        final_output = self.nc_model(attr_embeddings, inter_sample_edges.T)

        return final_output, gc_output

    @staticmethod
    def _extract_from_tab(tab_dataset: TabDataset, batch_size: int = 32):
        graphs_dataset = GraphsDataset.from_tab(tab_data=tab_dataset)
        graph_dataset = GraphDataset.from_tab(tab_data=tab_dataset)

        graphs_train_loader = graphs_dataset.get_train_data(
            as_loader=True, batch_size=batch_size
        )
        inter_samples_train_edges = graph_dataset.train.edges

        graphs_val_loader = graphs_dataset.get_val_data(
            as_loader=True, batch_size=batch_size
        )
        inter_samples_val_edges = graph_dataset.val.edges

        graphs_test_loader = graphs_dataset.get_test_data(
            as_loader=True, batch_size=batch_size
        )
        inter_samples_test_edges = graph_dataset.test.edges

        return (
            graphs_train_loader,
            inter_samples_train_edges,
            graphs_val_loader,
            inter_samples_val_edges,
            graphs_test_loader,
            inter_samples_test_edges,
        )

    def fit(
        self,
        tab_dataset: TabDataset,
        n_epochs: int = 100,
        batch_size: int = 32,
        lr_gc: float = 0.1,
        lr_final: float = 0.001,
        weight_decay_gc: float = 0,
        weight_decay_final: float = 0,
        optimizer: torch.optim = optim.Adam,
    ):
        (
            graphs_train_loader,
            inter_samples_train_edges,
            graphs_val_loader,
            inter_samples_val_edges,
            graphs_test_loader,
            inter_samples_test_edges,
        ) = self._extract_from_tab(tab_dataset, batch_size)

        self.gc_model.train()
        self.nc_model.train()

        train_final_losses, val_final_losses = [], []
        train_gc_losses, val_gc_losses = [], []
        train_final_aucs, val_final_aucs = [], []
        train_gc_aucs, val_gc_aucs = [], []

        params = list(self.gc_model.parameters()) + list(self.nc_model.parameters())

        gc_optimizer = optimizer(
            # self.gc_model.parameters(), lr=lr_gc, weight_decay=weight_decay_gc
            params,
            lr=lr_gc,
            weight_decay=weight_decay_gc,
        )

        final_optimizer = optimizer(
            # list(self.gc_model.parameters()) + list(self.nc_model.parameters()),
            params,
            lr=lr_final,
            weight_decay=weight_decay_final,
        )

        n_classes = graphs_train_loader.dataset.gdp.num_classes

        gc_optimizer.zero_grad()
        final_optimizer.zero_grad()

        for epoch in range(1, n_epochs + 1):
            gc_optimizer.zero_grad()
            final_optimizer.zero_grad()

            self.train()

            final_output, gc_output = self(
                graphs_train_loader, inter_samples_train_edges
            )

            train_labels = graphs_train_loader.dataset.gdp.Y

            # train_loss_gc = nn.CrossEntropyLoss()(gc_output, train_labels.float())
            train_loss_gc = nn.BCEWithLogitsLoss()(gc_output, train_labels.float())
            train_loss_gc.backward(retain_graph=True)

            # train_loss_final = nn.CrossEntropyLoss()(
            #     final_output[:, 0].view(-1, 1), train_labels.float()
            # )

            train_loss_final = nn.BCEWithLogitsLoss()(
                final_output[:, 0].view(-1, 1), train_labels.float()
            )
            train_loss_final.backward()

            gc_optimizer.step()
            final_optimizer.step()

            train_gc_losses.append(train_loss_gc.item())
            train_final_losses.append(train_loss_final.item())

            train_gc_auc = roc_auc_score(
                one_hot(train_labels.view(-1).long(), n_classes).detach().numpy(),
                torch.cat((gc_output, 1 - gc_output), dim=1).detach().numpy(),
            )
            train_final_auc = roc_auc_score(
                one_hot(train_labels.view(-1).long(), n_classes).detach().numpy(),
                final_output.detach().numpy(),
            )

            train_gc_aucs.append(train_gc_auc)
            train_final_aucs.append(train_final_auc)

            with torch.no_grad():
                self.gc_model.eval()
                self.nc_model.eval()

                final_output, gc_output = self(
                    graphs_val_loader, inter_samples_val_edges
                )

                val_labels = graphs_val_loader.dataset.gdp.Y

                # val_loss_gc = nn.CrossEntropyLoss()(gc_output, val_labels.float())
                val_loss_gc = nn.BCEWithLogitsLoss()(gc_output, val_labels.float())

                # val_loss_final = nn.CrossEntropyLoss()(
                #     final_output[:, 0].view(-1, 1), val_labels.float()
                # )

                val_loss_final = nn.BCEWithLogitsLoss()(
                    final_output[:, 0].view(-1, 1), val_labels.float()
                )

                val_gc_losses.append(val_loss_gc)
                val_final_losses.append(val_loss_final)

                val_gc_auc = roc_auc_score(
                    one_hot(val_labels.view(-1).long(), n_classes).detach().numpy(),
                    torch.cat((gc_output, 1 - gc_output), dim=1).detach().numpy(),
                )

                val_final_auc = roc_auc_score(
                    one_hot(val_labels.view(-1).long(), n_classes).detach().numpy(),
                    final_output.detach().numpy(),
                )

                val_gc_aucs.append(val_gc_auc)
                val_final_aucs.append(val_final_auc)

            print(
                f"Epoch {epoch}:\n"
                # f"\tTrain GC loss:\t{train_loss_gc}\n",
                # f"\tVal GC loss:\t{val_loss_gc}\n",
                # f"\tTrain final loss:\t{train_loss_final}\n",
                # f"\tVal final loss:\t{val_loss_final}\n",
                # f"\tTrain GC AUC:\t{train_gc_auc}\n",
                # f"\tVal GC AUC:\t{val_gc_auc}\n",
                f"\tTrain final AUC:\t{train_final_auc}\n",
                f"\tVal final AUC:\t{val_final_auc}\n",
            )

        return val_final_auc


if __name__ == "__main__":
    from datasets.graphsDataset import GraphsDataset
    from datasets.graphDataset import GraphDataset

    data_dir = "../data/Banknote/processed/90"
    tab_dataset = TabDataset.load(data_dir=data_dir)

    graphs_dataset = GraphsDataset.from_tab(tab_data=tab_dataset)
    graph_dataset = GraphDataset.from_tab(tab_data=tab_dataset)

    graphs_train_loader = graphs_dataset.get_train_data(as_loader=True, batch_size=32)
    inter_samples_train_edges = graph_dataset.train.edges

    if args.nni:
        params = nni.get_next_parameter()

        model = GraphNodeClassification(
            gc_kwargs={
                "input_example": graphs_dataset,
                "RECEIVED_PARAMS": {
                    "preweight": params["gc_preweight"],
                    "layer_1": params["gc_layer_1"],
                    "layer_2": params["gc_layer_2"],
                    "activation": params["gc_activation"],
                    "dropout": params["gc_dropout"],
                },
            },
            nc_kwargs={
                "h_layers": [params["nc_layer_1"], params["nc_layer_2"]],
                "dropouts": [params["nc_dropout_1"], params["nc_dropout_2"]],
                "activations": [params["nc_activation_1"], params["nc_activation_2"]],
            },
        )

        # model(graphs_train_loader, inter_samples_train_edges)
        val_final_auc = model.fit(
            tab_dataset=tab_dataset,
            n_epochs=300,
            lr_final=params["nc_lr"],
            lr_gc=params["gc_lr"],
        )
        nni.report_final_result(val_final_auc)

    else:
        model = GraphNodeClassification(
            gc_kwargs={
                "input_example": graphs_dataset,
                "RECEIVED_PARAMS": {
                    "preweight": 10,
                    "layer_1": 7,
                    "layer_2": 7,
                    "activation": "elu",
                    "dropout": 0.2,
                },
            }
        )

        # model(graphs_train_loader, inter_samples_train_edges)
        model.fit(tab_dataset=tab_dataset, n_epochs=1000, lr_final=0.1, lr_gc=0.001)
