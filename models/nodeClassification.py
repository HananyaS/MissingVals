import torch
from torch import nn
from torch_geometric.nn import GCNConv, Sequential
from .abstractNN import AbstractNN
from typing import List, Union, Type
from torch.nn.functional import one_hot


class NodeClassification(AbstractNN):
    _activation_dict = {
        "relu": nn.ReLU(),
        "elu": nn.ELU(),
        "tanh": nn.Tanh(),
    }

    def __init__(
        self,
        device: torch.device = torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu"),
        n_features: int = -1,
        n_classes: int = 2,
        h_layers: List[int] = [10, 5],
        dropouts: List[float] = None,
        activations: List[str] = ["elu", "elu"],
    ):
        assert dropouts is None or len(dropouts) == len(h_layers)
        assert activations is None or len(activations) == len(h_layers)
        assert set(activations) <= set(self._activation_dict.keys())
        assert dropouts is None or all([0 <= d < 1 for d in dropouts])

        super(NodeClassification, self).__init__(device)
        self.n_classes = n_classes
        self.h_layers = h_layers
        self.dropouts = dropouts
        self.activations = activations
        self.n_features = n_features

        self.one_before_last_layer, self.classifier = self._get_layers(
            n_features=n_features,
            n_classes=n_classes,
            h_layers=h_layers,
            dropouts=dropouts,
            activations=activations,
        )

    def _get_layers(self, n_features, n_classes, h_layers, dropouts, activations):
        start_layers = []
        end_layers = []

        for i in range(len(h_layers)):
            if i == 0:
                gcn_layer = GCNConv(n_features, h_layers[i])
            else:
                gcn_layer = GCNConv(h_layers[i - 1], h_layers[i])

            start_layers.append((gcn_layer, "x, edge_index -> x"))
            # start_layers.append("x, edge_index -> x")

            if dropouts is not None:
                if i < len(dropouts) - 1:
                    start_layers.append(nn.Dropout(dropouts[i], inplace=True))
                else:
                    end_layers.append(nn.Dropout(dropouts[i], inplace=True))

            if activations:
                if i < len(activations) - 1:
                    start_layers.append(
                        (self._activation_dict[activations[i]], "x -> x")
                    )
                    # start_layers.append("x -> x, edge_index")

                else:
                    end_layers.append((self._activation_dict[activations[i]], "x -> x"))
                    # end_layers.append("x -> x, edge_index")

        end_layers.append((nn.Linear(h_layers[-1], n_classes), "x -> x"))

        start_layers = Sequential("x, edge_index", start_layers)
        end_layers = Sequential("x", end_layers)

        return start_layers, end_layers

    def _forward_one_before_last_layer(self, *args, **kwargs):
        return self.one_before_last_layer(*args, **kwargs)

    def _forward_last_layer(self, *args, **kwargs):
        return self.classifier(*args, **kwargs)

    def _transform_input(self, data: Union[Type[torch.Tensor], List]):
        *input_, labels = [data[i][0] for i in range(len(data))]

        if len(input_[0].shape) == 3 and input_[0].shape[-1] == 1:
            input_[0] = torch.squeeze(input_[0], -1)

        input_[1] = input_[1].long()

        if input_[1].shape[0] != 2:
            if input_[1].shape[1] != 2:
                raise Exception("Format doesn't match")
            else:
                input_[1] = input_[1].T

        return input_, labels

    def _transform_output(self, output):
        return output

    def get_num_classes(self):
        return self.n_classes

    def _eval_loss(
        self,
        output: torch.Tensor,
        labels: torch.Tensor,
        loss_func: torch.nn.modules.loss,
        n_classes: int = 2,
    ) -> torch.nn.modules.loss:
        labels = one_hot(labels.long(), num_classes=n_classes).float()
        # output = torch.cat([output, 1 - output], dim=1)
        loss = loss_func(output, labels.squeeze(1))
        return loss

    def __str__(self):
        return "Node Classification model"
