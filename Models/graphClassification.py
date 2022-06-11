import torch
import torch.nn as nn
from torch.nn.functional import one_hot

from abstractNN import AbstractNN


class ValuesAndGraphStructure(AbstractNN):
    def __init__(
        self, nodes_number, feature_size, RECEIVED_PARAMS, device, num_classes=1
    ):
        super(ValuesAndGraphStructure, self).__init__(device=device)
        self.feature_size = feature_size  # dimension of features of nodes
        self.nodes_number = nodes_number
        self.device = device
        self.RECEIVED_PARAMS = RECEIVED_PARAMS

        # self.pre_weighting = nn.Linear(self.feature_size, int(self.RECEIVED_PARAMS["preweight"]))
        self.pre_weighting = nn.Linear(1, int(self.RECEIVED_PARAMS["preweight"]))
        self.fc1 = nn.Linear(
            int(self.RECEIVED_PARAMS["preweight"]) * self.nodes_number,
            int(self.RECEIVED_PARAMS["layer_1"]),
        )  # input layer
        self.fc2 = nn.Linear(
            int(self.RECEIVED_PARAMS["layer_1"]), int(self.RECEIVED_PARAMS["layer_2"])
        )
        self.fc3 = nn.Linear(int(self.RECEIVED_PARAMS["layer_2"]), num_classes)
        self.activation_func = self.RECEIVED_PARAMS["activation"]
        self.dropout = nn.Dropout(p=self.RECEIVED_PARAMS["dropout"])

        self.alpha = nn.Parameter(torch.rand(1, requires_grad=True, device=self.device))

        self.activation_func_dict = {
            "relu": nn.ReLU(),
            "elu": nn.ELU(),
            "tanh": nn.Tanh(),
        }
        if self.feature_size > 1:
            self.transform_mat_to_vec = nn.Linear(self.feature_size, 1)

        self.gcn_layer = nn.Sequential(
            self.pre_weighting, self.activation_func_dict[self.activation_func]
        )

        self.classifier = nn.Sequential(
            self.fc1,
            self.activation_func_dict[self.activation_func],
            self.dropout,
            self.fc2,
            self.activation_func_dict[self.activation_func],
        )

    def _forward_one_before_last_layer(self, x, adjacency_matrix):
        # multiply the matrix adjacency_matrix by (learnt scalar) self.alpha
        a, b, c = adjacency_matrix.shape
        I = torch.eye(b).to(self.device)
        alpha_I = I * self.alpha.expand_as(I)  # 𝛼I
        normalized_adjacency_matrix = self.calculate_adjacency_matrix(
            adjacency_matrix
        )  # Ã
        alpha_I_plus_A = alpha_I + normalized_adjacency_matrix  # 𝛼I + Ã
        x = torch.einsum("ijk, ik->ij", alpha_I_plus_A, x).view(
            *x.shape, 1
        )  # (𝛼I + Ã)·x

        x = self.gcn_layer(x)
        x = torch.flatten(x, start_dim=1)  # flatten the tensor
        x = self.classifier(x)
        return x

    def forward_one_before_last_layer(self, *args, **kwargs):
        return self._forward_one_before_last_layer(*args, **kwargs)

    def _forward_last_layer(self, x):
        x = self.fc3(x)
        x = nn.Sigmoid()(x)
        return x

    def forward_last_layer(self, *args, **kwargs):
        return self._forward_last_layer(*args, **kwargs)

    def calculate_adjacency_matrix(self, batched_adjacency_matrix):
        # D^(-0.5)
        def calc_d_minus_root_sqr(batched_adjacency_matrix):
            r = []
            for adjacency_matrix in batched_adjacency_matrix:
                sum_of_each_row = adjacency_matrix.sum(1)
                sum_of_each_row_plus_one = torch.where(
                    sum_of_each_row != 0,
                    sum_of_each_row,
                    torch.tensor(1.0, device=self.device),
                )
                r.append(torch.diag(torch.pow(sum_of_each_row_plus_one, -0.5)))
            s = torch.stack(r)
            if torch.isnan(s).any():
                print("Alpha when stuck", self.alpha.item())
                print(
                    "batched_adjacency_matrix",
                    torch.isnan(batched_adjacency_matrix).any(),
                )
                print("The model is stuck", torch.isnan(s).any())
            return s

        D__minus_sqrt = calc_d_minus_root_sqr(batched_adjacency_matrix)
        normalized_adjacency = torch.matmul(
            torch.matmul(D__minus_sqrt, batched_adjacency_matrix), D__minus_sqrt
        )
        return normalized_adjacency

    def _transform_input(self, data: torch.Tensor):
        xs, adjs, labels = data
        return [xs, adjs], labels

    def _eval_loss(
        self,
        output: torch.Tensor,
        labels: torch.Tensor,
        loss_func: torch.nn.modules.loss,
        n_classes: int = 2,
    ) -> torch.nn.modules.loss:
        labels = one_hot(labels.long(), num_classes=n_classes).float()
        output = torch.cat([output, 1 - output], dim=1)
        loss = loss_func(output, labels)
        return loss

