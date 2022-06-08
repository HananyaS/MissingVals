import torch
import torch.nn as nn
from torch.nn.functional import one_hot
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score


class ValuesAndGraphStructure(nn.Module):
    def __init__(
        self, nodes_number, feature_size, RECEIVED_PARAMS, device, num_classes=1
    ):
        super(ValuesAndGraphStructure, self).__init__()
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

    def forward(self, x, adjacency_matrix):
        # multiply the matrix adjacency_matrix by (learnt scalar) self.alpha
        a, b, c = adjacency_matrix.shape

        if self.feature_size > 1:
            x = self.transform_mat_to_vec(x)
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
        x = self.fc3(x)

        x = nn.Sigmoid()(x)
        return x

    def extract_one_before_last_layer(self, x, adjacency_matrix):
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

    def fit(
        self,
        train_loader,
        val_loader,
        optimizer,
        epochs=100,
        verbose=True,
        plot_loss=True,
        plot_accuracy=True,
        val_labels=None,
    ):
        assert (
            val_labels is not None or not val_loader.dataset.test
        ), "Validation labels are not provided"

        if val_labels is not None and not isinstance(val_labels, torch.Tensor):
            val_labels = torch.tensor(val_labels).to(self.device)

        if len(val_labels.shape) == 1 or val_labels.shape[1] == 1:
            val_labels = one_hot(val_labels.long(), num_classes=2).float()

        if verbose:
            print("Training started")

        self.train()
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []

        for epoch in range(epochs):
            total_train_loss = 0
            for data in train_loader:
                xs, adjs, labels = data

                xs = xs.to(self.device)
                adjs = adjs.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                output = self(xs, adjs)
                labels = one_hot(labels.long(), num_classes=2).float()
                output = torch.cat([output, 1 - output], dim=1)
                loss = nn.BCELoss(reduction="mean")(output, labels)

                loss.backward()
                optimizer.step()
                total_train_loss += loss.item() * xs.size(0)

            total_train_loss = total_train_loss / len(train_loader.dataset)

            # calculate loss on validation set
            with torch.no_grad():
                self.eval()
                pred_val = self.predict(val_loader, probs=True)
                val_loss = nn.BCELoss(reduction="mean")(pred_val, val_labels).item()

            train_auc = self.evaluate(train_loader)
            val_auc = self.evaluate(val_loader, val_labels)

            if verbose:
                print(f"Epoch: {epoch + 1}/{epochs}")
                print(f"\tTrain loss: {total_train_loss:.4f}")
                print(f"\tVal loss: {val_loss:.4f}")
                print(f"\tTrain AUC: {train_auc:.4f}")
                print(f"\tVal AUC: {val_auc:.4f}")

            if plot_loss:
                train_losses.append(total_train_loss)
                val_losses.append(val_loss)

            if plot_accuracy:
                train_accs.append(train_auc)
                val_accs.append(val_auc)

        if plot_loss:
            plt.clf()
            plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train loss")
            plt.plot(range(1, len(val_losses) + 1), val_losses, label="Val loss")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title("Loss")
            plt.legend()
            plt.savefig("loss.png")

            if verbose:
                plt.show()
            else:
                plt.clf()

        if plot_accuracy:
            plt.clf()
            plt.plot(range(1, len(train_accs) + 1), train_accs, label="Training AUC")
            plt.plot(range(1, len(val_accs) + 1), val_accs, label="Validation AUC")
            plt.xlabel("Epochs")
            plt.ylabel("AUC")
            plt.title("Training and Validation AUC")
            plt.legend()
            plt.savefig("auc.png")

            if verbose:
                plt.show()
            else:
                plt.clf()

    def predict(self, loader, probs=False):
        self.eval()
        pred = torch.Tensor().to(self.device)

        with torch.no_grad():
            for data in loader:
                x, adj = data[:2]
                x = x.to(self.device)
                adj = adj.to(self.device)

                output = self(x, adj)
                output = torch.cat([output, 1 - output], dim=1)
                if not probs:
                    output = torch.argmax(output, dim=1)
                pred = torch.cat((pred, output), dim=0)

        return pred

    def evaluate(self, loader, labels=None, metric="AUC"):
        assert (
            not loader.dataset.test or not labels is None
        ), "If you are testing, you must provide labels"
        self.eval()

        if labels is None:
            labels = loader.dataset.labels
            labels = one_hot(labels.long(), num_classes=2).float()

        elif not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels).to(self.device)

        if metric.upper() == "AUC":
            pred = self.predict(loader, probs=True)
            auc = roc_auc_score(labels.cpu().numpy(), pred.cpu().numpy())
            return auc

        else:
            pred = self.predict(loader, probs=False)
            return (pred == labels).sum().item() / len(pred)
