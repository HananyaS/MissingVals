import torch
import matplotlib.pyplot as plt

# Global Variables
TNSR = type(torch.tensor([]))
BOOL = type(False)
MSG = "\nEarly Stoped"
INF = float("inf")
GPU = "cuda:1"
CPU = "cpu"
TWO = 2
LEN = 2

# useful functions
torchify = lambda x, device: x if type(x) == TNSR else torch.from_numpy(x).to(device)
rmse = lambda mat1, mat2: torch.sqrt(torch.mean(torch.pow(mat1 - mat2, 2))).float()
gap = lambda ary: INF if len(ary) < LEN else ary[-2] - ary[-1]


# Graph Feature Propagation Object
class GFP:
    # This Object using massege passing mechanism for fill in
    # missing feature in graph dataset.

    # based on:
    # "On the Unreasonable Effectiveness of Feature propagation
    #  in Learning on Graphs with Missing Node Features"
    #  https://arxiv.org/abs/2111.12128

    # The Object contain also some improvment that the original article not inculded,
    # as early stop mechanism, delta measuring & plot function

    # The all implementation build using pytorch

    def __init__(
        self, index=False, iters=50, eps=1e-3, delta_func=rmse, early_stop=True
    ):
        """
        Initilaize the GFP Object
        :index: index for missing values (default: False)
        :iters: number of iteration (default: 50)
        :eps: epsilon for early stop (default: 1e-3)
        :delta f: function for delta measuring (default: rmse)
        :early_stop: enable early stop (default: True)
        """

        # define the device (cuda/cpu)
        self.device = GPU if torch.cuda.is_available() else CPU

        # index for missing values
        self.index = (
            torchify(index, self.device).bool() if type(index) != BOOL else index
        )

        # enable early stop
        self.early = early_stop

        # number of propagation iterations
        self.iters = iters

        # the A hat matrix for message passing
        self.A_hat = None

        # Delta between known value to propagate value
        self.delta = []

        # mark True if early stop activated
        self.stop = False

        # epsilon value for early stop
        self.eps = eps

        # function for delta calculation
        self.f = delta_func

    def build_a_hat(self, nodes_num, edges):
        """
        building the A hat Matrix
        :nodes_num: number of the nodes in tha graph
        :edges: edge index matrix
        """
        # make sure the we using the right dimensions
        edges = edges.to(CPU) if edges.size()[0] == TWO else edges.T.to(CPU)

        # add in-links to edge index matrix (links between node to himself)
        # eye = torch.arange(nodes_num, device=self.device)  ## CHANGED
        eye = torch.arange(nodes_num)  ## CHANGED
        row = torch.cat((edges[0, :], eye))
        col = torch.cat((edges[1, :], eye))

        edges = torch.stack((row, col)).to(self.device)

        # calculate the weight for each cell in the matrix
        _, D = torch.unique(edges[0, :], return_counts=True)
        D_inv = torch.pow(D, -0.5)
        weight = D_inv[row] * D_inv[col]

        # build A hat, as sparse matrix
        self.A_hat = torch.sparse.FloatTensor(edges, weight).to(self.device)

        # free memory
        del weight, D, D_inv, row, col, edges

        return None

    def define_index(self, X):
        """
        define index (if the object didnt get one)
        in this case, each 0 value in X defined as missing values
        :X: feature matrix
        """
        if type(self.index) != BOOL:
            return None
        self.index = (
            (X == 0).bool() if type(X) == TNSR else torch.from_numpy(X == 0).bool()
        )
        return None

    def prop(self, X, edges):
        """
        feature propagation
        :X: feature matrix
        :edges: edges index
        :return: X hat - with propagate features
        """
        # convert feature & edges to torch tensors
        X = torchify(X, self.device).float()
        edges = torchify(edges, self.device)

        # building index & A hat
        self.define_index(X)
        self.build_a_hat(X.size()[0], edges)

        # feature propagate
        old_mat = X
        for i in range(self.iters):

            # updating X
            AX = torch.sparse.mm(self.A_hat, X)
            X = torch.where(self.index, AX, old_mat)

            # measure delta between update AX to the real features
            self.delta.append(
                self.f(AX[self.index == False], old_mat[self.index == False])
            )

            # early stop mechanism
            if self.early and gap(self.delta) < self.eps:
                self.stop = True
                return X.detach().cpu().numpy()

        return X.detach().cpu().numpy()

    def plot(self, size=10):
        """
        plot the delta between Ax to actual values
        :size: plot size
        """
        axe = torch.arange(len(self.delta))
        msg = MSG if self.stop else ""

        plt.figure(figsize=(size, size * 0.6))
        plt.title(f"Delta Between Predicion to Actual Features:{msg}", fontsize=18)

        # CHANGED
        del_to_lst = [d.item() for d in self.delta]  ## CHANGED

        plt.plot(axe, del_to_lst, color="r", label="Delta")
        plt.xlabel("Iterations", fontsize=14)
        plt.ylabel("Delta", fontsize=14)
        plt.ylim(0, del_to_lst[0] * 1.01)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(fontsize=14)
        plt.show()


# © Etzion Harari
# https://github.com/EtzionR
# MIT  License
