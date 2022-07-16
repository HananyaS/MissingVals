from datasets.graphDataPair import GraphDataPair
from datasets.tabDataset import TabDataset

from copy import deepcopy


class GraphDataset(TabDataset):
    def __init__(self, **kwargs):
        # def __init__(self, train_graph: GraphDataPair, val_graph: GraphDataPair, test_graph: GraphDataPair):
        #     assert (
        #         train_graph.get_num_features()
        #         == val_graph.get_num_features()
        #         == test_graph.get_num_features()
        #     ), "Train, val_graph and test_graph sets must have the same number of features."
        #     assert (
        #         train_graph.get_num_classes() == val_graph.get_num_classes() == test_graph.get_num_classes()
        #     ), "Train, val_graph and test_graph sets must have the same number of classes."
        super(GraphDataset, self).__init__(**kwargs)

        train_mask = ["train"] * self.train.get_num_samples()
        val_mask = ["val"] * self.val.get_num_samples()
        test_mask = ["test"] * self.test.get_num_samples()

        self.mask = train_mask + val_mask + test_mask
        self.train_idx = list(
            filter(lambda i: self.mask[i] == "train", list(range(len(self.mask))))
        )
        #
        self.data = deepcopy(self.train)
        self.data += self.val
        self.data += self.test

    @classmethod
    def from_tab(cls, tab_data: TabDataset, **kwargs):
        train = GraphDataPair.from_tab(tab_data=tab_data.train, **kwargs)

        val = (
            None
            if not tab_data.val_exists
            else GraphDataPair.from_tab(tab_data=tab_data.val, **kwargs)
        )

        test = (
            None
            if not tab_data.test_exists
            else GraphDataPair.from_tab(tab_data=tab_data.test, **kwargs)
        )

        graph_dataset = cls(
            train=train,
            val=val,
            test=test,
            normalize=False,
            name=f"{tab_data.name} - graph",
        )

        graph_dataset.normalized = tab_data.normalized

        return graph_dataset

    def get_graph_data(self):
        self.data.edges_to_lst(inplace=True)
        return self.data

    def get_train_loader(self, **kwargs):
        return self.train.to_loader(**kwargs)

    def get_val_loader(self, **kwargs):
        return self.val.to_loader(**kwargs)

    def get_test_loader(self, **kwargs):
        return self.test.to_loader(**kwargs)


if __name__ == "__main__":
    from time import time
    from models.nodeClassification import NodeClassification

    st = time()

    data_dir = "../data/Banknote/processed/90"

    td = TabDataset.load(
        data_dir=data_dir,
        normalize=True,
        shuffle=True,
        add_existence_cols=False,
    )

    graph_dataset = GraphDataset.from_tab(
        tab_data=td,
        knn_kwargs={"distance": "euclidian"},
    )

    train_graph = graph_dataset.get_train_loader()
    val_graph = graph_dataset.get_val_loader()
    test_graph = graph_dataset.get_test_loader()

    model = NodeClassification(n_features=train_graph.dataset.gdp.get_num_features())

    model.fit(
        train_loader=train_graph,
        val_loader=val_graph,
        auc_plot_path="nc_auc.png",
        loss_plot_path="nc_loss.png",
        save_results=True,
        plot_results=True,
        show_results=True,
        lr=0.01,
        n_epochs=50,
        verbose=True,
        save_model=False
    )

    test_auc = model.evaluate(loader=test_graph)
    print(f"Test AUC: {test_auc:.4f}")

    print(f"Time elapsed: {time() - st} seconds")
