import sys

sys.path.append('../../')

import json
import os.path
from argparse import Namespace

from pipeline.pipline import Pipeline

from pipeline.stages.tabDataStage import TabDataStage
from pipeline.stages.graphDataStage import GraphDataStage
from pipeline.stages.graphsDataStage import GraphsDataStage
from pipeline.stages.nnModelStage import NNModelStage

from models.graphClassification import ValuesAndGraphStructure as VAGS
from models.nodeClassification import NodeClassification

from typing import Union, Type, Dict


class Method5PL(Pipeline):
    default_params = {
        "ec": True,
        "batch_size": 32,
        "distance": "heur_dist",
        "alpha": 1,
        "beta": 1,
        "gc_epochs": 40,
        "gc_lr": 0.001,
        "gc_weight_decay": 0,
        "nc_epochs": 40,
        "nc_lr": 0.001,
        "nc_weight_decay": 0,
    }

    def __init__(self, params: Union[Type[str], Dict] = {}, verbose: bool = True):
        self.args = self.transform_args(params)

        loadData = TabDataStage(
            name="Load Data",
            task="load",
            run_kwargs={"normalize": False, "add_existence_cols": self.args.ec},
            store_in="tab_dataset",
        )

        normData = TabDataStage(
            name="Normalize Data",
            task="norm",
            store_in="tab_dataset",
            input_from={"data": "tab_dataset"},
            force_store=True,
        )

        tab2Graphs = GraphsDataStage(
            name="Tab2Graph",
            task="from_tab",
            input_from={"tab_data": "tab_dataset"},
            store_in="graphs_dataset",
            force_store=True,
            run_kwargs={
                "knn_kwargs": {
                    "distance": self.args.distance,
                    "dist_params": self.args.dist_params,
                }
            },
        )

        getTrain = GraphsDataStage(
            name="Get Train",
            task="get_train",
            run_kwargs={"as_loader": True, "batch_size": self.args.batch_size},
            input_from={"data": "graphs_dataset"},
            store_in="train_loader",
        )
        getTest = GraphsDataStage(
            name="Get Test",
            task="get_test",
            run_kwargs={"as_loader": True, "batch_size": self.args.batch_size},
            input_from={"data": "graphs_dataset"},
            store_in="test_loader",
        )

        getVal = GraphsDataStage(
            name="Get Val",
            task="get_val",
            run_kwargs={"as_loader": True, "batch_size": self.args.batch_size},
            input_from={"data": "graphs_dataset"},
            store_in="val_loader",
        )

        buildGCModel = NNModelStage(
            name="Build Graph Classification Model",
            task="build",
            input_from={"input_example": "train_loader"},
            run_kwargs={"model_type": VAGS},
            store_in="GCModel",
        )

        fitGC = NNModelStage(
            name="Fit Graph Classification Model",
            task="fit",
            run_kwargs={
                "n_epochs": self.args.gc_epochs,
                "lr": self.args.gc_lr,
                "weight_decay": self.args.gc_weight_decay,
                "plot_results": True,
                "show_results": True,
                "verbose": verbose,
            },
            input_from={
                "train_loader": "train_loader",
                "val_loader": "val_loader",
                "model": "GCModel",
            },
            store_in="GCModel",
            force_store=True,
        )

        extractTrainEmbeddings = NNModelStage(
            name="Extract Train Embeddings",
            task="extract_one_before_last_layer",
            input_from={"model": "GCModel", "loader": "train_loader"},
            store_in="train_embeddings",
        )

        extractValEmbeddings = NNModelStage(
            name="Extract Val Embeddings",
            task="extract_one_before_last_layer",
            input_from={"model": "GCModel", "loader": "val_loader"},
            store_in="val_embeddings",
        )

        extractTestEmbeddings = NNModelStage(
            name="Extract Test Embeddings",
            task="extract_one_before_last_layer",
            input_from={"model": "GCModel", "loader": "test_loader"},
            store_in="test_embeddings",
        )

        setXTrain = TabDataStage(
            name="Set Train Embeddings",
            task="setX",
            run_kwargs={"set_": "train"},
            input_from={"data": "tab_dataset", "X": "train_embeddings"},
            store_in="tab_dataset",
            force_store=True,
        )

        setXVal = TabDataStage(
            name="Set Val Embeddings",
            task="setX",
            run_kwargs={"set_": "val"},
            input_from={"data": "tab_dataset", "X": "val_embeddings"},
            store_in="tab_dataset",
            force_store=True,
        )

        setXTest = TabDataStage(
            name="Set Test Embeddings",
            task="setX",
            run_kwargs={"set_": "test"},
            input_from={"data": "tab_dataset", "X": "test_embeddings"},
            store_in="tab_dataset",
            force_store=True,
        )

        tab2Graph = GraphDataStage(
            name="Generate Graph",
            task="from_tab",
            input_from={"tab_data": "tab_dataset"},
            run_kwargs={
                "knn_kwargs": {
                    "distance": self.args.distance,
                    "dist_params": self.args.dist_params,
                }
            },
            store_in="graph_dataset",
            force_store=True,
        )

        getTrainLoader = GraphDataStage(
            name="Get Train Loader",
            task="get_train_loader",
            input_from={"data": "graph_dataset"},
            store_in="train_loader",
            force_store=True,
        )

        getTestLoader = GraphDataStage(
            name="Get Test Loader",
            task="get_test_loader",
            input_from={"data": "graph_dataset"},
            store_in="test_loader",
            force_store=True,
        )

        getValLoader = GraphDataStage(
            name="Get Val Loader",
            task="get_val_loader",
            input_from={"data": "graph_dataset"},
            store_in="val_loader",
            force_store=True,
        )

        buildNCModel = NNModelStage(
            name="Build Graph Classification Model",
            task="build",
            run_kwargs={"model_type": NodeClassification},
            store_in="NCModel",
        )

        fitNC = NNModelStage(
            name="Fit Node Classification Model",
            task="fit",
            run_kwargs={
                "n_epochs": self.args.nc_epochs,
                "lr": self.args.nc_lr,
                "weight_decay": self.args.nc_weight_decay,
                "plot_results": True,
                "show_results": True,
                "save_model": False,
                "verbose": verbose,
            },
            input_from={
                "train_loader": "train_loader",
                "val_loader": "val_loader",
                "model": "NCModel",
            },
            store_in="NCModel",
            force_store=True,
        )

        evalNCTrain = NNModelStage(
            name="Evaluate Model - Train",
            task="evaluate",
            input_from={"model": "NCModel", "loader": "train_loader"},
            store_in="train_results",
        )

        evalNCVal = NNModelStage(
            name="Evaluate Model - Val",
            task="evaluate",
            input_from={"model": "NCModel", "loader": "val_loader"},
            store_in="val_results",
        )

        evalNCTest = NNModelStage(
            name="Evaluate Model - Test",
            task="evaluate",
            input_from={"model": "NCModel", "loader": "test_loader"},
            store_in="test_results",
        )

        stage_list = [
            loadData,
            normData,
            tab2Graphs,
            getTrain,
            getTest,
            getVal,
            buildGCModel,
            fitGC,
            extractTrainEmbeddings,
            extractTestEmbeddings,
            extractValEmbeddings,
            setXTrain,
            setXVal,
            setXTest,
            tab2Graph,
            getTrainLoader,
            getTestLoader,
            getValLoader,
            buildNCModel,
            fitNC,
            evalNCTrain,
            evalNCVal,
            evalNCTest,
        ]

        super(Method5PL, self).__init__(
            stages=stage_list, name="Method 5", verbose=verbose
        )

    def transform_args(self, params: Union[Type[str], Dict]):
        if isinstance(params, str):
            assert params.endswith(".json"), "params file must be of type json."
            assert os.path.isfile(params), "params file doesn't exist."

            with open(params, "rb") as f:
                params = json.load(f)

        args = Namespace()

        for k, v in self.default_params.items():
            setattr(args, k, params.get(k, v))

        if args.distance == "euclidian":
            args.alpha = None
            args.beta = None
            args.dist_params = {}

        else:
            args.dist_params = {"alpha": args.alpha, "beta": args.beta}

        return args


if __name__ == "__main__":
    data_dir = "../../data/Banknote/processed/90"

    params = {"distance": "euclidian", "gc_epochs": 20, "nc_epochs": 50}

    pipe = Method5PL(params=params, verbose=True)
    print(pipe)
    auc = pipe(data_dir)
    print("AUC Results:")
    print(f"Train:\t{pipe.cache['train_results']:.4f}")
    print(f"Val:\t{pipe.cache['val_results']:.4f}")
    print(f"Test:\t{pipe.cache['test_results']:.4f}")
