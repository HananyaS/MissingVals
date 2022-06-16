from pipeline.pipline import Pipeline
from pipeline.stages.dataStage import DataStage
from pipeline.stages.mlModelStage import MLModelStage
from models.graphClassification import ValuesAndGraphStructure as VGS
import torch

loadData = DataStage(name="Load Data", task="load", run_kwargs={"normalize": False})
normData = DataStage(name="Normalize Data", task="norm", store_in="dataset")

getTrain = DataStage(
    name="Get Train",
    task="get_train",
    run_kwargs={"as_loader": True, "batch_size": 32},
    input_from={"data": "dataset"},
    store_in="train_loader",
)

getTest = DataStage(
    name="Get Test",
    task="get_test",
    run_kwargs={"as_loader": True, "batch_size": 32},
    input_from={"data": "dataset"},
    store_in="test_loader",
)

vgsBuilder = MLModelStage(
    name="XGBoostBuilder",
    task="build",
    run_kwargs={
        "model_type": VGS,
        "device": torch.device("cpu"),
        "RECEIVED_PARAMS": {
            "preweight": 4,
            "layer_1": 12,
            "layer_2": 3,
            "activation": "relu",
            "dropout": 0.2,
        }
    },
    input_from={"input_example": "train_loader"},
    store_in="model",
)

vgsFit = MLModelStage(
    name="XGBoostFit",
    task="fit",
    input_from={
        "model": "model",
        "train_loader": "train_loader",
        "val_loader": "test_loader",
    },
    run_kwargs={"n_epochs": 10, "lr": 0.001, "save_results": False},
)

vgsEval = MLModelStage(
    name="XGBoostEval",
    task="evaluate",
    input_from={"model": "model", "val_loader": "test_loader"},
    store_in="test_results",
)

pipe = Pipeline(
    name="Test",
    verbose=True,
    stages=[loadData, normData, getTrain, getTest, vgsBuilder, vgsFit, vgsEval],
)

print(pipe)

data_dir = "../../data/Banknote/processed/90"
model = pipe(data_dir)
print(model)
