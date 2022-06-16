from pipeline.pipline import Pipeline
from pipeline.stages.dataStage import DataStage
from pipeline.stages.mlModelStage import MLModelStage
from models.xgb import XGBoost

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

xgbBuilder = MLModelStage(
    name="XGBoostBuilder",
    task="build",
    run_kwargs={"model_type": XGBoost, "n_estimators": 100},
    input_from={},
    store_in="xgb_model",
)

xgbFit = MLModelStage(
    name="XGBoostFit",
    task="fit",
    model_type=XGBoost,
    input_from={"model": "xgb_model", "train_loader": "train_loader"},
)

xgbEval = MLModelStage(
    name="XGBoostEval",
    task="evaluate",
    model_type=XGBoost,
    input_from={"model": "xgb_model", "val_loader": "test_loader"},
    store_in="test_results",
)

pipe = Pipeline(
    name="Test",
    verbose=True,
    stages=[loadData, normData, getTrain, getTest, xgbBuilder, xgbFit, xgbEval],
)

print(pipe)

data_dir = "../../data/Banknote/processed/90"
model = pipe(data_dir)
print(model)
