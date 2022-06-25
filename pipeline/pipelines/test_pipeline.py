from pipeline.pipline import Pipeline
from pipeline.stages.dataStage import DataStage
from pipeline.stages.mlModelStage import MLModelStage
from pipeline.stages.gfpStage import GFPStage
from models.xgb import XGBoost
from models.graphClassification import ValuesAndGraphStructure as VGS

loadData = DataStage(
    name="Load Data",
    task="load",
    run_kwargs={"normalize": False, "add_existence_cols": True},
)
normData = DataStage(name="Normalize Data", task="norm", store_in="dataset")

gfpBuild = GFPStage(name="GFPBuilder", task="build", store_in="gfp")

gfpProp = GFPStage(
    name="GFPProp",
    task="prop",
    input_from={"gfp_obj": "gfp", "data": "dataset"},
    run_kwargs={"distance": "heur_dist"},
)

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
    input_from={"model": "xgb_model", "train_loader": "train_loader"},
)

xgbEval = MLModelStage(
    name="XGBoostEval",
    task="evaluate",
    input_from={"model": "xgb_model", "val_loader": "test_loader"},
    store_in="test_results",
)


pipe = Pipeline(
    name="Test",
    verbose=True,
    stages=[
        loadData,
        normData,
        gfpBuild,
        gfpProp,
        getTrain,
        getTest,
        xgbBuilder,
        xgbFit,
        xgbEval,
    ],
)

print(pipe)
data_dir = "../../data/RoysData/processed"
model = pipe(data_dir)
print(model)
