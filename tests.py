from Models.xgb import XGBoost as xgb
from Datasets.tabDataPair import TabDataPair as TDP
from Models.graphClassification import ValuesAndGraphStructure as VAGS
import json
import torch


data_dir = "Data/RoysData/processed"
train_filename = "train.csv"
val_filename = "test.csv"

train = TDP.load(
    data_dir=data_dir,
    data_file_name=train_filename,
    fill_na=True,
    add_existence_cols=True,
    shuffle=True,
)

val = TDP.load(
    data_dir=data_dir,
    data_file_name=val_filename,
    fill_na=True,
    add_existence_cols=True,
    shuffle=True,
)

with open(
    "Graph-Feature-Propagation/nni_results/method1/params/roysdata_params_ec.json", "r"
) as f:
    params = json.load(f)

xgb_clf = xgb(**params)
xgb_clf.fit(train.to_loader())
xgb_clf.evaluate(loader=train.to_loader())
xgb_clf.evaluate(loader=val.to_loader())

with open("Graph-Feature-Propagation/recieved_params.json", "r") as f:
    recieved_params = json.load(f)

vags = VAGS(
    nodes_number=train.get_num_features(),
    feature_size=1,
    device=torch.device("cpu"),
    RECEIVED_PARAMS=recieved_params,
)