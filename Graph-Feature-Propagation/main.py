import argparse
import warnings

import os
from itertools import product

from methods import *
from utils import *

from matplotlib import pyplot as plt

warnings.filterwarnings("ignore")

str2bool = lambda v: v.lower() in ("yes", "true", "t", "1")
parser = argparse.ArgumentParser(description="Graph-Feature-Propagation")
parser.add_argument("--dataset", type=str, default="RonsData", help="dataset name")
# parser.add_argument("--dataset", type=str, default="Diabetes", help="dataset name")
parser.add_argument("--missing_ratio", type=int, default=90, help="missing ratio")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--n_epochs", type=int, default=100, help="number of n_epochs")
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--runall", type=str2bool, default=False, help="run all datasets")
parser.add_argument(
    "--nni", type=int, default=-1, help="method to run on NNI (-1 for disable NNI)"
)
parser.add_argument(
    "--use_existence_cols", type=str2bool, default=True, help="use existence columns"
)

parser.add_argument(
    "--use_best_params", type=str2bool, default=False, help="use best model"
)

parser.add_argument("--distance", type=str, default="euclidian", help="distance metric")
parser.add_argument("--alpha_hur_dist", type=float, default=0.5, help="method to run")
parser.add_argument("--beta_hur_dist", type=float, default=0.5, help="method to run")

args = parser.parse_args()

data_prefix = "/home/dsi/shacharh/Projects/MissingVals/Data"

get_data_path = lambda data_name, missing_ratio: os.path.join(
    data_prefix,
    data_name,
    "processed",
    f'{str(missing_ratio) if missing_ratio is not None else ""}',
)
get_config_file_path = lambda data_name: os.path.join(
    data_prefix, data_name, "processed", "config.json"
)

MISSING_RATIO = args.missing_ratio

ALL_DATASETS = ["Banknote", "Diabetes", "RonsData", "RoysData"]
ALL_MISSING_RATIOS = [90, 90, None, None]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def zscore_norm(data, mean=None, std=None):
    if mean is None:
        mean = data.mean()

    if std is None:
        std = data.std()

    return (data - mean) / std, mean, std


def add_existence_columns(data):
    return pd.concat(
        (
            data,
            pd.DataFrame(
                (1 - data.isna()).values,
                columns=[f"{c}_Existence" for c in data.columns],
            ),
        ),
        axis=1,
    )


def load_data(dataset, missing_ratio=None):
    config_file = open(get_config_file_path(dataset), "r")
    meta_data = json.load(config_file)
    config_file.close()

    if meta_data["scalable"]:
        assert missing_ratio in meta_data["missing_ratio"]

    else:
        missing_ratio = None

    data_dir = get_data_path(dataset, missing_ratio)

    train = pd.read_csv(os.path.join(data_dir, "train.csv"), index_col=0)
    test = pd.read_csv(os.path.join(data_dir, "test.csv"), index_col=0)
    val = pd.read_csv(os.path.join(data_dir, "val.csv"), index_col=0)

    # shuffle the data
    train = train.sample(frac=1).reset_index(drop=True)
    val = val.sample(frac=1).reset_index(drop=True)
    test = test.sample(frac=1).reset_index(drop=True)

    train_X, train_Y = split_features_label(train, target_col=meta_data["target_col"])
    test_X, test_Y = split_features_label(test, target_col=meta_data["target_col"])
    val_X, val_Y = split_features_label(val, target_col=meta_data["target_col"])

    return train_X, train_Y, test_X, test_Y, val_X, val_Y


def print_results(score, msg):
    print(f"{msg}:\t{score}")


def find_feat_label_corr():
    means = []
    maxs = []
    for ds in ALL_DATASETS:
        config_file = get_config_file_path(ds)
        config_file = open(config_file, "r")
        meta_data = json.load(config_file)
        config_file.close()

        corr_labels_dir = "feat_corr_labels"

        os.makedirs(corr_labels_dir, exist_ok=True)

        if meta_data["scalable"]:
            mr_lst = meta_data["missing_ratio"][:1]
        else:
            mr_lst = [None]

        for mr in mr_lst:
            train = pd.read_csv(
                os.path.join(get_data_path(ds, mr), "train.csv"), index_col=0
            )
            corr = train.corr()
            corr_label = abs(
                corr[meta_data["target_col"]].drop(meta_data["target_col"])
            )
            max_corr, min_corr, mean_corr, median_corr = (
                corr_label.max(),
                corr_label.min(),
                corr_label.mean(),
                corr_label.median(),
            )
            means.append(mean_corr)
            maxs.append(max_corr)
            with open(os.path.join(corr_labels_dir, f"{ds}.txt"), "w") as f:
                f.write(", ".join([str(x.round(2)) for x in corr_label.values]))
                f.write("\n")
                f.write("Max: " + str(max_corr.round(2)))
                f.write("\n")
                f.write("Min: " + str(min_corr.round(2)))
                f.write("\n")
                f.write("Mean: " + str(mean_corr.round(2)))
                f.write("\n")
                f.write("Median: " + str(median_corr.round(2)))

    # plot a bar chart of the means and maxs next to each other
    plt.bar(range(len(maxs)), maxs, color="c", label="Max")
    plt.bar(range(len(means)), means, color="y", label="Mean")
    plt.xticks(range(len(ALL_DATASETS)), ALL_DATASETS)
    plt.title("Mean and Max Correlation of Labels")
    plt.legend()
    plt.savefig(os.path.join(corr_labels_dir, "mean_max_corr_labels.png"))
    plt.show()


@timer
def main(
    dataset,
    missing_ratio,
    use_existence_cols=False,
    methods_to_run="all",
    method_to_return: int = 0,
    params=None,
):
    assert methods_to_run == "all" or method_to_return in methods_to_run

    train_X, train_Y, test_X, test_Y, val_X, val_Y = load_data(
        dataset, missing_ratio=missing_ratio
    )

    train_X, mean, std = zscore_norm(train_X)
    val_X, _, _ = zscore_norm(val_X, mean, std)
    test_X, _, _ = zscore_norm(test_X, mean, std)

    features_corr = abs(train_X.copy(deep=True).fillna(0).corr()).values

    results = {}

    # naive classifier - random forest
    if methods_to_run == "all" or 1 in methods_to_run:
        print("Running naive classifier...")

        if args.use_best_params:
            params = get_best_params(dataset, 1)

        naive_score = run_naive(
            train_X,
            train_Y,
            test_X,
            test_Y,
            use_existence_cols=use_existence_cols,
            params=params,
        )

        print("Naive classifier done!")
        print_results(naive_score, "Naive")
        results[1] = naive_score

    # method 2 - imputation using KNN (GFP method) + naive classifier
    if methods_to_run == "all" or 2 in methods_to_run:
        print("Running method 2...")

        if args.use_best_params:
            params = get_best_params(dataset, 2)

        method_2_score = run_method_2(
            train_X,
            train_Y,
            test_X,
            test_Y,
            distance=args.distance,
            n_iters=100,
            use_existence_cols=use_existence_cols,
            params=params,
            metric_params=get_metric_params(),
        )
        print("Method 2 done!")
        print_results(method_2_score, "Method 2")
        results[2] = method_2_score

    # method 3
    if methods_to_run == "all" or 3 in methods_to_run:
        print("Running method 3...")

        if args.use_best_params:
            params = get_best_params(dataset, 3)

        method_3_score = run_method_3(
            train_X,
            train_Y,
            test_X,
            test_Y,
            features_corr,
            distance=args.distance,
            lr=args.lr,
            batch_size=args.batch_size,
            n_epochs=args.epochs,
            device=device,
            verbose=False,
            params=params,
            metric_params=get_metric_params(),
        )
        print("Method 3 done!")
        print_results(method_3_score, "Method 3")
        results[3] = method_3_score

    # method 4
    if methods_to_run == "all" or 4 in methods_to_run:
        print("Running method 4...")

        if args.use_best_params:
            params = get_best_params(dataset, 4)

        method_4_score = run_method_4(
            train_X,
            train_Y,
            test_X,
            test_Y,
            features_corr,
            distance=args.distance,
            lr=args.lr,
            batch_size=args.batch_size,
            n_epochs=args.epochs,
            device=device,
            verbose=False,
            params=params,
            metric_params=get_metric_params(),
        )
        print("Method 4 done!")
        print_results(method_4_score, "Method 4")
        results[4] = method_4_score

    # method 6
    if methods_to_run == "all" or 6 in methods_to_run:
        print("Running method 6...")

        if args.use_best_params:
            params = get_best_params(dataset, 6)

        method_6_score = run_method_6(
            train_X,
            train_Y,
            test_X,
            test_Y,
            features_corr,
            lr=args.lr,
            batch_size=args.batch_size,
            n_epochs=args.epochs,
            device=device,
            verbose=False,
            params=params,
        )
        print("Method 6 done!")
        print_results(method_6_score, "Method 6")
        results[6] = method_6_score

    return results[method_to_return]


def get_best_params(ds, method):
    params_path = os.path.join(
        "nni_results",
        f"method{method}",
        "params",
        f"{ds.lower()}_params_{'n' if not args.use_existence_cols else ''}ec.json",
    )

    with open(params_path, "r") as f:
        params = json.load(f)

    return params


def get_metric_params():
    if args.distance == "euclidian":
        return {}

    elif args.distance == "hur_dist":
        return {
            p: v
            for p, v in zip(
                ["alpha", "beta"], [args.alpha_hur_dist, args.beta_hur_dist]
            )
            if v is not None
        }

    raise NotImplementedError


@run_nni
def main_nni(nni_params, *args, **kwargs):
    return main(params=nni_params, *args, **kwargs)


if __name__ == "__main__":
    methods_to_run = [4]

    assert not args.runall or args.nni == -1
    assert not args.use_best_params or args.nni == -1

    if args.nni > -1:
        methods_to_run = [args.nni]

    main_func = main_nni if args.nni != -1 else main

    # run on all datasets
    if args.runall:
        for (ds, mr), use_existence_cols in product(
            *[list(zip(ALL_DATASETS, ALL_MISSING_RATIOS)), [True, False]]
        ):
            # name = dd.split("/")[0].upper()
            print(
                f'Running {ds} with missing ratio {mr} with{(not use_existence_cols) * "out"}'
                f" existence columns"
            )
            # ds_dir = get_data_path(ds, mr)
            main_func(
                ds,
                mr,
                use_existence_cols=use_existence_cols,
                methods_to_run=methods_to_run,
                method_to_return=args.nni if args.nni != -1 else methods_to_run[0],
            )

            # os.rename('loss.png', f'plots/loss_{ds}{f"_{mr}" if mr is not None else ""}_{use_existence_cols}.png')
            # os.rename('auc.png', f'plots/auc_{ds}{f"_{mr}" if mr is not None else ""}_{use_existence_cols}.png')
            print("~~~~~~~~~~~~~~~~~~~")

    # run only the dataset given in the arguments
    else:
        main_func(
            dataset=args.dataset,
            missing_ratio=MISSING_RATIO,
            use_existence_cols=args.use_existence_cols,
            methods_to_run=methods_to_run,
            method_to_return=args.nni if args.nni != -1 else methods_to_run[0],
        )
