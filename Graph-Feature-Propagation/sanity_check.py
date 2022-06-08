import os
import csv
import json
from itertools import product


class SanityCheck:
    def __init__(
        self, search_space_file, method_num, method_func, intermediate_result_file
    ):
        self.method_num = method_num
        self.method_func = method_func

        self.search_space_file = search_space_file
        self.search_space_keys, self.search_space_values = self.get_search_space()

        assert intermediate_result_file.endswith(".csv")

        self.intermediate_result_file = intermediate_result_file
        self.validate_csv_path(self.intermediate_result_file, ["dataset", *self.search_space_keys, "score"])

    def get_search_space(self):
        with open(self.search_space_file, "r") as f:
            data = json.load(f)
        keys = data.keys()
        values = list(product(*data.values()))
        return keys, values

    def validate_csv_path(self, path, header_lst=None):
        assert path.endswith(".csv"), "Path must end with .csv"

        if "/" in path:
            os.makedirs(os.path.dirname(path), exist_ok=True)

        if not os.path.isfile(path):
            assert header_lst is not None, "Header must be provided for new file"

            with open(path, "w") as f:
                dw = csv.DictWriter(f, delimiter=",", fieldnames=header_lst)
                dw.writeheader()

        return self

    def data_to_csv(self, data, path, header_lst=None):
        self.validate_csv_path(path, header_lst)

        with open(path, "a") as f:
            writer = csv.writer(f, delimiter=",")
            writer.writerows(data)

        return self

    def run(self, ds_list, ds_names, additional_params={}, ds_result_files=None):
        if ds_result_files is not None:
            for f in ds_result_files:
                self.validate_csv_path(f, ["dataset", *self.search_space_keys, "score"])

        intermediate_results_data = []

        for i, (ds_name, ds) in enumerate(zip(ds_names, ds_list)):
            train_X, train_Y, test_X, test_Y = ds

            print(f"Running sanity check on {ds_name}")

            ds_results = []

            for i, comb in enumerate(self.search_space_values):
                kwargs = dict(zip(self.search_space_keys, comb))

                kwargs["train_X"] = train_X
                kwargs["train_Y"] = train_Y
                kwargs["test_X"] = test_X
                kwargs["test_Y"] = test_Y

                result = self.method_func(**kwargs, **additional_params)

                data_cache = [ds_name, *comb, result]

                if ds_result_files is not None:
                    ds_results.append(data_cache)

                intermediate_results_data.append(data_cache)

            self.data_to_csv(ds_results, ds_result_files[i])
            print(f"Finished sanity check on {ds_name}")

        self.data_to_csv(intermediate_results_data, self.intermediate_result_file)


def test():
    from main import load_data, zscore_norm
    from methods import run_naive

    dataset = "Banknote"
    missing_ratio = 90

    train_X, train_Y, test_X, test_Y, val_X, val_Y = load_data(
        dataset, missing_ratio=missing_ratio
    )

    train_X, mean, std = zscore_norm(train_X)
    val_X, _, _ = zscore_norm(val_X, mean, std)
    test_X, _, _ = zscore_norm(test_X, mean, std)

    search_space_file = (
        os.getcwd()
        + "/Projects/MissingVals/Graph-Feature-Propagation/sanity_checks/search_spaces/method1.json"
    )
    intermediate_result_file = "sanity_checks/intermediate_results/method1.csv"

    ds_result_files = [f"sanity_checks/intermediate_results/{dataset}_results.csv"]

    sc = SanityCheck(
        search_space_file,
        1,
        run_naive,
        intermediate_result_file,
    )

    sc.run(
        [
            (train_X, train_Y, test_X, test_Y),
        ],
        [dataset],
        ds_result_files=ds_result_files,
    )


if __name__ == "__main__":
    test()
