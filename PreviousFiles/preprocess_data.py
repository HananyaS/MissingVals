import os
import pandas as pd
import numpy as np

rons_data_dir = "../data/RonsData/raw"
rons_data_path = os.path.join(rons_data_dir, "data.csv")

roys_data_dir = "../data/RoysDataOld/processed"


def preprocess_rons_data():
    target_col = "Target"
    cols_to_drop = ["Acquiror", "cik"]

    data = pd.read_csv(rons_data_path)
    data = data.drop(columns=cols_to_drop)

    target = data[target_col]
    data = data.drop(columns=[target_col])

    # normalize data using z-score
    data = (data - data.mean()) / data.std()
    data = data.join(target)

    test_ratio = 0.2
    val_ratio = 0.2
    train_data = data.sample(frac=1)
    test_data = train_data.sample(frac=test_ratio)
    train_data = train_data.drop(test_data.index)
    val_data = train_data.sample(frac=val_ratio)
    train_data = train_data.drop(val_data.index)

    # reset index
    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)
    val_data = val_data.reset_index(drop=True)

    os.makedirs("../data/RonsData/processed", exist_ok=True)
    train_data.to_csv("Data/RonsData/processed/train.csv")
    test_data.to_csv("Data/RonsData/processed/test.csv")
    val_data.to_csv("Data/RonsData/processed/val.csv")

    n_missings = pd.DataFrame(
        {
            "Train": train_data.drop(columns=[target_col]).isnull().sum().values.T
            / train_data.shape[0],
            "Test": test_data.drop(columns=[target_col]).isnull().sum().values.T
            / test_data.shape[0],
            "Val": val_data.drop(columns=[target_col]).isnull().sum().values.T
            / val_data.shape[0],
        },
        index=train_data.columns[:-1],
    )

    print(
        "Train ratio:",
        train_data.shape[0]
        / (train_data.shape[0] + test_data.shape[0] + val_data.shape[0]),
    )
    print(
        "Test ratio:",
        test_data.shape[0]
        / (train_data.shape[0] + test_data.shape[0] + val_data.shape[0]),
    )
    print(
        "Val ratio:",
        val_data.shape[0]
        / (train_data.shape[0] + test_data.shape[0] + val_data.shape[0]),
    )

    n_missings.to_csv("Data/RonsData/processed/missing.csv")
    print(n_missings)


def preprocess_new_gdm_data():
    data = pd.read_csv("../data/NewGDM/raw/after_preprocess.csv", index_col=0)
    tags = pd.read_csv("../data/NewGDM/raw/new_data.csv", index_col=0)['Study GDM ']
    tags = tags.replace(['b no GDM', 'a GDM'], [0, 1])

    data['tags'] = tags

    # convert the categorical tags to numbers
    # tags = tags.replace(['לא'], [1, 0])

    # shuffle data
    data = data.sample(frac=1)

    test_ratio = 0.2
    val_ratio = 0.2

    train_data, test_data, val_data = split_missing_df(
        data, "tags", test_ratio, val_ratio
    )

    os.makedirs("../data/NewGDM/processed", exist_ok=True)
    train_data.to_csv("Data/NewGDM/processed/train.csv")
    test_data.to_csv("Data/NewGDM/processed/test.csv")
    val_data.to_csv("Data/NewGDM/processed/val.csv")


def preprocess_roys_data():
    raw_data = pd.read_csv("../data/RoysDataOld/raw/week_14_ new.csv")
    data = raw_data.drop(columns=raw_data.columns[:71], inplace=False)

    # translate columns from hebrew to english
    data.rename(
        columns={
            "עישון": "Smoking",
            "גיל": "Age",
            "גובה": "Height",
            "משקל לפני הריון": "W-BP",
            "bmi לפני היריון": "BMI-BP",
            "לחץ דם": "BloodPressure",
            "הריון בר סיכון": "Risky",
        },
        inplace=True,
    )

    labels = pd.read_csv("../data/RoysDataOld/raw/gdm.csv")
    data["tags"] = labels.values[:, 1]

    test_ratio = 0.2
    val_ratio = 0.2

    train_data, test_data, val_data = split_missing_df(
        data, "tags", test_ratio, val_ratio
    )

    assert (
        train_data.shape[0] + test_data.shape[0] + val_data.shape[0]
        == raw_data.shape[0]
    )

    os.makedirs("../data/RoysDataOld/processed", exist_ok=True)
    train_data.to_csv("Data/RoysData/processed/train.csv")
    test_data.to_csv("Data/RoysData/processed/test.csv")
    val_data.to_csv("Data/RoysData/processed/val.csv")


# def preprocess_sapirs_data():
#     import data_for_efs_newformat


def preprocess_diabetes():
    columns = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
        "Class",
    ]

    data = pd.read_csv(
        "../data/Diabetes/raw/pima_indians_diabetes.csv", header=None, names=columns
    )
    missing_ratios = list(range(50, 100, 10))
    full_cols = np.random.choice(
        len(data.columns), size=int(0.33 * len(data.columns)), replace=False
    )
    cols_w_nans = list(set(range(len(data.columns))) - set(full_cols))

    for mr in missing_ratios:
        data_vals = data.values
        for j in cols_w_nans:
            if data.columns[j] == "Class":
                continue

            missing_indices = np.random.choice(
                data_vals.shape[0],
                size=int(mr * data_vals.shape[0] / 100),
                replace=False,
            )
            data_vals[missing_indices, j] = np.nan

        data_w_missings = pd.DataFrame(data_vals, columns=data.columns)

        test_ratio = 0.2
        val_ratio = 0.2

        train_data, test_data, val_data = split_missing_df(
            data_w_missings,
            target_col="Class",
            test_ratio=test_ratio,
            val_ratio=val_ratio,
        )

        os.makedirs(f"Data/Diabetes/processed/{mr}", exist_ok=True)
        train_data.to_csv(f"Data/Diabetes/processed/{mr}/train.csv")
        test_data.to_csv(f"Data/Diabetes/processed/{mr}/test.csv")
        val_data.to_csv(f"Data/Diabetes/processed/{mr}/val.csv")


def preprocess_banknote():
    columns = ["variance", "skewness", "curtosis", "entropy", "class"]

    data = pd.read_csv(
        "../data/Banknote/raw/data_banknote_authentication.txt", header=None, names=columns
    )
    missing_ratios = list(range(50, 100, 10))
    full_cols = np.random.choice(
        len(data.columns), size=int(0.33 * len(data.columns)), replace=False
    )
    cols_w_nans = list(set(range(len(data.columns))) - set(full_cols))

    for mr in missing_ratios:
        data_vals = data.values
        for j in cols_w_nans:
            if data.columns[j] == "class":
                continue

            missing_indices = np.random.choice(
                data_vals.shape[0],
                size=int(mr * data_vals.shape[0] / 100),
                replace=False,
            )
            data_vals[missing_indices, j] = np.nan

        data_w_missings = pd.DataFrame(data_vals, columns=data.columns)

        test_ratio = 0.2
        val_ratio = 0.2

        train_data, test_data, val_data = split_missing_df(
            data_w_missings,
            target_col="class",
            test_ratio=test_ratio,
            val_ratio=val_ratio,
        )

        os.makedirs(f"Data/Banknote/processed/{mr}", exist_ok=True)

        train_data.to_csv(f"Data/Banknote/processed/{mr}/train.csv")
        test_data.to_csv(f"Data/Banknote/processed/{mr}/test.csv")
        val_data.to_csv(f"Data/Banknote/processed/{mr}/val.csv")


def split_missing_df(df, target_col, test_ratio=0.2, val_ratio=0.2, normalize=True):
    train_data = df.sample(frac=1)

    test_data = train_data.sample(frac=test_ratio)
    train_data = train_data.drop(test_data.index)
    val_data = train_data.sample(frac=val_ratio)
    train_data = train_data.drop(val_data.index)

    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)
    val_data = val_data.reset_index(drop=True)


    train_Y = train_data[target_col]
    train_X = train_data.drop(columns=[target_col])

    test_Y = test_data[target_col]
    test_X = test_data.drop(columns=[target_col])

    val_Y = val_data[target_col]
    val_X = val_data.drop(columns=[target_col])

    if normalize:
        mu, sigma = train_X.mean(), train_X.std()
        train_X = (train_X - mu) / sigma
        test_X = (test_X - mu) / sigma
        val_X = (val_X - mu) / sigma

    train_data = pd.concat([train_X, train_Y], axis=1)
    test_data = pd.concat([test_X, test_Y], axis=1)
    val_data = pd.concat([val_X, val_Y], axis=1)

    return train_data, test_data, val_data


if __name__ == "__main__":
    preprocess_new_gdm_data()
    # preprocess_roys_data()
    # preprocess_sapirs_data()
    """
    train_data = pd.read_csv('Data/RoysData/processed/train.csv', index_col=0)
    test_data = pd.read_csv('Data/RoysData/processed/test.csv', index_col=0)
    val_data = pd.read_csv('Data/RoysData/processed/val.csv', index_col=0)

    print(train_data.shape)
    print(test_data.shape)
    print(val_data.shape)

    print(train_data.isnull().mean())
    """
