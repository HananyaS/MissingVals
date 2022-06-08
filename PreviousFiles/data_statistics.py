import os
import json
import pandas as pd

data_prefix = "/home/dsi/shacharh/Projects/MissingVals/Data"

get_data_path = lambda data_name, missing_ratio: os.path.join(data_prefix, data_name, 'processed',
                                                              f'{str(missing_ratio) if missing_ratio is not None else ""}')
get_config_file_path = lambda data_name: os.path.join(data_prefix, data_name, 'processed', 'config.json')


def split_features_label(all_data, target_col):
    Y = all_data[target_col].values
    X = all_data.drop(columns=[target_col])
    return X, Y


def load_train_data(dataset, missing_ratio=None):
    config_file = open(get_config_file_path(dataset), 'r')
    meta_data = json.load(config_file)
    config_file.close()

    if meta_data['scalable']:
        assert missing_ratio in meta_data['missing_ratio']

    else:
        missing_ratio = None

    data_dir = get_data_path(dataset, missing_ratio)

    train = pd.read_csv(os.path.join(data_dir, 'train.csv'), index_col=0)
    train_X, train_Y = split_features_label(train, target_col=meta_data['target_col'])

    return train_X, train_Y, meta_data['name']


banknote_data = load_train_data('Banknote', 60)
diabetes_data = load_train_data('Diabetes', 70)
gdm_data = load_train_data('RoysData', None)
mNa_data = load_train_data('RonsData', None)
newGDM = load_train_data('NewGDM', None)


data_list = [banknote_data, diabetes_data, gdm_data, mNa_data, newGDM]

for X, Y, name in data_list:
    print(f'{name}:')
    print(f'\tNum features: {X.shape[1]}')
    print(f'\tNum samples: {X.shape[0]}')
    missing_ratios = ((X.isnull().sum() / X.shape[0] * 100).values)
    print(f'\tNum large missing rate columns: {(missing_ratios > 50).sum()}')
    print()