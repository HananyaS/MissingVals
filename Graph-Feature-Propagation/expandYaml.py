import os
from itertools import product


def expand_yaml():
    import yaml
    from yaml.loader import SafeLoader

    ALL_DATASETS = ['Banknote', 'Diabetes', 'RonsData', 'RoysData']
    ALL_MISSING_RATIOS = [90, 90, -1, -1]

    nni_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'nni')

    for m, status in product(os.listdir(nni_path), ['ec', 'nec']):
        method_nni_path = os.path.join(nni_path, m)
        orig_config_file = os.path.join(method_nni_path, f'config.yaml')
        with open(orig_config_file, 'r') as f:
            config = yaml.load(f, Loader=SafeLoader)

        for ds, mr in zip(ALL_DATASETS, ALL_MISSING_RATIOS):
            new_config_file = os.path.join(method_nni_path, f'config_{ds.lower()}_{status}.yaml')
            new_config = config.copy()
            new_config['experimentName'] += f' - {ds.upper()} - {status.upper()}'
            trial = new_config['trial'].copy()
            trial['command'] = trial[
                                   'command'] + f' --dataset {ds} --missing_ratio {mr} --use_existence_cols {status == "ec"}'
            new_config['trial'] = trial
            with open(new_config_file, 'w') as f:
                yaml.dump(new_config, f)


if __name__ == '__main__':
    expand_yaml()
