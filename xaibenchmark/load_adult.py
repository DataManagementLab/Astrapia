import os
import json
import pandas as pd
from sklearn.utils import Bunch
from numpy.random import RandomState
from xaibenchmark.dataset import Dataset


def load_csv_data(dataset_name, root_path='data', seed=0):
    """
    Parse a csv dataset to be used. This function assumes you have a folder $name under data, containing a file
    $name.data with a comma-separated training set, and a JSON file containing feature names (amongst other info).

    The default data directory ('data/') con be overwritten through the root_path parameter.
    For exemplary data preparation, see data/adult/setup_adult.py

    :param seed: RNG seed for numpyRandomState
    :param dataset_name: name of the dataset, used for path/file names
    :param root_path: path to the root data directory, defaults to 'data/'
    :return: data as an sklearn Bunch
    """
    path = os.path.join(root_path, dataset_name)

    # Load meta information
    with open(os.path.join(path, 'meta.json'), 'r') as infile:
        meta = json.load(infile)

    names = meta['feature_names']  # just for convenience

    # Load training data, splitting off dev set
    train_dev_data = pd.read_csv(os.path.join(path, f'{dataset_name}.data'), names=names, skipinitialspace=True,
                                 na_values=meta['na_values'])
    rng = RandomState(seed) if seed else RandomState()
    train = train_dev_data.sample(frac=0.7, random_state=rng)
    dev = train_dev_data.loc[~train_dev_data.index.isin(train.index)]

    # Load test data
    test = pd.read_csv(os.path.join(path, f'{dataset_name}.test'), names=names, skipinitialspace=True,
                       na_values=meta['na_values'])

    # Remove the target from the categorical features if necessary
    if meta['target_categorical']:
        meta['categorical_features'].pop(meta['target'])

    # Remove the target from feature name list
    names.remove(meta['target'])

    # Return the Bunch with the appropriate data chunked apart
    return Dataset(
        name=dataset_name,
        data=train[names],
        target=pd.DataFrame(train[meta['target']]),
        data_dev=dev[names],
        target_dev=pd.DataFrame(dev[meta['target']]),
        data_test=test[names],
        target_test=pd.DataFrame(test[meta['target']]),
        target_name=meta['target'],
        target_categorical=meta['target_categorical'],
        target_names=meta['target_names'],
        feature_names=meta['feature_names'],
        categorical_features=meta['categorical_features'],
    )

