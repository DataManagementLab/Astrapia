import os
import json
import pandas as pd
from sklearn.utils import Bunch
from numpy.random import RandomState


def load_csv_data(dataset_name, root_path='data', seed=0):
    """
    Parse a csv dataset to be used. This function assumes you have a folder $name under data, containing a file
    $name.data with a comma-separated training set, and a JSON file containing feature names (amongst other info).

    The default data directory ('data/') con be overwritten through the root_path parameter.

    :param seed: RNG seed for numpyRandomState
    :param dataset_name: name of the dataset, used for path/file names
    :param root_path: path to the root data directory, defaults to 'data/'
    :return: data as an astrapia.Dataset
    """

    path = os.path.join(root_path, dataset_name)

    # Load meta information
    with open(os.path.join(path, 'meta.json'), 'r') as infile:
        meta = json.load(infile)

    if dataset_name == 'breast':

        train = pd.read_csv(os.path.join(path, 'train.csv'))
        dev = pd.read_csv(os.path.join(path, 'dev.csv'))
        test = pd.read_csv(os.path.join(path, 'test.csv'))

        return Dataset(
            name=dataset_name,
            data=train,
            target=train['target'].to_frame(),
            data_dev=dev,
            target_dev=dev['target'].to_frame(),
            data_test=test,
            target_test=test['target'].to_frame(),
            target_name=meta['target_name'],
            target_categorical=meta['target_categorical'],
            target_names=meta['target_names'],
            feature_names=meta['feature_names'],
            categorical_features=meta['categorical_features'],
        )

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
    if meta['target_categorical'] and meta['categorical_features']:
        meta['categorical_features'].pop(meta['target'])

    # Remove the target from feature name list
    if meta['target'] in names:
        names.remove(meta['target'])

    # # Return the Bunch with the appropriate data chunked apart
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


class Dataset(Bunch):

    def __init__(self, 
        data: pd.DataFrame,
        feature_names: list,
        categorical_features: dict,
        
        target: pd.DataFrame,
        target_names: list,
        target_name: str,

        target_categorical: bool = True,

        name: str = None,
        data_dev: pd.DataFrame = None,
        target_dev: pd.DataFrame = None,
        data_test: pd.DataFrame = None,
        target_test: pd.DataFrame = None,
        ) -> None:
        """
        :param data: training data as pandas DataFrame
        :param feature_names: list of feature names
        :param categorical_features: dictionary of categorical features
        :param target: target feature as pandas DataFrame
        :param target_names: list of target feature values
        :param target_name: name of the target feature
        :param target_categorical: whether the target is categorical (currently only True supported)
        :param name: name of the dataset
        :param data_dev: development data as pandas DataFrame
        :param target_dev: development target as pandas DataFrame
        :param data_test: test data as pandas DataFrame
        :param target_test: test target as pandas DataFrame
        """
        super(Dataset, self).__init__(
            name=name,
            data=data,
            target=target,
            target_name=target_name,
            target_categorical=target_categorical,
            target_names=target_names,
            feature_names=feature_names,
            categorical_features=categorical_features,
            data_dev=data_dev,
            target_dev=target_dev,
            data_test=data_test,
            target_test=target_test
        )
