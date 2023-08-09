"""
Set up the UCI Iris dataset for Astrapia usage.
"""
import json
import os

import pandas as pd
import requests

EXAMPLE_DATASET = 'iris'
EXAMPLE_URLS = ["https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.names"]  # data as csv + meta info
EXAMPLE_NAMES = [
    'sepal_length',  # in cm
    'sepal_width',  # in cm
    'petal_length',  # in cm
    'petal_width',  # in cm
    'class'
]  # extracted manually from iris.names in this specific case
EXAMPLE_TARGET_INDEX = 4  # index of the label attribute (here: 'income') in the list above
EXAMPLE_NA_CHAR = '?'


def download_files(urls):
    """
    Download files from the given paths to the current working directory.

    :param urls: List of Strings containing URLs of desired files
    :return: None
    """

    for url in urls:
        response = requests.get(url)
        name = os.path.basename(url)
        with open(name, 'w') as out_file:
            out_file.write(response.content.decode('utf-8'))


def write_meta(data, target_index):
    """
    Construct and write a JSON meta file for the dataset.

    :param target_index: int specifying the index of the target feature
    :param data: pandas DataFrame containing training data including feature names
    :return: None
    """
    meta = {
        'target': list(data.columns)[target_index],
        'target_categorical': data[list(data.columns)[target_index]].dtype == 'object',
        'target_names': list(data[list(data.columns)[target_index]].unique()),
        'feature_names': list(data.columns),
        'categorical_features': {
            column: list(data[column].unique())
            for column in data.columns
            if data[column].dtype == 'object'
        },
        'na_values': '?',
    }

    with open('meta.json', 'w') as f:
        json.dump(meta, f, indent=2)


if __name__ == '__main__':
    """ Perform all necessary setup steps for this dataset.
    """
    download_files(EXAMPLE_URLS)  # download source files from UCI repository

    example_data = pd.read_csv(f'{EXAMPLE_DATASET}.data',
                               names=EXAMPLE_NAMES,
                               skipinitialspace=True,
                               na_values=EXAMPLE_NA_CHAR
                               )  # read training data file providing names manually, ignoring whitespace, set N/A value

    write_meta(example_data, EXAMPLE_TARGET_INDEX)  # create the json file containing the datasets meta information
