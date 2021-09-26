"""
Setup the UCI breast cancer dataset for Astrapia usage.
"""

import json
import os
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    data = load_breast_cancer(as_frame=True)
    train, test = train_test_split(data.frame, test_size=0.2)
    train, dev = train_test_split(train, test_size=0.25)

    train.to_csv('./train.csv')
    test.to_csv('./test.csv')
    dev.to_csv('./dev.csv')

    meta = {
        'target_name': data.frame.columns[-1],
        'target_categorical':True,
        'target_names': data.target_names.tolist(),
        'feature_names': data.feature_names.tolist(),
        'categorical_features': {
            column: list(data.frame[column].unique())
            for column in data.frame.columns
            if data.frame[column].dtype == 'object'
        }
    }

    with open('meta.json', 'w') as f:
        json.dump(meta, f, indent=2)