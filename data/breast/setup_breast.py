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
    data.frame['target'] = data.frame['target'].map(lambda x: data.target_names[x])
    train, test = train_test_split(data.frame, test_size=0.2)

    train.to_csv('./breast.data', index=False, header=False)
    test.to_csv('./breast.test', index=False, header=False)

    meta = {
        'target': 'target',
        'target_categorical':True,
        'target_names': data.target_names.tolist(),
        'feature_names': data.feature_names.tolist() + ['target'],
        'categorical_features': {
            'target': ['malignant', 'benign']
        },
        'na_values': '?'
    }

    with open('meta.json', 'w') as f:
        json.dump(meta, f, indent=2)