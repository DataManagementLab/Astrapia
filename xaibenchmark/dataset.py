from typing import List
import pandas as pd
from sklearn.utils import Bunch


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
        :param data:
        :param categorical_features:
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

        

