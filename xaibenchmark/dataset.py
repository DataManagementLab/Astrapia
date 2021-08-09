from typing import List
import pandas as pd
from typing import List, Union
from sklearn.utils import Bunch
import math

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

    def export_for_lime(self):
        """
        exports the dataset in a format readable by lime
        """

        def process_single(df):
        
            cat_df = pd.get_dummies(df, columns=self.categorical_features.keys())
            missing_cols = {cat+'_'+str(attr) for cat in self.categorical_features \
                            for attr in self.categorical_features[cat]} - set(cat_df.columns)
            for c in missing_cols:
                cat_df[c] = 0
                
            cont_idx = list(set(self.data.keys()) - set(self.categorical_features.keys()))
            cat_idx = [cat+'_'+str(attr) for cat in self.categorical_features \
                    for attr in self.categorical_features[cat]]
            idx = cont_idx + cat_idx
            return cat_df[idx]

        return {
            'data': process_single(self.data),
            'feature_names': self.data.keys(),
            'class_names': self.target_names
        }

    def export_for_anchors(self):
        """
        exports the dataset in a format readable by anchors
        """

        anchors_bunch = Bunch(
            data=self.data.copy(),
            data_dev=self.data_dev.copy(),
            data_test=self.data_test.copy(),
            target=self.target[self.target_name].map(lambda x: self.target_names.index(x)).to_numpy(),
            #target=(ab.target_test['income']=='<=50K').map(int).to_numpy(),
            target_dev=self.target_dev[self.target_name].map(lambda x: self.target_names.index(x)).to_numpy(),
            target_test=self.target_test[self.target_name].map(lambda x: self.target_names.index(x)).to_numpy(),
            target_names=self.target_names,
            feature_names=self.feature_names,
            categorical_features=self.categorical_features,
            
        )

        for feature in anchors_bunch.categorical_features.keys():
            def clearNan(x):
                return None if (type(x)==float and math.isnan(x)) else x
            transform = {clearNan(y): x for x, y in enumerate(anchors_bunch.categorical_features[feature])}
            anchors_bunch.data[feature] = anchors_bunch.data[feature].map(lambda x: transform.get(clearNan(x)))
            if anchors_bunch.data_dev is not None: anchors_bunch.data_dev[feature] = anchors_bunch.data_dev[feature].map(lambda x: transform.get(clearNan(x)))
            if anchors_bunch.data_test is not None: anchors_bunch.data_test[feature] = anchors_bunch.data_test[feature].map(lambda x: transform.get(clearNan(x)))

        for dataset in ['data', 'data_dev', 'data_test']:
            anchors_bunch[dataset] = anchors_bunch[dataset].to_numpy()

        return anchors_bunch

        

