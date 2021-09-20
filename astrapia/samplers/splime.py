from astrapia.samplers.base_sampler import Sampler
import astrapia as xb
from lime import submodular_pick
import pandas as pd
import numpy as np
import lime

class SPLimeSampler(Sampler):

    def sample(self, data: xb.Dataset, count: int, pred_fn, sample_size=5000, **kwargs):
        """
        Samples a dataset using the SPLime algorithm.

        :param data: The astrapia dataset to sample from.
        :param count: The number of samples to return.
        :param pred_fn: A function that takes multiple data point and returns a probabiliy distribution over both classes for each input.
        :param sample_size: The number of samples to take.
        :return: A list of samples.
        """

        if pred_fn is None:
            raise ValueError("splime requires pred_fn attribute must be set.")

        # Convert data to numpy array
        data_np = data.data.to_numpy()
        
        # Convert categorical features to ids
        categorical_idxs = [i for i, label in (enumerate(data.feature_names)) if label in data.categorical_features.keys()]
        for feature_idx in categorical_idxs:
            feature_map = {str(feature): idx for idx, feature in enumerate(data.categorical_features[data.feature_names[feature_idx]])}
            data_np[:, feature_idx] = np.vectorize(lambda x: feature_map[str(x)])(data_np[:, feature_idx])

        # Initialize a lime explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(data_np, feature_names=data.feature_names, class_names=data.target_names, categorical_features=categorical_idxs, verbose=False, discretize_continuous=False)

        # wrapper around pred_fn to make it compatible with lime
        def custom_predict(X):
            result = pd.DataFrame(X, columns=data.feature_names)
            for feature_idx in categorical_idxs:
                result[data.feature_names[feature_idx]] = result[data.feature_names[feature_idx]].map(lambda x: data.categorical_features[data.feature_names[feature_idx]][int(x)])
            return pred_fn(result)

        # Run submodular pick
        sp_obj = submodular_pick.SubmodularPick(explainer, data_np, custom_predict, sample_size=20, num_exps_desired=count)

        # Process features as submodular pick returns categorical features in the form: feature=index
        def process_feature(name: str, value):
            
            if name.rsplit('=', 1)[0] not in data.categorical_features.keys():
                return float(value)
            
            f_name, f_idx = name.rsplit('=', 1)
            f_idx = int(f_idx)

            return data.categorical_features[f_name][f_idx]

        # Combine the samples chosen by submodular pick 
        samples_df = pd.DataFrame(columns=data.data.columns)
        for exp in sp_obj.sp_explanations:
            series = pd.Series([process_feature(name, value) for name, value in zip(exp.domain_mapper.feature_names, exp.domain_mapper.feature_values)], index=data.data.columns)
            samples_df = samples_df.append(series, ignore_index=True)

        return samples_df
