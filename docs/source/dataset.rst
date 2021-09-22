Dataset
##############

Astrapia datasets are used to unify dataset representations across different models and explainers. 
Each interface between a model and an explainer needs to support the following dataset format.


.. autoclass:: astrapia.Dataset()
    :members:

    The Dataset class represents a dataset. It should be able to capture diverse attributes of 

    The Explainer class wraps an explainer and provides a unified interface for it. 
    Initialization depends on the specific explainer. 
    This class should *not* be used as is but rather extended.

    .. automethod:: __init__


You can easily load a dataset into a dataset object by using the ``load_csv_data`` method.

.. automethod:: astrapia.dataset.load_csv_data
        
When using your own dataset, either use the initialization function or include the following files:

- A ``datasetname.data`` file containing the training data in a csv format.

- A ``datasetname.test`` file containing the test data in a csv format.

- A ``meta.json`` file containing the following metadata in a json format.

    - ``target``: The name of the target column.
    
    - ``target_categorical``: Whether the target is categorical or not (currently only true is supported).

    - ``target_names``: A list of the values of the target categories.

    - ``features_names``: A list of the names of the features.

    - ``categorical_features``: Dictionary mapping feature names to a list of the values of the respective feature.

    - ``na_values``: Token used for missing values.

Off-the-shelf datasets
==========================
To allow for quickly starting with benchmarking, astrapia supplies multiple datasets ready to be used. Visit https://github.com/DataManagementLab/Astrapia to download them.

- **UCI Adult Data Set**: This is a multivariate dataset that has 48,842 instances for predicting whether income exceeds $50K per year based on census data, also known as census income dataset.
- **UCI Breast Cancer Wisconsin Dataset**: This breast cancer dataset is used for a relatively simple binary classification task, whether the diagnosis of the patient is malignant or benign according to its features.
