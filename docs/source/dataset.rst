Dataset
##############

Astrapia datasets are used to unify dataset representations across different models and explainers. 
Each interface between a model and an explainer needs to support the following dataset format.


.. autoclass:: astrapia.Dataset
    :members:

    The Dataset class represents a dataset. It should be able to capture diverse attributes of 

    The Explainer class wraps an explainer and provides a unified interface for it. 
    Initialization depends on the specific explainer. 
    This class should *not* be used as is but rather extended.

    .. py:method:: __init__()

        Returns a list of implemented metrics


Off-the-shelf datasets
==========================
To allow for quickly starting with benchmarking, astrapia supplies multiple datasets ready to be used.

- **UCI Adult Data Set**: This is a multivariate dataset that has 48,842 instances for predicting whether income exceeds $50K per year based on census data, also known as census income dataset.
- **UCI Breast Cancer Wisconsin Dataset**: This breast cancer dataset is used for a relatively simple binary classification task, whether the diagnosis of the patient is malignant or benign according to its features.
