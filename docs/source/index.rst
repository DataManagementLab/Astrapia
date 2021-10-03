.. astrapia documentation master file, created by
   sphinx-quickstart on Fri Aug 27 20:53:57 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Astrapia's documentation!
====================================

*Astrapia* is a Python framework for comparing and evaluating tabular post-hoc explainers.
Explainers can be used to understand hte behaviour of modern opaque and intransparent machine learning models. 
Still, some explainer are better than others. This framework is designed to compare them using a generalised set of metrics.

Astrapia is **not** a framework for ranking explainers. It barely aids users in judging advantages and disadvantages of different explainers.

.. toctree::
   :caption: Table of Contents
   :maxdepth: 2

   installation
   math
   metric_ref
   dataset
   explainers
   comparator
   transfer
   sampler
   developers



Quickstart
######################

See the :doc:`installation` section to learn how to install Astrapia. 

Currently, we offer two examples: ``UCI adult`` dataset and ``UCI breast cancer* dataset``. These examples can be found under ```notebooks/AstrapiaComparatorDemo.ipynb`` Here we show you how to use Astrapia to compare different explainers using the ``UCI adult`` dataset. First, navigate into ``data/adult/`` and run

.. code-block:: bash

   $ python setup_adult.py

Files for the datasets will be generated under the corresponding folder. Now load the dataset:

.. code-block:: python

   data = dataset.load_csv_data('adult', root_path='../data')

Import the dependencies

.. code-block:: python

   import astrapia as xb
   from astrapia import explainers, dataset
   from astrapia.comparator import ExplainerComparator
   from astrapia.visualize_metrics import print_metrics, load_metrics_from_json
   import sklearn.ensemble

Then, train a machine learning classifier that you want to explain.

.. code-block:: python

   rf = sklearn.ensemble.RandomForestClassifier(n_estimators=50, n_jobs=5)
   rf.fit(xb.utils.onehot_encode(data.data, data), data.target.to_numpy().reshape(-1))
   pred_fn = lambda x: rf.predict_proba(xb.utils.onehot_encode(x, data))

Prepare post-hoc explainers that you want to compare. Here we chose LIME and Anchors.

.. code-block:: python

   ex_lime = explainers.LimeExplainer(data, pred_fn, discretize_continuous=False)
   ex_anchors = explainers.AnchorsExplainer(data, pred_fn, 0.9)

Astrapia offers a convenient interface to compare between explainers by instantiating a `ExplainerComparator` class and appending the explainer to it:

.. code-block:: python

   comp = ExplainerComparator()
   comp.add_explainer(ex_anchors, 'ANCHORS 0.9')
   comp.add_explainer(ex_lime, 'LIME')

Choose an instance or multiple instances to explain:

.. code-block:: python

   comp.explain_instances(data.data.iloc[[0]]) # single instance

or

.. code-block:: python

   comp.explain_instances(data.data.iloc[[111, 222, 333, 444]]) # multiple instances

Store metric data as json and assert that storing and reloading data does not modify it.

.. code-block:: python

   metric_data = comp.get_metric_data()
   comp.store_metrics()
   assert load_metrics_from_json('metrics.json') == metric_data

To visualize metrics as tables or bar charts:

.. code-block:: python

   # show all explainers
   print_metrics(metric_data, plot='table', show_metric_with_one_value=True)
   print_metrics(metric_data, plot='bar', show_metric_with_one_value=False)

   # show single explainer result
   print_metrics(metric_data, explainer='ANCHORS 0.9')
   print_metrics(metric_data, plot="bar", explainer='LIME')


* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
