******************************
Comparator
******************************

The main goal of Astrapia is to allow simple comparison
of explainers. The Comparator_ class allows for setting up 
comparisons, storing, and visualizing them.  

.. autoclass:: astrapia.comparator.ExplainerComparator

    .. automethod:: add_explainer

    .. automethod:: explain_instances

    .. automethod:: explain_representative

Comparing explainers
======================

To compare :doc:`explainers`, they, as well as the Comparator_ itself first need to be initialized.
For details on how to initialize specific explainers, visit the :doc:`explainers` reference.

.. code-block:: python

    from astrapia.comparator import ExplainerComparator
    from astrapia.dataset import load_dataset
    from astrapia.explainers import LimeExplainer, AnchorsExplainer

    data = load_dataset(...) # load a dataset for comparing on

    lime = LimeExplainer(...) # initialize a lime explainer
    anchors = AnchorsExplainer(...) # initialize an anchors explainer

    comparator = ExplainerComparator() # initialize a comparator

Then, add each explainer using the ``add_explainer`` method.

.. code-block:: python

    comparator.add_explainer(lime, 'Lime') 
    comparator.add_explainer(anchors, 'Anchors')

Now, you can explain instances using the ``explain_instances`` method.

.. code-block:: python

    comparator.explain_instances(data.train.iloc[[0, 1, 2]])

Visualize the results using the ``visualization`` module.


Representative Sampling
========================

As explaining can take a bit of time, it is often useful to 
compare explainers on a representative subset of samples.
By using smart :doc:`sampler`, you can limit the number of samples
while keeping a meaninful comparison.

Instead of using the ``explain_instances`` method, you can run

.. code-block:: python

    comparator.explain_representative(data, sampler='splime', count=5, pred_fn=pred_fn)

Notice that for this method, you need to specify a prediction function as 
the :doc:`SP-Lime Sampler <sampler>` needs to know how the model is predicting different samples.

