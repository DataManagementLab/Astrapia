******************************
Comparator
******************************

The main goal of Astrapia is to allow simple comparison
of explainers. The Comparator_ class allows for setting up 
comparisons, storing, and visualizing them.  

.. autoclass:: xaibenchmark.comparator.ExplainerComparator

    .. automethod:: add_explainer

    .. automethod:: explain_instances

    .. automethod:: explain_representative

Comparing explainers
======================

To compare :doc:`explainers`, they, as well as the Comparator_ itself first need to be initialized.
For details on how to initialize specific explainers, visit the :doc:`explainers` reference.

.. code-block:: python

    from xaibenchmark.comparator import ExplainerComparator
    from xaibenchmark.dataset import load_dataset
    from xaibenchmark.explainers import LimeExplainer, AnchorsExplainer

    data = load_dataset(...) # load a dataset for comparing on

    lime = LimeExplainer(...) # initialize a lime explainer
    anchors = AnchorsExplainer(...) # initialize an anchors explainer

    comparator = ExplainerComparator() # initialize a comparator

    comparator.add_explainer(lime, 'Lime') 
    comparator.add_explainer(anchors, 'Anchors')

Then, add each explainer using the 


Representative Sampling
========================

