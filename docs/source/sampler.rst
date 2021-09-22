.. _sampler:

Samplers
==========

As explaining many instances can take a large amount of time, Astrapia provides the :ref:`Sampler <sampler>` class.
This classes goal is to choose representative samples given a dataset and a strategy. By only having to explain representative samples, 
computation time can be drastically reduced while still representing most of the dataset.

.. autoclass:: astrapia.samplers.base_sampler.Sampler

    .. automethod:: sample

Writing your own Sampler
-----------------------------
To write your own sampler, simply extend the :ref:`Sampler <sampler>` class and implement the ``sample`` method. 
You may require any additional arguments for the ``sample`` method. 
However, if you want to share your sampler, make sure that missing arguments lead to well-explained exceptions.

.. code-block:: python

    class YourOwnSampler(Sampler):
        """
        A Sampler returning the first n instances from the dataset
        """

        def sample(self, dataset, count, **kwargs):

            return dataset.data.iloc[[0:count]]

Off-the-shelf Samplers
---------------------------

To allow users to quickly start benchmarking Explainers, Astrapia includes some freely usable samplers.


SP-Lime Sampler 
********************

.. autoclass:: astrapia.samplers.splime.SPLimeSampler

    .. automethod:: sample

Random Sampler
****************

.. autoclass:: astrapia.samplers.random.RandomSampler

    .. automethod:: sample


