For Developers
---------------------

Astrapia is a fully open-source project. Anyone is welcome to contribute to the project. 
This page provides some guidelines for adding new :doc:`explainers`, :doc:`sampler` and :doc:`transfer` 
functions to the main Astrapia codebase for everyone to use. This might be especially useful for 
developers contributing new :doc:`explainers` as users will be able to use and understand your Explainer's advantages.

Documentation
******************
Due to the size of the Astrapia codebase, this extensive documentation is compiled to help new users get started. 
If you add new functionality, please document it in the appropriate section.
The following segment provides you with the tools needed to get started working on the documentation.

Astrapia is using sphinx_ and reStructuredText_ to generate the documentation you are reading.

To improve on it, navigate to the ``docs`` directory. To add content, look into the ``docs/source`` directory and add your content at the appropriate place.
To see your changes on a linux based distribution, run 

.. code-block:: bash

    $ make html

On Windows, run 

.. code-block:: bash

    $ ./Makefile html

The documentation will be generated in the ``docs/build/html`` directory. Use a regular html viewer such as ``firefox`` or ``chrome`` to view the documentation. 
To build the documentation to other formats such as PDF or LaTeX, view the `sphinx documentation`_ 

.. _sphinx: https://www.sphinx-doc.org/en/master/
.. _reStructuredText: https://en.wikipedia.org/wiki/ReStructuredText
.. _sphinx documentation: https://www.sphinx-doc.org/en/master/usage/quickstart.html


Explainers
*************

Astrapia aims to provide a large set of off-the-shelf :doc:`explainers <explainers>`. 
The more :doc:`explainers <explainers>` and types of :doc:`explainers <explainers>` we provide, the more users will learn about the advantages and disadvantages of their own :doc:`explainers <explainers>`.
As such, we welcome you to contribute your own :doc:`explainers <explainers>` to the Astrapia codebase.

To add your own :doc:`explainer <explainers>`, visit the ``astrapia/explainers`` directory and add a new python file containing your :doc:`explainer <explainers>`. 
Then, add your :doc:`explainer <explainers>` to the ``astrapia/explainers/__init__.py`` file.

You should now be able to import your :doc:`explainer <explainers>` with

.. code-block:: python

    from astrapia.explainers import MyExplainer

Please make sure to document your :doc:`explainer <explainers>` both in-code and in the documentation_.

Samplers
***************

To add your own :doc:`sampler <sampler>`, visit the ``astrapia/samplers`` directory and add a new python file containing your :doc:`sampler <sampler>`.

To simplify the usage of your :doc:`sampler <sampler>`, you can add your :doc:`sampler <sampler>` to the ``explain_representative`` function in the ``astrapia/comparator.py`` file. 
Otherwise, you can just use your :doc:`sampler <sampler>` object directly.

Please make sure to document your :doc:`sampler <sampler>` both in-code and in the documentation_.



Transfer Functions
***********************

Evaluating :doc:`explainers` is difficult. No metric can fully represent the quality of an :doc:`explainer <explainers>`. While astrapia provides a few 
metrics to evaluate on, new :doc:`explainers <explainers>` might require a new set of metrics. While you can define them for each :doc:`explainer <explainers>` individually, 
:doc:`transfer functions <transfer>` allow you to define them indirectly. 

To add new :doc:`transfer functions <transfer>`, visit the ``astrapia/transfer_functions.py`` file. 
Add your new indirect metrics to the ``generate_default_transfer_functions`` method.

The following would be an example of adding the *area_absolute_log* metric defined in the :doc:`transfer` document as a general metric.

.. code-block:: python

    # astrapia/transfer_functions.py

    import math # import math for the log function

    def generate_default_transfer_functions(add_transfer):

        # [...] other transfer functions

        # define a new metric depending on the *area_absolute* metric
        def area_absolute_log(area_absolute):
            return math.log(area_absolute()) # remember, metrics are functions (not values)
        add_transfer(area_norm) # add the metric to the global transfer module
