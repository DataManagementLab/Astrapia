*******************
Explainers
*******************
Explainers are used to explain the behavour of an arbitrary machine learning model.

.. autoclass:: astrapia.Explainer
    :members: metrics, props, report, explain_instance

    .. method:: infer_metrics(printing=True)

        Uses :doc:`transfer` to infer additional metrics

        :param printing: boolean value on whether to print the inferred metrics



Preimplemented Explainers
======================================

.. toctree::
    :maxdepth: 2
 
    explainers/lime
    explainers/dlime
    explainers/anchors


 

Metrics, Properties and Utilitites
======================================
Metrics_, properties_ and utilities_ are the three types of optionally defined functions.
When declaring a new explainer, you should specify as many of them as possible. 
Should two explainers share a metric or property, they will be compared by Astrapia

Metrics
---------------
Metrics such as *accuracy*, *coverage* and *area* are used to compare 
different explainers. Each metric is a function that takes no parameters beyond the explainer itself.
Each metric also need to be prepended with the *metric* decorator.

The following might be an example of a metric:

.. code-block:: python

    @astrapia.metric
    def global_accuracy(self):
        return sum(self.preds == self.target_labels) / len(self.target_labels)

Properties
---------------
While metrics may depend on the current state of the explainer, properties are static.
They represent properties of the explainer as a whole. 
Examples of properties are *name* and *neighborhood_shape*.

They are implement equivalently to `metrics`_ but require the *prop* decorator. The following might be an example of a property:

.. code-block:: python

    @astrapia.prop
    def name(self):
        return 'Lime'

Utilities
---------------
While metrics_ and properties_ are used to compare explainers, 
utilities are used to automatically infer metrics. For example, 
given a function to weight samples, Astrapia can infer the *coverage* metric.

Utilities require the *utility* decorator. 
They can also have parameters other than the explainer itself.
An example of a utility is:

.. code-block:: python

    @astrapia.utility
    def distance(self, x, y):
        return np.linalg.norm(x - y)

Translation
=================
Different explainers use different dataset formats. 
While the LimeExplainer uses pandas DataFrames, the AnchorsExplainer uses numpy arrays.
To allow different explainers to be compared on the same models, 
Astrapia introduces a translation layer. 

The translation layer is responsible for
- converting a general dataset into an explainer specific dataset and
- converting and explainer specific dataset or instance into a general format

While the following functions are never used internally, 
it is recommended to implement them as they are needed at many points in time.

The meta parameter is a reference to the dataset object the explainer is initalized with. 
While this object will not hold the data to be translated, 
the meta information (such as a list of categorical features) can be very usefull.

.. py:function:: translate_dataset(self, data, meta)

    Translates an Astrapia dataset into an explainer specific format

.. py:function:: inverse_translate_dataset(self, data, meta)

    Translates a dataset into the astrapia dataset format.


