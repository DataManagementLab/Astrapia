Transfer
#######################

Often, metrics are derived from other metrics. 
They may represent the same value, but normalized or 
rescaled to allow for more general comparison. 

As such derived functions are not dependant on the explainer, 
but rather on the already defined metrics, you should not need to 
implement them for each explainer but only once for all of them.

To solve this, Astrapia provides the Transfer_ module. 
By defining a new metric in terms of other metrics, the new metric 
can be applied to all explainers

Adding transfer functions
**************************

In this example we will implement the *area_absolute_log* metric. 
As absolute area tends to grow exponentially with the number of 
dimensions the explainer considers, 
comparing them in the log domain is visually much more appealing.

.. code-block:: python

    from astrapia import transfer # import the transfer module
    import math # import math for the log function

    # define a new metric depending on the *area_absolute* metric
    def area_absolute_log(area_absolute):
        return math.log(area_absolute()) # remember, metrics are functions (not values)

    add_transfer(area_norm) # add the metric to the global transfer module


Now every explainer that has defined the *area_absolute* metric 
will be able to infer the *area_absolute_log* metric.

Utilizing transfer functions
******************************

To save on unnecessary computation time, 
explainers by default do not infer any metrics. 
Inferring metrics needs to be explicitly started.

.. code-block:: python

    exp = LimeExplainer(...) # instantiate an explainer
    exp.infer_metrics() # apply derived metrics on the explainer

