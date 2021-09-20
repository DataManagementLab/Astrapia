Metric and Property Reference
=======================================
The following metrics are supported by default. 
By following them, you can be sure that pre-implemented 
Astrapia explainers include your metrics. You may extend them as needed. 
For additional info about mathematical definitions, view the :doc:`math`.


.. py:function:: accuracy(self)

    Computes accuracy. The accuracy is defined as 

    .. math:: 
        
        \text{accuracy} := \frac{1}{n} \sum_{x\in X} 1 - |\hat y_{model}(x)-\hat y_{explainer}(x)|

.. py:function:: coverage_absolute(self)

    Computes absolute coverage. An intuitive description would be the number of samples within the explainers neighbourhood

    .. math::

        \text{coverage_absolute} := \sum_{x \in \mathbb{X}} w_{e,i}(x)


.. py:function:: coverage(self)

    Computes coverage. The coverage is defined as 

    .. math:: 
        
        \text{coverage} := \frac{\text{coverage_absolute}}{|X|}


.. py:function:: balance(self)

    Computes balance. The balance is defined as 

    .. math:: 
        
        \text{accuracy} := \frac{1}{n} \sum_{i=1}^{n} \text{true_positives}(i)


.. py:function:: area_absolute(self)

    Computes absolute area. Absolute area differs from relative area in that it represents the absolute size of an explainers kernel. The absolute area is defined as 
    
    .. math:: 
            
        \text{area_absolute} := \int_\mathbb{D} w_{e,i}(x) dx


.. py:function:: area_relative(self)

    Computes relative area. It differs from absolute area in that it computes the size of the explainers kernel as a fraction of the total input space. The relative area is defined as 

    .. math:: 
        
        \text{area_relative} := 

.. py:function:: area_normalized(self)

    Computes normalized area. The normalized area normalized the absolute area over the number of used dimensions. This is needed as the area of isotropic kernels grows exponentially with the number of dimensions. The normalized area is defined as 

    .. math:: 
        
        \text{accuracy} := {\text{area_absolute} \over N}

