*******************
Explainers
*******************
<Todo write what explainers are>

Preimplemented Explainers
======================================
* LimeExplainer
* AnchorsExplainer

Metrics, Properties and Utilitites
======================================
Explain what Metrics, properties and Utilities are

Metrics
---------------
Explain what metrics are

Properties
---------------
Explain what properties are

Utilities
---------------
Explain what utilities are

Translation
=================
<Explain why explainers need translation>

.. py:function:: translate_dataset(self, data, meta)

    Translates an Astrapia dataset into an explainer specific format

.. py:function:: inverse_translate_dataset(self, data, meta)

    Translates a dataset into the astrapia dataset format.


Reference
=======================================
The following metrics are supported by default. You may extend them as needed.

.. py:function:: accuracy(self)

    Computes accuracy. The accuracy is defined as 

    .. math:: 
        
        \text{accuracy} := \frac{1}{n} \sum_{i=1}^{n} \text{true_positives}(i)


.. py:function:: coverage(self)

    Computes coverage. The coverage is defined as 

    .. math:: 
        
        \text{accuracy} := \frac{1}{n} \sum_{i=1}^{n} \text{true_positives}(i)


.. py:function:: balance(self)

    Computes balance. The balance is defined as 

    .. math:: 
        
        \text{accuracy} := \frac{1}{n} \sum_{i=1}^{n} \text{true_positives}(i)


.. py:function:: absolute_area(self)

    Computes absolute area. Absolute area differs from relative area in that it represents the absolute size of an explainers kernel. The absolute area is defined as 
    
    .. math:: 
            
        \text{accuracy} := \frac{1}{n} \sum_{i=1}^{n} \text{true_positives}(i)


.. py:function:: relative_area(self)

    Computes relative area. It differs from absolute area in that it computes the size of the explainers kernel as a fraction of the total input space. The relative area is defined as 

    .. math:: 
        
        \text{accuracy} := \frac{1}{n} \sum_{i=1}^{n} \text{true_positives}(i)

.. py:function:: area_normalized(self)

    Computes normalized area. The normalized area normalized the absolute area over the number of used dimensions. This is needed as the area of isotropic kernels grows exponentially with the number of dimensions. The normalized area is defined as 

    .. math:: 
        
        \text{accuracy} := \frac{1}{n} \sum_{i=1}^{n} \text{true_positives}(i)

