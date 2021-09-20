Mathematical Background
###########################

To allow for comparing explainers, we first need to 
define a common ground on the respective metrics. Any new 
metrics should be defined by extending upon this document.

Definitions
*************

:math:`X`      
    the input data.

:math:`x`
    is a single instance from the input data, :math:`x\in X` holds.

:math:`\mathbb{D}`
    is the domain of the input data. As such, :math:`X \subseteq \mathbb{D}` holds.

:math:`t(x)`
    denotes the true target label of :math:`x`. 
    As Astrapia is limited to binary classification, :math:`f(x)\in \{0,1\}` holds.

:math:`y_t(x)`
    denotes the predicted label for the instance :math:`x` where :math:`t` is either *model* or *explainer*. 
    Not every explainer is able to do prediction of their own. As such :math:`y_{explainer}(x)` is undefined for them.
    As the model might return probabilities, :math:`y_t(x) \in [0,1]` and not just :math:`\{0,1\}`.

:math:`\hat y_t(x)`
    is defined as 

    .. math::
        \hat y_t(x) =
            \left\{
                \begin{array}{ll}
                    0  & \mbox{if } y_t(x) \geq 0.5 \\
                    1 & \mbox{if } y_t(x) < 0.5
                \end{array}
            \right.

    while :math:`y_t(x)` represents a probabiliy distribution, 
    :math:`\hat y_t(x)` represents the most likely label.

:math:`D_i`
    is the domain of the i'th feature of :math:`D`

:math:`N`
    is the dimensionality of D. 
    
    As such the following should hold

    .. math::
        \mathbb{D} = \Pi_{i=1}^N D_i


Weight Functions
*******************
Local explainers consider a local area around an instance 
for their explanations. This area will be henceforth referred 
to as the explainers *neighbourhood*. To represent such neighbourhoods 
different shapes, sizes and densities, a weight 
function :math:`w_{e,i}(x)` is introduced for every 
explainer :math:`e` and explained instance :math:`i`. 
They are defined such that for any datapoint :math:`x` 

.. math::
    w_{e,i}(x) \in [0, 1]

As such, :math:`w_{e,i}(x)` represents how much of 
instance :math:`x` is inside the explainers neighbourhood 
centered around instance :math:`i`.

Example Weight Functions
---------------------------
To further clarify how a weight functions might be defined 
for your explainer, the following section lists a few 
example weight functions.

A weight function including every instance in :math:`\mathbb{D}`
    .. math::
        w_{{e_1},i}(x) := 1

A weight function including only the center instance
    .. math::
        w_{{e_2},i}(x) :=
            \left\{
                \begin{array}{ll}
                    1  & \mbox{if } x = i \\
                    0  & \mbox{otherwise } 
                \end{array}
            \right.

A weight function including samples within a circle around the center instance
    .. math::
        w_{{e_3},i}(x) :=
            \left\{
                \begin{array}{ll}
                    1  & \mbox{if } ||x - i||_2 \leq 1 \\
                    0  & \mbox{otherwise } 
                \end{array}
            \right.

A weight function representing an exponential kernel around the center instance
    .. math::
        w_{{e_4},i}(x) := e^{-||x - i||_2^2}