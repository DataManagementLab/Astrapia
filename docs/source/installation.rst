
Installation
==================


Start by cloning the repository and move to the project folder.

.. code-block:: bash

   $ git clone git@github.com:DataManagementLab/Astrapia.git && cd Astrapia

Run the following command to install necessary dependencies. A symbolic link will be built to *astrapia* allowing you to change the source code without reinstallation.

.. code-block:: bash

    $ pip install -r requirements.txt

To fetch the ``adult`` dataset, navigate into ``data/adult/`` and run

.. code-block:: bash

    $ python setup_adult.py
