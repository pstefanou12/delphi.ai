
**********************************************************
delphi: A Library for Robust Statistics and Machine Learning
**********************************************************

To use the library, clone repository and then install dependencies using 

.. code-block:: bash

   pip install --user -r requirements.txt


**********************************************************
Efficient Truncated Regression with Unknown Noise Variance
**********************************************************
.. ---------------------------------------
.. ICML 2021 Supplementary Code Submission
.. ---------------------------------------

.. Introduction
.. ============

.. This repository contains three jupyter notebooks to run the experiments mentioned
.. in the paper. These notebooks are all located within the ``/notebooks/`` directory.

.. The jupyter notebook called ``Varying C.ipynb`` contains the code for the 10D regression on synthetic data for which we adjust a 
.. truncation threshold C.

.. The jupyter notebook called  ``Union of Interval Truncation Regression.ipynb``, contains the code for truncated 
.. regression with a union of interval truncation set. Instructions for running the experiment are located within the notebook.

.. The jupyter notebook called ``Istanbul Stock Exchange Data Experiment.ipynb``,
.. contains the code for the semi-synthetic data experiment. For this experiment, we use the 
.. `ISTANBUL STOCK EXCHANGE Data Set <https://archive.ics.uci.edu/ml/datasets/ISTANBUL+STOCK+EXCHANGE>`_, from the `UCI 
.. Machine Learning Repository <https://archive.ics.uci.edu/ml/index.php>`_. Note that you will need to download 
.. the data in order to run the notebook. Read through notebook to see adjust hyperparameters, and provide user-specific inputs.

.. Download
.. --------

.. To run these notebooks, you will need to download all of the files from this anonymous repository.


.. Dependency Setup
.. ----------------

.. Here are step-by-step instructions to install the correct dependencies for running the experiments. 

.. 1. Create a virtual environment for this repository.

.. .. code-block:: bash

..    python3 -m virtualenv /path/to/environment/trunc-reg 


.. 2. Activate virtual environment. 

.. .. code-block:: bash

..    source /path/to/environment/trunc-reg/bin/activate


.. 3. Install a jupyter kernel. This will install a kernel inside the environment, to use to run in the Jupyter notebook there.

.. .. code-block:: bash

..    ipython kernel install --user --name=trunc-reg 

.. 4. Install packages in root directory of repository.

.. code-block:: bash

   pip install -r requirements.txt


.. Data
.. ----

.. You will need to download the `ISTANBUL STOCK EXCHANGE Data Set <https://archive.ics.uci.edu/ml/datasets/ISTANBUL+STOCK+EXCHANGE>`_
.. from the UCI repository to run the ``Istanbul Stock Exchange Data Experiment.ipynb``.

.. Note
.. ----

.. There will be additional instructions within the notebooks on how to run experiments.

.. Additional parameters can be changed by adjusting the ``args`` hyperparameters object
.. in the ``delphi/stats/truncated_regression.py`` file (hyperparameters that can be changed include
.. lr drop frequency, momentum, and weight-decay). 








