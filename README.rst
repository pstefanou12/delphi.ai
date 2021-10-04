
**********************************************************
Efficient Truncated Regression with Unknown Noise Variance - NeurIPs - 2021
**********************************************************

To use the library, clone repository and then install dependencies using 

.. code-block:: bash

   pip install --user -r requirements.txt

This repository contains code for the experiments run in the paper Efficient Truncated Regression 
with Unknown Noise Variance that was submitted to NeurIPs 2021. 

The notebooks for all of the experiments that we run can be located in the 
efficient-truncated-regression-with-unknown-noise-variance/notebooks/neurips directory of the repository. Within 
this directory there are five notebooks and one R script. The R script is for running experiments 
on the `truncreg <https://www.rdocumentation.org/packages/truncreg/versions/0.2-5/topics/truncreg>`_ R package. 

All five of the notebooks should run once all of the dependencies are installed. 

We perform one semi-synthetic experiment on the PM10 dataset. We have included the csv file for this dataset in the notebooks folder.
Nevertheless, the dataset can be found online, `here <http://lib.stat.cmu.edu/datasets/PM10.dat>`_.

Within each notebook, the only thing that will need to be changed is the out_dir and exp keys that are located in the args dictionary
of the second cell of each notebook. The out_dir key signifies where the experiment results will be logged in. The exp key 
is the name of the experiment that you are currently running.