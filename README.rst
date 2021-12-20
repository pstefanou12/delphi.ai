delphi.ai package
=================
Install via ``pip``: ``pip install delphi.ai``

This library holds a collection of algorithms that can be used 
debias models that have been defected due to truncation, or missing data. A few 
projects using the library can found in: 
  
* `Code for Efficient Truncated Linear Regression with Unknown Noise Variance <https://github.com/pstefanou12/Truncated-Regression-With-Unknown-Noise-Variance-NeurIPS-2021>`_

We demonstrate how to use the library in a set of walkthroughs and our API
reference. Functionality provided by the library includes:

For best results using the package, the data should have mean 0 and variance 1.

Before running PSGD, the library will check that all of the required 
arguments are provided for runnning the procedure with an internal function. After this, all other hyperparameters can be provided by the user, or their defaults values will be used. The current 
default hyperparameters can be seen by looking at the ``delphi.utils.defaults.py`` file.

For logging experiment information, we use MadryLab's `cox <https://github.com/MadryLab/cox>`_. More information and tutorials on how to use the logging framework, check out the link.

Contents:
--------

* `distributions <#distributions>`__: ``distributions`` module includes algorithms for learning from censored (known truncation) and truncated (unknown truncation; unsupervised learning) distributions

  * `CensoredNormal <#CensoredNormal>`__
  * `CensoredMultivariateNormal <#CensoredMultivariateNormal>`__
  * `TruncatedNormal <#TruncatedNormal>`__
  * `TruncatedMultivariateNormal <#TruncatedMultivariateNormal>`__
  * `TruncatedBernoulli <#TruncatedBernoulli>`__

* `stats <#stats>`__ : ``stats`` module includes models for regression and classification from truncated samples
 
  * `TruncatedLinearRegression <#TruncatedLinearRegression>`__
  * `TruncatedLassoRegression <#TruncatedLassoRegression>`__
  * `TruncatedRidgeRegression <#TruncatedRidgeRegression>`__
  * `TruncatedElasticNetRegression <#TruncatedElasticNetRegression>`__
  * `TruncatedLogisticRegression <#TruncatedLogisticRegression>`__
  * `TruncatedProbitRegression <#TruncatedProbitRegression>`__

distributions
=============

CensoredNormal:
---------------
``CensoredNormal`` learns censored normal distributions, by maximizing the truncated log likelihood.
The algorithm that we use for this procedure is described in the following
paper `Efficient Statistics in High Dimensions from Truncated Samples <https://arxiv.org/abs/1809.03986>`_.

When evaluating censored normal distributions, the user needs three things; an oracle, a Callable that 
indicates whether a sample falls within the truncation set, the model's ``alpha``, survival probability, and the ``CensoredNormal`` module. The ``CensoredNormal`` module accepts 
a parameters object that the user can define for running the PSGD procedure.

Parameters:
-----------

* ``args`` (delphi.utils.Parameters): parameters object that holds hyperparameters for experiment. Possible hyperparameters include:

  * ``phi`` (Callable)): required argument; callable class that receives num_samples by 1 input ``torch.Tensor``, and returns a num_samples by 1 outputs a num_samples by 1 ``Tensor`` with ``(0, 1)`` representing membership in ``S`` or not.
  * ``alpha`` (float): required argument; survivial probability for truncated regression
  * ``variance`` (float): provide distribution's variance, if the distribution's variance is given, the mean is exclusively calculated 
  * ``epochs`` (int): maximum number of times to iterate over dataset
  * ``trials`` (int): maximum number of trials to perform PSGD; after trials, model with smallest loss on the dataset is returned
  * ``val`` (float): percentage of dataset to use for validation set; default .2
  * ``lr`` (float): initial learning rate to use for regression weights; default 1e-1
  * ``step_lr`` (int): number of gradient steps to take before adjusting learning rate by value ``step_lr_gamma``; default 100
  * ``step_lr_gamma`` (float): amount to adjust learning rate, every ``step_lr`` steps ``new_lr = curr_lr * step_lr_gamma``
  * ``custom_lr_multiplier`` (str): `cosine` or `cyclic` for cosine annealing learning rate scheduling or cyclic learning rate scheduling; default None
  * ``momentum`` (float): momentum; default 0.0 
  * ``adam`` (bool): use adam adaptive learning rate optimizer; default False
  * ``eps`` (float): epsilon denominator for gradients (ie. to prevent divide by zero calculations); default 1e-5
  * ``r`` (float): initial projection set radius; default 1.0
  * ``rate`` (float): at the end of each trial, the projection set radius is increased at rate `rate`; default 1.5
  * ``batch_size`` (int): the number of samples to use for each gradient step; default 50
  * ``tol`` (float): if using early stopping, threshold for when to stop; default 1e-3
  * ``workers`` (int): number of workers to use for procedure; default 1
  * ``num_samples`` (int): number of samples to sample from distribution in gradient for each sample in batch (ie. if batch size is 10, and num_samples is 100, the each gradient step with sample 100 * 10 samples from a gaussian distribution); default 50
  * ``early_stopping`` (bool): whether to check loss for convergence; compares the best avg validation loss at the end of an epoch, with current avg epoch loss estimate, if :math:`best_loss - curr_loss < tol` for `n_iter_no_change`, then procedure terminates; default False
  * ``n_iter_no_change`` (int): number of iterations to check for change before declaring convergence; default 5
  * ``verbose`` (bool): whether to print a verbose output with loss logs, etc.; default False 

* ``store`` (cox.store.Store): logging object to keep track distribution's train and validation losses   

Attributes:
~~~~~~~~~~~

* ``loc_`` (torch.Tensor): distribution's estimated mean 
* ``variance_`` (torch.Tensor): distribution's estimated variance 

In the following code block, here, we show an example of how to use the censored normal distribution module: 
   
.. code-block:: python

  from delphi.distributions.censored_normal import CensoredNormal
  from delphi import oracle
  from delphi.utils.helpers import Parameters
  from cox.store import Store

  OUT_DIR = 'PATH_TO_EXPERIMENT_LOGGING_DIRECTORY'
  store = Store(OUT_DIR)

  # left truncate 0 (ie. S = {x >= 0 for all x in S})
  phi = oracle.Left_Distribution(0.0)
  # pass algorithm parameters in through Parameters object
  train_kwargs = Parameters({'phi': phi, 
                              'alpha': alpha})
  # define censored normal distribution object
  censored = CensoredNormal(train_kwargs, store=store)
  # fit to dataset
  censored.fit(S)
  # close store 
  store.close()

CensoredMultivariateNormal:
--------------------------
``CensoredMultivariateNormal`` learns censored multivariate normal distributions, by maximizing the truncated log likelihood.
The algorithm that we use for this procedure is described in the following
paper `Efficient Statistics in High Dimensions from Truncated Samples <https://arxiv.org/abs/1809.03986>`_.

When evaluating censored multivariate normal distributions, the user needs three things; an oracle, a Callable that 
indicates whether a sample falls within the truncation set, the model's ``alpha``, survival probability, and the ``CensoredMultivariateNormal`` module. The ``CensoredMultivariateNormal`` module accepts 
a parameters object that the user can define for running the PSGD procedure.

Parameters:
-----------

* ``args`` (delphi.utils.Parameters): parameters object that holds hyperparameters for experiment. Possible hyperparameters include:

  * ``phi`` (Callable): required argument; callable class that receives num_samples by 1 input ``torch.Tensor``, and returns a num_samples by 1 outputs a num_samples by 1 ``Tensor`` with ``(0, 1)`` representing membership in ``S`` or not.
  * ``alpha`` (float): required argument; survivial probability for truncated regression
  * ``covariance_matrix`` (torch.Tensor): provide distribution's covariance_matrix, if the distribution's covariance_matrix is given, the mean vector is exclusively calculated 
  * ``epochs`` (int): maximum number of times to iterate over dataset
  * ``trials`` (int): maximum number of trials to perform PSGD; after trials, model with smallest loss on the dataset is returned
  * ``val`` (float): percentage of dataset to use for validation set; default .2
  * ``lr`` (float): initial learning rate to use for regression weights; default 1e-1
  * ``step_lr`` (int): number of gradient steps to take before adjusting learning rate by value ``step_lr_gamma``; default 100
  * ``step_lr_gamma`` (float): amount to adjust learning rate, every ``step_lr`` steps ``new_lr = curr_lr * step_lr_gamma``
  * ``custom_lr_multiplier`` (str): `cosine` or `cyclic` for cosine annealing learning rate scheduling or cyclic learning rate scheduling; default None
  * ``momentum`` (float): momentum; default 0.0 
  * ``adam`` (bool): use adam adaptive learning rate optimizer; default False
  * ``eps`` (float): epsilon denominator for gradients (ie. to prevent divide by zero calculations); default 1e-5
  * ``r`` (float): initial projection set radius; default 1.0
  * ``rate`` (float): at the end of each trial, the projection set radius is increased at rate `rate`; default 1.5
  * ``batch_size`` (int): the number of samples to use for each gradient step; default 50
  * ``tol`` (float): if using early stopping, threshold for when to stop; default 1e-3
  * ``workers`` (int): number of workers to use for procedure; default 1
  * ``num_samples`` (int): number of samples to sample from distribution in gradient for each sample in batch (ie. if batch size is 10, and num_samples is 100, the each gradient step with sample 100 * 10 samples from a gaussian distribution); default 50
  * ``early_stopping`` (bool): whether to check loss for convergence; compares the best avg validation loss at the end of an epoch, with current avg epoch loss estimate, if :math:`best_loss - curr_loss < tol` for `n_iter_no_change`, then procedure terminates; default False
  * ``n_iter_no_change`` (int): number of iterations to check for change before declaring convergence; default 5
  * ``verbose`` (bool): whether to print a verbose output with loss logs, etc.; default False 

* ``store`` (cox.store.Store): logging object to keep track distribution's train and validation losses   

Attributes:
~~~~~~~~~~~

* ``loc_`` (torch.Tensor): distribution's estimated mean 
* ``covariance_matrix_`` (torch.Tensor): distribution's estimated covariance matrix 

In the following code block, here, we show an example of how to use the censored multivariate normal distribution module: 
   
.. code-block:: python

  from torch import Tensor
  from delphi.distributions.censored_multivariate_normal import CensoredMultivariateNormal
  from delphi import oracle
  from delphi.utils.helpers import Parameters
  from cox.store import Store

  OUT_DIR = 'PATH_TO_EXPERIMENT_LOGGING_DIRECTORY'
  store = Store(OUT_DIR)

  # left truncate 0 (ie. S = {x >= 0 for all x in S})
  phi = oracle.Left_Distribution(Tensor([0.0, 0.0]))
  # pass algorithm parameters in through Parameters object
  train_kwargs = Parameters({'phi': phi, 
                              'alpha': alpha})
  # define censored multivariate normal distribution object
  censored = CensoredMultivariateNormal(train_kwargs, store=store)
  # fit to dataset
  censored.fit(S)
  # close store 
  store.close()

TruncatedNormal:
--------------------------
``TruncatedNormal`` learns truncated normal distributions, with unknown truncation, by maximizing the truncated log likelihood.
The algorithm that we use for this procedure is described in the following
paper `Efficient Truncated Statistics with Unknown Truncation <https://arxiv.org/abs/1908.01034>`_.

When evaluating truncated normal distributions, the user needs to ``import`` the ``TruncatedNormal`` module. The ``TruncatedNormal`` module accepts 
a parameters object that the user can define for running the PSGD procedure. When *debiasing* truncated normal distributions, we don't require a membership 
oracle, as it is unknown. However, after running our procedure, we are able to provide an approximation of what the truncation set is. Since the user 
inputs a membership oracle in the ``args`` object, when the truncation set is known, we add the learned membership oracle to the ``args`` object as well.

**NOTE:** when learning truncation sets, the user can not pass in a ``Parameters`` object directly into the ``TruncatedNormal`` object, because they will not 
be able to access the ``Parameters`` object afterwards.

Parameters:
-----------

* ``args`` (delphi.utils.Parameters): parameters object that holds hyperparameters for experiment. Possible hyperparameters include:

  * ``alpha`` (float): required argument; survivial probability for truncated regression
  * ``covariance_matrix`` (torch.Tensor): provide distribution's covariance_matrix, if the distribution's covariance_matrix is given, the mean vector is exclusively calculated 
  * ``epochs`` (int): maximum number of times to iterate over dataset
  * ``trials`` (int): maximum number of trials to perform PSGD; after trials, model with smallest loss on the dataset is returned
  * ``val`` (float): percentage of dataset to use for validation set; default .2
  * ``lr`` (float): initial learning rate to use for regression weights; default 1e-1
  * ``step_lr`` (int): number of gradient steps to take before adjusting learning rate by value ``step_lr_gamma``; default 100
  * ``step_lr_gamma`` (float): amount to adjust learning rate, every ``step_lr`` steps ``new_lr = curr_lr * step_lr_gamma``
  * ``custom_lr_multiplier`` (str): `cosine` or `cyclic` for cosine annealing learning rate scheduling or cyclic learning rate scheduling; default None
  * ``momentum`` (float): momentum; default 0.0 
  * ``adam`` (bool): use adam adaptive learning rate optimizer; default False
  * ``eps`` (float): epsilon denominator for gradients (ie. to prevent divide by zero calculations); default 1e-5
  * ``r`` (float): initial projection set radius; default 1.0
  * ``rate`` (float): at the end of each trial, the projection set radius is increased at rate `rate`; default 1.5
  * ``batch_size`` (int): the number of samples to use for each gradient step; default 50
  * ``tol`` (float): if using early stopping, threshold for when to stop; default 1e-3
  * ``workers`` (int): number of workers to use for procedure; default 1
  * ``num_samples`` (int): number of samples to sample from distribution in gradient for each sample in batch (ie. if batch size is 10, and num_samples is 100, the each gradient step with sample 100 * 10 samples from a gaussian distribution); default 50
  * ``early_stopping`` (bool): whether to check loss for convergence; compares the best avg validation loss at the end of an epoch, with current avg epoch loss estimate, if :math:`best_loss - curr_loss < tol` for `n_iter_no_change`, then procedure terminates; default False
  * ``n_iter_no_change`` (int): number of iterations to check for change before declaring convergence; default 5
  * ``verbose`` (bool): whether to print a verbose output with loss logs, etc.; default False 
  * ``d`` (int): degree of expansion to use for Hermite polynomial when learning truncation set; default 100

* ``store`` (cox.store.Store): logging object to keep track distribution's train and validation losses   

Attributes:
~~~~~~~~~~~

* ``loc_`` (torch.Tensor): distribution's estimated mean 
* ``variance_`` (torch.Tensor): distribution's estimated variance 

In the following code block, here, we show an example of how to fit the truncated normal distribution module: 
   
.. code-block:: python

  from delphi.distributions.truncated_normal import TruncatedNormal
  from delphi import oracle
  from delphi.utils.helpers import Parameters
  from cox.store import Store

  OUT_DIR = 'PATH_TO_EXPERIMENT_LOGGING_DIRECTORY'
  store = Store(OUT_DIR)

  # left truncate 0 (ie. S = {x >= 0 for all x in S})
  phi = oracle.Left_Distribution(0.0)
  # pass algorithm parameters in through Parameters object
  train_kwargs = Parameters({'phi': phi, 
                              'alpha': alpha, 
                              'd': 100})
  # define truncated normal distribution object
  truncated = TruncatedNormal(train_kwargs, store=store)
  # fit to dataset
  truncated.fit(S)
  # close store 
  store.close()

After fitting the distribution, we now have a membership oracle that we learned through a hermite polynomial. In the following code block, 
we show an example of how use the membership oracle: 

.. code-block:: python

  import torch as ch
  from torch.distributions.multivariate_normal import MultivariateNormal 

  # generate samples from a standard multivariate normal distribution
  M = MultivariateNormal(ch.zeros(1,), ch.eye(1))
  samples = M.rsample([1000,])
  # filter samples with learning membership oracle
  filtered = train_kwargs.phi(samples)

TruncatedMultivariateNormal:
--------------------------
``TruncatedMultivariateNormal`` learns truncated multivariate normal distributions, with unknown truncation, by maximizing the truncated log likelihood.
The algorithm that we use for this procedure is described in the following
paper `Efficient Truncated Statistics with Unknown Truncation <https://arxiv.org/abs/1908.01034>`_.

When evaluating truncated multivariate normal distributions, the user needs to ``import`` the ``TruncatedMultivariateNormal`` module. The ``TruncatedMultivariateNormal`` module accepts 
a parameters object that the user can define for running the PSGD procedure. When *debiasing* truncated normal distributions, we don't require a membership 
oracle, as it is unknown. However, after running our procedure, we are able to provide an approximation of what the truncation set is. Since the user 
inputs a membership oracle in the ``args`` object, when the truncation set is known, we add the learned membership oracle to the ``args`` object as well.


**NOTE:** when learning truncation sets, the user can not pass in a ``Parameters`` object directly into the ``TruncatedMultivariateNormal`` object, because they will not 
be able to access the ``Parameters`` object afterwards.

Parameters:
-----------

* ``args`` (delphi.utils.Parameters): parameters object that holds hyperparameters for experiment. Possible hyperparameters include:

  * ``phi`` (Callable): required argument; callable class that receives num_samples by 1 input ``torch.Tensor``, and returns a num_samples by 1 outputs a num_samples by 1 ``Tensor`` with ``(0, 1)`` representing membership in ``S`` or not.
  * ``alpha`` (float): required argument; survivial probability for truncated regression
  * ``variance`` (float): provide distribution's variance, if the distribution's variance is given, the mean is exclusively calculated 
  * ``epochs`` (int): maximum number of times to iterate over dataset
  * ``trials`` (int): maximum number of trials to perform PSGD; after trials, model with smallest loss on the dataset is returned
  * ``val`` (float): percentage of dataset to use for validation set; default .2
  * ``lr`` (float): initial learning rate to use for regression weights; default 1e-1
  * ``step_lr`` (int): number of gradient steps to take before adjusting learning rate by value ``step_lr_gamma``; default 100
  * ``step_lr_gamma`` (float): amount to adjust learning rate, every ``step_lr`` steps ``new_lr = curr_lr * step_lr_gamma``
  * ``custom_lr_multiplier`` (str): `cosine` or `cyclic` for cosine annealing learning rate scheduling or cyclic learning rate scheduling; default None
  * ``momentum`` (float): momentum; default 0.0 
  * ``adam`` (bool): use adam adaptive learning rate optimizer; default False
  * ``eps`` (float): epsilon denominator for gradients (ie. to prevent divide by zero calculations); default 1e-5
  * ``r`` (float): initial projection set radius; default 1.0
  * ``rate`` (float): at the end of each trial, the projection set radius is increased at rate `rate`; default 1.5
  * ``batch_size`` (int): the number of samples to use for each gradient step; default 50
  * ``tol`` (float): if using early stopping, threshold for when to stop; default 1e-3
  * ``workers`` (int): number of workers to use for procedure; default 1
  * ``num_samples`` (int): number of samples to sample from distribution in gradient for each sample in batch (ie. if batch size is 10, and num_samples is 100, the each gradient step with sample 100 * 10 samples from a gaussian distribution); default 50
  * ``early_stopping`` (bool): whether to check loss for convergence; compares the best avg validation loss at the end of an epoch, with current avg epoch loss estimate, if :math:`best_loss - curr_loss < tol` for `n_iter_no_change`, then procedure terminates; default False
  * ``n_iter_no_change`` (int): number of iterations to check for change before declaring convergence; default 5
  * ``verbose`` (bool): whether to print a verbose output with loss logs, etc.; default False 
  * ``d`` (int): degree of expansion to use for Hermite polynomial when learning truncation set; default 100

* ``store`` (cox.store.Store): logging object to keep track distribution's train and validation losses   

Attributes:
~~~~~~~~~~~

* ``loc_`` (torch.Tensor): distribution's estimated mean 
* ``covariance_matrix_`` (torch.Tensor): distribution's estimated covariance matrix 

In the following code block, here, we show an example of how to use the truncated multivariate normal distribution module: 
   
.. code-block:: python

  from torch import Tensor
  from delphi.distributions.truncated_multivariate_normal import TruncatedMultivariateNormal
  from delphi.utils.helpers import Parameters
  from delphi import oracle
  from cox.store import Store

  OUT_DIR = 'PATH_TO_EXPERIMENT_LOGGING_DIRECTORY'
  store = Store(OUT_DIR)

  # left truncate 0 (ie. S = {x >= 0 for all x in S})
  phi = oracle.Left_Distribution(Tensor([0.0, 0.0]))
  # pass algorithm parameters in through Parameters object
  train_kwargs = Parameters({'phi': phi, 
                              'alpha': alpha, 
                              'd': 100})
  # define truncated multivariate normal distribution object
  truncated = TruncatedMultivariateNormal(train_kwargs, store=store)
  # fit to dataset
  truncated.fit(S)
  # close store 
  store.close()

After fitting the distribution, we now have a membership oracle that we learned through a hermite polynomial. In the following code block, 
we show an example of how use the membership oracle: 

.. code-block:: python

  import torch as ch
  from torch.distributions.multivariate_normal import MultivariateNormal 

  # generate samples from a standard multivariate normal distribution
  M = MultivariateNormal(ch.zeros(2,), ch.eye(2))
  samples = M.rsample([1000,])
  # filter samples with learning membership oracle
  filtered = train_kwargs.phi(samples)


TruncatedBernoullli:
--------------------
``TruncatedBooleanProduct`` learns truncated boolean product distributions, by maximizing the truncated log likelihood.
The algorithm that we use for this procedure is described in the following
paper `Efficient Parameter Estimation of Truncated Boolean Product Distributions <https://arxiv.org/abs/2007.02392>`_.

When evaluating truncated multivariate normal distributions, the user needs to ``import`` the ``TruncatedBernoulli`` module. The ``TruncatedBernoulli`` module accepts 
a parameters object that the user can define for running the PSGD procedure. 

Parameters:
-----------

* ``args`` (delphi.utils.Parameters): parameters object that holds hyperparameters for experiment. Possible hyperparameters include:

  * ``phi`` (Callable): required argument; callable class that receives num_samples by 1 input ``torch.Tensor``, and returns a num_samples by 1 outputs a num_samples by 1 ``Tensor`` with ``(0, 1)`` representing membership in ``S`` or not.
  * ``alpha`` (float): required argument; survivial probability for truncated regression
  * ``epochs`` (int): maximum number of times to iterate over dataset
  * ``trials`` (int): maximum number of trials to perform PSGD; after trials, model with smallest loss on the dataset is returned
  * ``val`` (float): percentage of dataset to use for validation set; default .2
  * ``lr`` (float): initial learning rate to use for regression weights; default 1e-1
  * ``step_lr`` (int): number of gradient steps to take before adjusting learning rate by value ``step_lr_gamma``; default 100
  * ``step_lr_gamma`` (float): amount to adjust learning rate, every ``step_lr`` steps ``new_lr = curr_lr * step_lr_gamma``
  * ``custom_lr_multiplier`` (str): `cosine` or `cyclic` for cosine annealing learning rate scheduling or cyclic learning rate scheduling; default None
  * ``momentum`` (float): momentum; default 0.0 
  * ``adam`` (bool): use adam adaptive learning rate optimizer; default False
  * ``eps`` (float): epsilon denominator for gradients (ie. to prevent divide by zero calculations); default 1e-5
  * ``r`` (float): initial projection set radius; default 1.0
  * ``rate`` (float): at the end of each trial, the projection set radius is increased at rate `rate`; default 1.5
  * ``batch_size`` (int): the number of samples to use for each gradient step; default 50
  * ``tol`` (float): if using early stopping, threshold for when to stop; default 1e-3
  * ``workers`` (int): number of workers to use for procedure; default 1
  * ``num_samples`` (int): number of samples to sample from distribution in gradient for each sample in batch (ie. if batch size is 10, and num_samples is 100, the each gradient step with sample 100 * 10 samples from a gaussian distribution); default 50
  * ``early_stopping`` (bool): whether to check loss for convergence; compares the best avg validation loss at the end of an epoch, with current avg epoch loss estimate, if :math:`best_loss - curr_loss < tol` for `n_iter_no_change`, then procedure terminates; default False
  * ``n_iter_no_change`` (int): number of iterations to check for change before declaring convergence; default 5
  * ``verbose`` (bool): whether to print a verbose output with loss logs, etc.; default False 

* ``store`` (cox.store.Store): logging object to keep track distribution's train and validation losses   

Attributes:
~~~~~~~~~~~

* ``probs_`` (torch.Tensor): distribution's d-dimensional probability vector 
* ``logits_`` (torch.Tensor): distribution's d-dimensional logits vector (log probabilities) 

In the following code block, here, we show an example of how to use the truncated multivariate normal distribution module: 
   
.. code-block:: python

  from torch import Tensor
  from delphi.distributions.truncated_boolean_product import TruncatedBernoulli
  from delphi.utils.helpers import Parameters
  from delphi import oracle
  from cox.store import Store

  OUT_DIR = 'PATH_TO_EXPERIMENT_LOGGING_DIRECTORY'
  store = Store(OUT_DIR)

  # sum floor truncate at 0 (ie. S = {x.sum() >= 50 for all x in S})
  phi = oracle.Sum_Floor(50)
  # pass algorithm parameters in through Parameters object
  train_kwargs = Parameters({'phi': phi, 
                              'alpha': alpha})
  # define truncated bernoulli distribution object
  trunc_bool = TruncatedBernoulli(train_kwargs, store=store)
  # fit to dataset
  trunc_bool.fit(S)
  # close store 
  store.close()

stats
=====

TruncatedLinearRegression:
--------------------------
``TruncatedLinearRegression`` learns from truncated linear regression model's with the noise 
variance is known or unknown. In the known setting we use the algorithm described in the following
paper: `Computationally and Statistically Efficient Truncated Regression <https://arxiv.org/abs/2010.12000>`_. When 
the variance of the ground-truth linear regression's model is unknown, we use the algorithm described in 
the following paper: `Efficient Truncated Linear Regression with Unknown Noise Variance`.

When evaluating truncated regression models, the user needs three things; an oracle, a Callable that 
indicates whether a sample falls within the truncation set, the model's ``alpha``, survival probability, and the ``TruncatedLinearRegression`` module.  The ``TruncatedLinearRegression`` module accepts 
a parameters object that the user can define for running the PSGD procedure.

Parameters:
~~~~~~~~~~

* ``args`` (delphi.utils.Parameters): parameters object that holds hyperparameters for experiment. Possible hyperparameters include:

  * ``phi`` (Callable): required argument; callable class that receives num_samples by 1 input ``torch.Tensor``, and returns a num_samples by 1 outputs a num_samples by 1 ``Tensor`` with ``(0, 1)`` representing membership in ``S`` or not.
  * ``alpha`` (float): required argument; survivial probability for truncated regression
  * ``epochs`` (int): maximum number of times to iterate over dataset
  * ``noise_var`` (float): provide noise variance, if the noise variance for the truncated regression model is known, else unknown variance procedure is run by default
  * ``fit_intercept`` (bool): whether to fit the intercept or not; default to True
  * ``trials`` (int): maximum number of trials to perform PSGD; after trials, model with smallest loss on the dataset is returned
  * ``val`` (float): percentage of dataset to use for validation set; default .2
  * ``lr`` (float): initial learning rate to use for regression weights; default 1e-1
  * ``var_lr`` (float): initial learning rate to use variance parameters, when running unknown variance 
  * ``step_lr`` (int): number of gradient steps to take before adjusting learning rate by value ``step_lr_gamma``; default 100
  * ``step_lr_gamma`` (float): amount to adjust learning rate, every ``step_lr`` steps ``new_lr = curr_lr * step_lr_gamma``
  * ``custom_lr_multiplier`` (str): `cosine` or `cyclic` for cosine annealing learning rate scheduling or cyclic learning rate scheduling; default None
  * ``momentum`` (float): momentum; default 0.0 
  * ``adam`` (bool): use adam adaptive learning rate optimizer; default False
  * ``eps`` (float): epsilon denominator for gradients (ie. to prevent divide by zero calculations); default 1e-5
  * ``r`` (float): initial projection set radius; default 1.0
  * ``rate`` (float): at the end of each trial, the projection set radius is increased at rate `rate`; default 1.5
  * ``normalize`` (bool): our methods assume that the :math:`max(||x_{i}||_{2}) <= 1`, so before running the procedure, you must  divide the input featurers :math:`X = {x_{(1)}, x_{(2)}, ... , x_{(n)}}` by :math:`\max(||x_{i}||_{2}) \dot \sqrt(k)`, where :math:`k` represents the number of dimensions the input features have; by default the procedure normalizes the features for the user
  * ``batch_size`` (int): the number of samples to use for each gradient step; default 50
  * ``tol`` (float): if using early stopping, threshold for when to stop; default 1e-3
  * ``workers`` (int): number of workers to use for procedure; default 1
  * ``num_samples`` (int): number of samples to sample from distribution in gradient for each sample in batch (ie. if batch size is 10, and num_samples is 100, the each gradient step with sample 100 * 10 samples from a gaussian distribution); default 50
  * ``early_stopping`` (bool): whether to check loss for convergence; compares the best avg validation loss at the end of an epoch, with current avg epoch loss estimate, if :math:`best_loss - curr_loss < tol` for `n_iter_no_change`, then procedure terminates; default False
  * ``n_iter_no_change`` (int): number of iterations to check for change before declaring convergence; default 5
  * ``verbose`` (bool): whether to print a verbose output with loss logs, etc.; default False 

* ``store`` (cox.store.Store): logging object to keep track regression's train and validation losses   

Attributes:
~~~~~~~~~~~

* ``coef_`` (torch.Tensor): regression weight coefficients 
* ``intercept_`` (torch.Tensor): regression intercept term 
* ``variance_`` (torch.Tensor): if the noise variance is unknown, this property provides its estimate

In the following code block, here, we show an example of how to use the library with unknown noise variance: 
   
.. code-block:: python

  from delphi.stats.truncated_linear_regression import TruncatedLinearRegression
  from delphi import oracle
  from delphi.utils.helpers import Parameters
  from cox.store import Store

  OUT_DIR = 'PATH_TO_EXPERIMENT_LOGGING_DIRECTORY'
  store = Store(OUT_DIR)

  # left truncate linear regression at 0 (ie. S = {y >= 0 for all (x, y) in S})
  phi = oracle.Left_Regression(0.0)
  # pass algorithm parameters in through Parameters object
  train_kwargs = Parameters({'phi': phi, 
                              'alpha': alpha})
  # define trunc linear regression object 
  trunc_reg = TruncatedLinearRegression(train_kwargs, store=store)
  # fit to dataset
  trunc_reg.fit(X, y)
  # close store 
  store.close()
  # make predictions with new regression
  print(trunc_reg.predict(X))

Methods: 
~~~~~~~~

* ``predict(X)``: predict regression points for input feature matrix X (num_samples by features)

TruncatedLassoRegression:
--------------------------
``TruncatedLassoRegression`` learns from truncated LASSO regression model's with the noise 
variance is known. In the known setting we use the algorithm described in the following
paper `Truncated Linear Regression in High Dimensions <https://arxiv.org/abs/2007.14539>`_

When evaluating truncated lasso regression models, the user needs three things; an oracle, a Callable that 
indicates whether a sample falls within the truncation set, the model's ``alpha``, survival probability, and the ``TruncatedLassoRegression`` module. The ``TruncatedLassoRegression`` module accepts 
a parameters object that the user can define for running the PSGD procedure.

Parameters:
~~~~~~~~~~~

* ``args`` (delphi.utils.Parameters): parameters object that holds hyperparameters for experiment. Possible hyperparameters include:

  * ``phi`` (Callable): required argument; callable class that receives num_samples by 1 input ``torch.Tensor``, and returns a num_samples by 1 outputs a num_samples by 1 ``Tensor`` with ``(0, 1)`` representing membership in ``S`` or not.
  * ``alpha`` (float): required argument; survivial probability for truncated regression
  * ``l1`` (float): l1 regularization
  * ``epochs`` (int): maximum number of times to iterate over dataset
  * ``noise_var`` (float): provide noise variance, if the noise variance for the truncated regression model is known, else unknown variance procedure is run by default
  * ``fit_intercept`` (bool): whether to fit the intercept or not; default to True
  * ``trials`` (int): maximum number of trials to perform PSGD; after trials, model with smallest loss on the dataset is returned
  * ``val`` (float): percentage of dataset to use for validation set; default .2
  * ``lr`` (float): initial learning rate to use for regression weights; default 1e-1
  * ``var_lr`` (float): initial learning rate to use variance parameters, when running unknown variance 
  * ``step_lr`` (int): number of gradient steps to take before adjusting learning rate by value ``step_lr_gamma``; default 100
  * ``step_lr_gamma`` (float): amount to adjust learning rate, every ``step_lr`` steps ``new_lr = curr_lr * step_lr_gamma``
  * ``custom_lr_multiplier`` (str): `cosine` or `cyclic` for cosine annealing learning rate scheduling or cyclic learning rate scheduling; default None
  * ``momentum`` (float): momentum; default 0.0 
  * ``adam`` (bool): use adam adaptive learning rate optimizer; default False
  * ``eps`` (float): epsilon denominator for gradients (ie. to prevent divide by zero calculations); default 1e-5
  * ``r`` (float): initial projection set radius; default 1.0
  * ``rate`` (float): at the end of each trial, the projection set radius is increased at rate `rate`; default 1.5
  * ``normalize`` (bool): our methods assume that the :math:`max(||x_{i}||_{2}) <= 1`, so before running the procedure, you must  divide the input featurers :math:`X = \{x_{(1)}, x_{(2)}, ... , x_{(n)}\}` by :math:`max(||x_{i}||_{2}) \dot \sqrt(k)`, where :math:`k` represents the number of dimensions the input features have; by default the procedure normalizes the features for the user
  * ``batch_size`` (int): the number of samples to use for each gradient step; default 50
  * ``tol`` (float): if using early stopping, threshold for when to stop; default 1e-3
  * ``workers`` (int): number of workers to use for procedure; default 1
  * ``num_samples`` (int): number of samples to sample from distribution in gradient for each sample in batch (ie. if batch size is 10, and num_samples is 100, the each gradient step with sample 100 * 10 samples from a gaussian distribution); default 50
  * ``early_stopping`` (bool): whether to check loss for convergence; compares the best avg validation loss at the end of an epoch, with current avg epoch loss estimate, if :math:`best_loss - curr_loss < tol` for `n_iter_no_change`, then procedure terminates; default False
  * ``n_iter_no_change`` (int): number of iterations to check for change before declaring convergence; default 5
  * ``verbose`` (bool): whether to print a verbose output with loss logs, etc.; default False 

* ``store`` (cox.store.Store): logging object to keep track lasso regression's train and validation losses   

Attributes:
~~~~~~~~~~~

* ``coef_`` (torch.Tensor): regression weight coefficients 
* ``intercept_`` (torch.Tensor): regression intercept term 
* ``variance_`` (torch.Tensor): if the noise variance is unknown, this property provides its estimate

In the following code block, here, we show an example of how to use the truncated lasso regression module with known noise variance: 
   
.. code-block:: python
  
  from delphi.stats.truncated_lasso_regression import TruncatedLassoRegression
  from delphi import oracle  
  from delphi.utils.helpers import Parameters
  from cox.store import Store

  OUT_DIR = 'PATH_TO_EXPERIMENT_LOGGING_DIRECTORY'
  store = Store(OUT_DIR)

  # left truncate lasso regression at 0 (ie. S = {y>= 0 for all (x, y) in S})
  phi = oracle.Left_Regression(0.0)
  # pass algorithm parameters in through Parameters object
  train_kwargs = Parameters({'phi': phi, 
                            'alpha': alpha, 
                            'noise_var': 1.0})
  # define trunc linear LASSO regression object
  trunc_lasso_reg = TruncatedLassoRegression(train_kwargs, store=store)
  # fit to dataset
  trunc_lasso_reg.fit(X, y)
  # close store 
  store.close()
  # make predictions with new regression
  print(trunc_lasso_reg.predict(X))

Methods: 
~~~~~~~~

* ``predict(X)``: predict regression points for input feature matrix X (num_samples by features)

TruncatedRidgeRegression:
--------------------------
``TruncatedRidgeRegression`` learns from truncated ridge regression model's when the noise 
variance is known or unknown. 

When evaluating truncated ridge regression models, the user needs three things; an oracle, a Callable that 
indicates whether a sample falls within the truncation set, the model's ``alpha``, survival probability, and the ``TruncatedRidgeRegression`` module. The ``TruncatedRidgeRegression`` module accepts 
a parameters object that the user can define for running the PSGD procedure.

Parameters:
~~~~~~~~~~~

* ``args`` (delphi.utils.Parameters): parameters object that holds hyperparameters for experiment. Possible hyperparameters include:

  * ``phi`` (Callable): required argument; callable class that receives num_samples by 1 input ``torch.Tensor``, and returns a num_samples by 1 outputs a num_samples by 1 ``Tensor`` with ``(0, 1)`` representing membership in ``S`` or not.
  * ``alpha`` (float): required argument; survivial probability for truncated regression
  * ``weight_decay`` (float): weight decay regularization
  * ``epochs`` (int): maximum number of times to iterate over dataset
  * ``noise_var`` (float): provide noise variance, if the noise variance for the truncated regression model is known, else unknown variance procedure is run by default
  * ``fit_intercept`` (bool): whether to fit the intercept or not; default to True
  * ``trials`` (int): maximum number of trials to perform PSGD; after trials, model with smallest loss on the dataset is returned
  * ``val`` (float): percentage of dataset to use for validation set; default .2
  * ``lr`` (float): initial learning rate to use for regression weights; default 1e-1
  * ``var_lr`` (float): initial learning rate to use variance parameters, when running unknown variance 
  * ``step_lr`` (int): number of gradient steps to take before adjusting learning rate by value ``step_lr_gamma``; default 100
  * ``step_lr_gamma`` (float): amount to adjust learning rate, every ``step_lr`` steps ``new_lr = curr_lr * step_lr_gamma``
  * ``custom_lr_multiplier`` (str): `cosine` or `cyclic` for cosine annealing learning rate scheduling or cyclic learning rate scheduling; default None
  * ``momentum`` (float): momentum; default 0.0 
  * ``adam`` (bool): use adam adaptive learning rate optimizer; default False
  * ``eps`` (float): epsilon denominator for gradients (ie. to prevent divide by zero calculations); default 1e-5
  * ``r`` (float): initial projection set radius; default 1.0
  * ``rate`` (float): at the end of each trial, the projection set radius is increased at rate `rate`; default 1.5
  * ``normalize`` (bool): our methods assume that the :math:`max(||x_{i}||_{2}) <= 1`, so before running the procedure, you must  divide the input featurers :math:`X = \{x_{(1)}, x_{(2)}, ... , x_{(n)}\}` by :math:`max(||x_{i}||_{2}) \dot \sqrt(k)`, where :math:`k` represents the number of dimensions the input features have; by default the procedure normalizes the features for the user
  * ``batch_size`` (int): the number of samples to use for each gradient step; default 50
  * ``tol`` (float): if using early stopping, threshold for when to stop; default 1e-3
  * ``workers`` (int): number of workers to use for procedure; default 1
  * ``num_samples`` (int): number of samples to sample from distribution in gradient for each sample in batch (ie. if batch size is 10, and num_samples is 100, the each gradient step with sample 100 * 10 samples from a gaussian distribution); default 50
  * ``early_stopping`` (bool): whether to check loss for convergence; compares the best avg validation loss at the end of an epoch, with current avg epoch loss estimate, if :math:`best_loss - curr_loss < tol` for `n_iter_no_change`, then procedure terminates; default False
  * ``n_iter_no_change`` (int): number of iterations to check for change before declaring convergence; default 5
  * ``verbose`` (bool): whether to print a verbose output with loss logs, etc.; default False 

* ``store`` (cox.store.Store): logging object to keep track lasso regression's train and validation losses   

Attributes:
~~~~~~~~~~~

* ``coef_`` (torch.Tensor): regression weight coefficients 
* ``intercept_`` (torch.Tensor): regression intercept term 
* ``variance_`` (torch.Tensor): if the noise variance is unknown, this property provides its estimate

In the following code block, here, we show an example of how to use the truncated lasso regression module with known noise variance: 
   
.. code-block:: python
  
  from delphi.stats.truncated_ridge_regression import TruncatedRidgeRegression
  from delphi import oracle  
  from delphi.utils.helpers import Parameters
  from cox.store import Store

  OUT_DIR = 'PATH_TO_EXPERIMENT_LOGGING_DIRECTORY'
  store = Store(OUT_DIR)

  # left truncate lasso regression at 0 (ie. S = {y>= 0 for all (x, y) in S})
  phi = oracle.Left_Regression(0.0)
  # pass algorithm parameters in through Parameters object
  train_kwargs = Parameters({'phi': phi, 
                            'alpha': alpha, 
                            'weight_decay': .01,
                            'noise_var': 1.0})
  # define trunc linear LASSO regression object
  trunc_ridge_reg = TruncatedRidgeRegression(train_kwargs, store=store)
  # fit to dataset
  trunc_ridge_reg.fit(X, y)
  # close store 
  store.close()
  # make predictions with new regression
  print(trunc_ridge_reg.predict(X))

Methods: 
~~~~~~~~

* ``predict(X)``: predict regression points for input feature matrix X (num_samples by features)

TruncatedElasticNetRegression:
--------------------------
``TruncatedElasticNetRegression`` learns from truncated elastic net regression model's when the noise 
variance is known or unknown. 

When evaluating truncated elastic net regression models, the user needs three things; an oracle, a Callable that 
indicates whether a sample falls within the truncation set, the model's ``alpha``, survival probability, and the ``TruncatedElasticNetRegression`` module. The ``TruncatedRidgeRegression`` module accepts 
a parameters object that the user can define for running the PSGD procedure.

Parameters:
~~~~~~~~~~~

* ``args`` (delphi.utils.Parameters): parameters object that holds hyperparameters for experiment. Possible hyperparameters include:

  * ``phi`` (Callable): required argument; callable class that receives num_samples by 1 input ``torch.Tensor``, and returns a num_samples by 1 outputs a num_samples by 1 ``Tensor`` with ``(0, 1)`` representing membership in ``S`` or not.
  * ``alpha`` (float): required argument; survivial probability for truncated regression
  * ``weight_decay`` (float): weight decay regularization
  * ``l1`` (float): l1 regularization
  * ``epochs`` (int): maximum number of times to iterate over dataset
  * ``noise_var`` (float): provide noise variance, if the noise variance for the truncated regression model is known, else unknown variance procedure is run by default
  * ``fit_intercept`` (bool): whether to fit the intercept or not; default to True
  * ``trials`` (int): maximum number of trials to perform PSGD; after trials, model with smallest loss on the dataset is returned
  * ``val`` (float): percentage of dataset to use for validation set; default .2
  * ``lr`` (float): initial learning rate to use for regression weights; default 1e-1
  * ``var_lr`` (float): initial learning rate to use variance parameters, when running unknown variance 
  * ``step_lr`` (int): number of gradient steps to take before adjusting learning rate by value ``step_lr_gamma``; default 100
  * ``step_lr_gamma`` (float): amount to adjust learning rate, every ``step_lr`` steps ``new_lr = curr_lr * step_lr_gamma``
  * ``custom_lr_multiplier`` (str): `cosine` or `cyclic` for cosine annealing learning rate scheduling or cyclic learning rate scheduling; default None
  * ``momentum`` (float): momentum; default 0.0 
  * ``adam`` (bool): use adam adaptive learning rate optimizer; default False
  * ``eps`` (float): epsilon denominator for gradients (ie. to prevent divide by zero calculations); default 1e-5
  * ``r`` (float): initial projection set radius; default 1.0
  * ``rate`` (float): at the end of each trial, the projection set radius is increased at rate `rate`; default 1.5
  * ``normalize`` (bool): our methods assume that the :math:`max(||x_{i}||_{2}) <= 1`, so before running the procedure, you must  divide the input featurers :math:`X = \{x_{(1)}, x_{(2)}, ... , x_{(n)}\}` by :math:`max(||x_{i}||_{2}) \dot \sqrt(k)`, where :math:`k` represents the number of dimensions the input features have; by default the procedure normalizes the features for the user
  * ``batch_size`` (int): the number of samples to use for each gradient step; default 50
  * ``tol`` (float): if using early stopping, threshold for when to stop; default 1e-3
  * ``workers`` (int): number of workers to use for procedure; default 1
  * ``num_samples`` (int): number of samples to sample from distribution in gradient for each sample in batch (ie. if batch size is 10, and num_samples is 100, the each gradient step with sample 100 * 10 samples from a gaussian distribution); default 50
  * ``early_stopping`` (bool): whether to check loss for convergence; compares the best avg validation loss at the end of an epoch, with current avg epoch loss estimate, if :math:`best_loss - curr_loss < tol` for `n_iter_no_change`, then procedure terminates; default False
  * ``n_iter_no_change`` (int): number of iterations to check for change before declaring convergence; default 5
  * ``verbose`` (bool): whether to print a verbose output with loss logs, etc.; default False 

* ``store`` (cox.store.Store): logging object to keep track lasso regression's train and validation losses   

Attributes:
~~~~~~~~~~~

* ``coef_`` (torch.Tensor): regression weight coefficients 
* ``intercept_`` (torch.Tensor): regression intercept term 
* ``variance_`` (torch.Tensor): if the noise variance is unknown, this property provides its estimate

In the following code block, here, we show an example of how to use the truncated lasso regression module with known noise variance: 
   
.. code-block:: python
  
  from delphi.stats.truncated_elastic_net_regression import TruncatedElasticNetRegression
  from delphi import oracle  
  from delphi.utils.helpers import Parameters
  from cox.store import Store

  OUT_DIR = 'PATH_TO_EXPERIMENT_LOGGING_DIRECTORY'
  store = Store(OUT_DIR)

  # left truncate lasso regression at 0 (ie. S = {y>= 0 for all (x, y) in S})
  phi = oracle.Left_Regression(0.0)
  # pass algorithm parameters in through Parameters object
  train_kwargs = Parameters({'phi': phi, 
                            'alpha': alpha, 
                            'weight_decay': .01,
                            'noise_var': 1.0})
  # define trunc linear LASSO regression object
  trunc_elastic_reg = TruncatedRidgeRegression(train_kwargs, store=store)
  # fit to dataset
  trunc_elastic_reg.fit(X, y)
  # close store 
  store.close()
  # make predictions with new regression
  print(trunc_elastic_reg.predict(X))

Methods: 
~~~~~~~~

* ``predict(X)``: predict regression points for input feature matrix X (num_samples by features)

TruncatedLogisticRegression:
--------------------------
``TruncatedLogisticRegression`` learns truncated logistic regression models by maximizing the truncated log likelihood.
The algorithm that we use for this procedure is described in the following
paper `A Theoretical and Practical Framework for Classification and Regression from Truncated Samples <https://proceedings.mlr.press/v108/ilyas20a.html>`_.
.

When evaluating truncated logistic regression models, the user needs three things; an oracle, a Callable that 
indicates whether a sample falls within the truncation set, the model's ``alpha``, survival probability, and the ``TruncatedLogisticRegression`` module. The ``TruncatedLogisticRegression`` module accepts 
a parameters object that the user can define for running the PSGD procedure. 

Parameters:
-----------

* ``args`` (delphi.utils.Parameters): parameters object that holds hyperparameters for experiment. Possible hyperparameters include:

  * ``phi`` (Callable): required argument; callable class that receives num_samples by 1 input ``torch.Tensor``, and returns a num_samples by 1 outputs a num_samples by 1 ``Tensor`` with ``(0, 1)`` representing membership in ``S`` or not.
  * ``alpha`` (float): required argument; survivial probability for truncated regression
  * ``epochs`` (int): maximum number of times to iterate over dataset
  * ``fit_intercept`` (bool): whether to fit the intercept or not; default to True
  * ``trials`` (int): maximum number of trials to perform PSGD; after trials, model with smallest loss on the dataset is returned
  * ``val`` (float): percentage of dataset to use for validation set; default .2
  * ``lr`` (float): initial learning rate to use for regression weights; default 1e-1
  * ``var_lr`` (float): initial learning rate to use variance parameters, when running unknown variance 
  * ``step_lr`` (int): number of gradient steps to take before adjusting learning rate by value ``step_lr_gamma``; default 100
  * ``step_lr_gamma`` (float): amount to adjust learning rate, every ``step_lr`` steps ``new_lr = curr_lr * step_lr_gamma``
  * ``custom_lr_multiplier`` (str): `cosine` or `cyclic` for cosine annealing learning rate scheduling or cyclic learning rate scheduling; default None
  * ``momentum`` (float): momentum; default 0.0 
  * ``adam`` (bool): use adam adaptive learning rate optimizer; default False
  * ``eps`` (float): epsilon denominator for gradients (ie. to prevent divide by zero calculations); default 1e-5
  * ``r`` (float): initial projection set radius; default 1.0
  * ``rate`` (float): at the end of each trial, the projection set radius is increased at rate `rate`; default 1.5
  * ``normalize`` (bool): our methods assume that the :math:`max(||x_{i}||_{2}) <= 1`, so before running the procedure, you must  divide the input featurers :math:`X = {x_{(1)}, x_{(2)}, ... , x_{(n)}}` by :math:`max(||x_{i}||_{2}) \dot \sqrt(k)`, where :math:`k` represents the number of dimensions the input features have; by default the procedure normalizes the features for the user
  * ``batch_size`` (int): the number of samples to use for each gradient step; default 50
  * ``tol`` (float): if using early stopping, threshold for when to stop; default 1e-3
  * ``workers`` (int): number of workers to use for procedure; default 1
  * ``num_samples`` (int): number of samples to sample from distribution in gradient for each sample in batch (ie. if batch size is 10, and num_samples is 100, the each gradient step with sample 100 * 10 samples from a gaussian distribution); default 50
  * ``early_stopping`` (bool): whether to check loss for convergence; compares the best avg validation loss at the end of an epoch, with current avg epoch loss estimate, if :math:`best_loss - curr_loss < tol` for `n_iter_no_change` epochs, then procedure terminates; default False
  * ``n_iter_no_change`` (int): number of iterations to check for change before declaring convergence; default 5
  * ``verbose`` (bool): whether to print a verbose output with loss logs, etc.; default False - just a tdqm output

* ``store`` (cox.store.Store): logging object to keep track logistic regression's train and validation losses and accuracy   

Attributes:
~~~~~~~~~~~

* ``coef_`` (torch.Tensor): regression weight coefficients 
* ``intercept_`` (torch.Tensor): regression intercept term 

In the following code block, here, we show an example of how to use the truncated logistic regression module: 
   
.. code-block:: python

  from delphi.stats.truncated_logistic_regression import TruncatedLogisticRegression
  from delphi import oracle
  from delphi.utils.helpers import Parameters
  from cox.store import Store

  OUT_DIR = 'PATH_TO_EXPERIMENT_LOGGING_DIRECTORY'
  store = Store(OUT_DIR)

  # left truncate logistic regression at 0 (ie. S = {z >= -.1 for all (x, y) in S})
  phi = oracle.Left_Regression(-0.1)
  # pass algorithm parameters in through parameter object
  train_kwargs = Parameters({'phi': phi, 
                              'alpha': alpha})
  # define truncated logistic regression object
  trunc_log_reg = TruncatedLogisticRegression(train_kwargs, store=store)
  # fit to dataset
  trunc_log_reg.fit(X, y)
  # close store 
  store.close()
  # make predictions with new regression
  print(trunc_log_reg.predict(X))


Methods: 
~~~~~~~~

* ``predict(X)``: predict classification for input feature matrix X (num_samples by features)


TruncatedProbitRegression:
--------------------------
``TruncatedProbitRegression`` learns truncated probit regression models, by maximizing the truncated log likelihood.
The algorithm that we use for this procedure is described in the following
paper `A Theoretical and Practical Framework for Classification and Regression from Truncated Samples <https://proceedings.mlr.press/v108/ilyas20a.html>`_.

When evaluating truncated logistic regression models, the user needs three things; an oracle, a Callable that 
indicates whether a sample falls within the truncation set, the model's ``alpha``, survival probability, and ``TruncatedProbitRegression`` module.  The ``TruncatedProbitRegression`` module accepts 
a parameters object that the user can define for running the PSGD procedure.

Parameters:
-----------

* ``args`` (delphi.utils.Parameters): parameters object that holds hyperparameters for experiment. Possible hyperparameters include:

  * ``phi`` (Callable): required argument; callable class that receives num_samples by 1 input ``torch.Tensor``, and returns a num_samples by 1 outputs a num_samples by 1 ``Tensor`` with ``(0, 1)`` representing membership in ``S`` or not.
  * ``alpha`` (float): required argument; survivial probability for truncated regression
  * ``epochs`` (int): maximum number of times to iterate over dataset
  * ``fit_intercept`` (bool): whether to fit the intercept or not; default to True
  * ``trials`` (int): maximum number of trials to perform PSGD; after trials, model with smallest loss on the dataset is returned
  * ``val`` (float): percentage of dataset to use for validation set; default .2
  * ``lr`` (float): initial learning rate to use for regression weights; default 1e-1
  * ``step_lr`` (int): number of gradient steps to take before adjusting learning rate by value ``step_lr_gamma``; default 100
  * ``step_lr_gamma`` (float): amount to adjust learning rate, every ``step_lr`` steps ``new_lr = curr_lr * step_lr_gamma``
  * ``custom_lr_multiplier`` (str): `cosine` or `cyclic` for cosine annealing learning rate scheduling or cyclic learning rate scheduling; default None
  * ``momentum`` (float): momentum; default 0.0 
  * ``adam`` (bool): use adam adaptive learning rate optimizer; default False
  * ``eps`` (float): epsilon denominator for gradients (ie. to prevent divide by zero calculations); default 1e-5
  * ``r`` (float): initial projection set radius; default 1.0
  * ``rate`` (float): at the end of each trial, the projection set radius is increased at rate `rate`; default 1.5
  * ``normalize`` (bool): our methods assume that the :math:`max(||x_{i}||_{2}) <= 1`, so before running the procedure, you must  divide the input featurers :math:`X = \{x_{(1)}, x_{(2)}, ... , x_{(n)}\}` by :math:`max(||x_{i}||_{2}) \dot \sqrt(k)`, where :math:`k` represents the number of dimensions the input features have; by default the procedure normalizes the features for the user
  * ``batch_size`` (int): the number of samples to use for each gradient step; default 50
  * ``tol`` (float): if using early stopping, threshold for when to stop; default 1e-3
  * ``workers`` (int): number of workers to use for procedure; default 1
  * ``num_samples`` (int): number of samples to sample from distribution in gradient for each sample in batch (ie. if batch size is 10, and num_samples is 100, the each gradient step with sample 100 * 10 samples from a gaussian distribution); default 50
  * ``early_stopping`` (bool): whether to check loss for convergence; compares the best avg validation loss at the end of an epoch, with current avg epoch loss estimate, if :math:`best_loss - curr_loss < tol` for `n_iter_no_change`, then procedure terminates; default False
  * ``n_iter_no_change`` (int): number of iterations to check for change before declaring convergence; default 5
  * ``verbose`` (bool): whether to print a verbose output with loss logs, etc.; default False 

* ``store`` (cox.store.Store): logging object to keep track probit regression's train and validation losses and accuracy 

Attributes:
~~~~~~~~~~~

* ``coef_`` (torch.Tensor): regression weight coefficients 
* ``intercept_`` (torch.Tensor): regression intercept term 

In the following code block, here, we show an example of how to use the truncated probit regression module: 
   
.. code-block:: python

  from delphi.stats.truncated_probit_regression import TruncatedProbitRegression
  from delphi import oracle
  from delphi.utils.helpers import Parameters
  from cox.store import Store

  OUT_DIR = 'PATH_TO_EXPERIMENT_LOGGING_DIRECTORY'
  store = Store(OUT_DIR)

  # left truncate probit regression at 0 (ie. S = {z >= -0.1 for all (x, y) in S})
  phi = oracle.Left_Regression(-0.1)
  # pass algorithm parameters in through dictionary
  train_kwargs = Parameters({'phi': phi, 
                              'alpha': alpha})
  # define truncated probit regression object
  trunc_prob_reg = TruncatedProbitRegression(train_kwargs, store=store)
  # fit to dataset
  trunc_prob_reg.fit(X, y)
  # close store 
  store.close()
  # make predictions with new regression
  print(trunc_prob_reg.predict(X))

Methods: 
~~~~~~~~

* ``predict(X)``: predict classification for input feature matrix X (num_samples by features)
