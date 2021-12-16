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
default hyperparameters can be seen by looking at the `delphi.utils.defaults` directory.


Contents:
--------

* `stats <#stats>`__ : ``stats`` module includes models for regression and classification from truncated samples
 
  * `TruncatedLinearRegression <#TruncatedLinearRegression>`__
  * `TruncatedLassoRegression <#TruncatedLassoRegression>`__
  * `TruncatedLogisticRegression <#TruncatedLogisticRegression>`__
  * `TruncatedProbitRegression <#TruncatedProbitRegression>`__

* `distributions <#distributions>`__: ``distributions`` module includes algorithms for learning from censored (known truncation) and truncated (unknown truncation; unsupervised learning) distributions

  * `CensoredNormal <#CensoredNormal>`__
  * `CensoredMultivariateNormal <#CensoredMultivariateNormal>`__
  * `TruncatedNormal <#TruncatedNormal>`__
  * `TruncatedMultivariateNormal <#TruncatedMultivariateNormal>`__


stats
=====

TruncatedLinearRegression:
--------------------------
``TruncatedLinearRegression`` learns from truncated linear regression model's with the noise 
variance is known or unknown. In the known setting we use the algorithm described in the following
paper: `Computationally and Statistically Efficient Truncated Regression <https://arxiv.org/abs/2010.12000>`_. When 
the variance of the ground-truth linear regression's model is unknown, we use the algorithm described in 
the following paper: `Efficient Truncated Linear Regression with Unknown Noise Variance`.

When evaluating truncated regression models, the user needs two objects; an oracle, an oracle, which is a Callable that 
indicates whether a sample falls within the truncation set, and the ``TruncatedLinearRegression`` module.  The ``TruncatedLinearRegression`` module accepts 
a parameters object that the user can define for running the PSGD procedure.
The possible arguments are: 

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
   
Additionally, the user can also provide a `Store` object which is a logging object from the `cox <https://github.com/MadryLab/cox>`_, an experimental design and analysis framework 
from MadryLab. The store will track the regression's train and validation losses.

In the following code block, here, we show an example of how to use the library with unknown noise variance: 
   
.. code-block:: python

  from delphi.stats.truncated_linear_regression import TruncatedLinearRegression
  from delphi import oracle
  from delphi.utils.helpers import Parameters
  from cox.store import Store

  OUT_DIR = 'PATH_TO_EXPERIMENT_LOGGING_DIRECTORY'
  store = Store(OUT_DIR)

  # left truncate linear regression at 0 (ie. S = {x >= 0 for all x in S})
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

TruncatedLassoRegression:
--------------------------
``TruncatedLassoRegression`` learns from truncated LASSO regression model's with the noise 
variance is known. In the known setting we use the algorithm described in the following
paper `Truncated Linear Regression in High Dimensions <https://arxiv.org/abs/2007.14539>`_

To use the package, the user needs 

When evaluating truncated lasso regression models, the user needs two objects; an oracle, which is a Callable that 
indicates whether a sample falls within the truncation set, and the ``TruncatedLassoRegression`` module. The ``TruncatedLassoRegression`` module accepts 
a parameters object that the user can define for running the PSGD procedure.
The possible arguments are: 

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
* ``l1`` (float): l1 regularization
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

Additionally, the user can also provide a `Store` object which is a logging object from the `cox <https://github.com/MadryLab/cox>`_, an experimental design and analysis framework 
from MadryLab. The store will track the regression's train and validation losses.

In the following code block, here, we show an example of how to use the truncated lasso regression module with known noise variance: 
   
.. code-block:: python
  
  from delphi.stats.truncated_lasso_regression import TruncatedLassoRegression
  from delphi import oracle  
  from delphi.utils.helpers import Parameters
  from cox.store import Store

  OUT_DIR = 'PATH_TO_EXPERIMENT_LOGGING_DIRECTORY'
  store = Store(OUT_DIR)

  # left truncate lasso regression at 0 (ie. S = {x >= 0 for all x in S})
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

TruncatedLogisticRegression:
--------------------------
``TruncatedLogisticRegression`` learns truncated logistic regression models by maximizing the truncated log likelihood.
The algorithm that we use for this procedure is described in the following
paper `A Theoretical and Practical Framework for Classification and Regression from Truncated Samples <https://proceedings.mlr.press/v108/ilyas20a.html>`_.
.

When evaluating truncated logistic regression models, the user needs two objects; an oracle, which is a Callable that 
indicates whether a sample falls within the truncation set, and the ``TruncatedLogisticRegression`` module. The ``TruncatedLogisticRegression`` module accepts 
a parameters object that the user can define for running the PSGD procedure. 

The possible arguments are: 

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

Additionally, the user can also provide a `Store` object which is a logging object from the `cox <https://github.com/MadryLab/cox>`_, an experimental design and analysis framework 
from MadryLab. The store will track the regression's loss and accuracy on both the training and validation sets.
   
In the following code block, here, we show an example of how to use the truncated logistic regression module: 
   
.. code-block:: python

  from delphi.stats.truncated_logistic_regression import TruncatedLogisticRegression
  from delphi import oracle
  from delphi.utils.helpers import Parameters
  from cox.store import Store

  OUT_DIR = 'PATH_TO_EXPERIMENT_LOGGING_DIRECTORY'
  store = Store(OUT_DIR)

  # left truncate logistic regression at 0 (ie. S = {x >= 0 for all x in S})
  phi = oracle.Left_Regression(0.0)

  # pass algorithm parameters in through dictionary
  train_kwargs = Parameters({'phi': phi, 
                              'alpha': alpha})
  # define truncated logistic regression object
  trunc_log_reg = TruncatedLogisticRegression(train_kwargs, store=store)
  # fit to dataset
  trunc_log_reg.fit(X, y)

  # close store 
  store.close()

TruncatedProbitRegression:
--------------------------
``TruncatedProbitRegression`` learns truncated probit regression models, by maximizing the truncated log likelihood.
The algorithm that we use for this procedure is described in the following
paper `A Theoretical and Practical Framework for Classification and Regression from Truncated Samples <https://proceedings.mlr.press/v108/ilyas20a.html>`_.

When evaluating truncated logistic regression models, the user needs two two objects, an oracle, which is a Callable that 
indicates whether a sample falls within the truncation set, and ``TruncatedProbitRegression`` module.  The ``TruncatedProbitRegression`` module accepts 
a parameters object that the user can define for running the PSGD procedure.
The possible arguments are: 

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

Additionally, the user can also provide a `Store` object which is a logging object from the `cox <https://github.com/MadryLab/cox>`_, an experimental design and analysis framework 
from MadryLab. The store will track the regression's loss and accuracy on both the training and validation sets.

In the following code block, here, we show an example of how to use the truncated probit regression module: 
   
.. code-block:: python

  from delphi.stats.truncated_probit_regression import TruncatedProbitRegression
  from delphi import oracle
  from delphi.utils.helpers import Parameters
  from cox.store import Store

  OUT_DIR = 'PATH_TO_EXPERIMENT_LOGGING_DIRECTORY'
  store = Store(OUT_DIR)

  # left truncate probit regression at 0 (ie. S = {x >= 0 for all x in S})
  phi = oracle.Left_Regression(0.0)

  # pass algorithm parameters in through dictionary
  train_kwargs = Parameters({'phi': phi, 
                              'alpha': alpha})
  # define truncated probit regression object
  trunc_prob_reg = TruncatedProbitRegression(train_kwargs, store=store)

  # fit to dataset
  trunc_prob_reg.fit(X, y)

  # close store 
  store.close()


distributions
=============

CensoredNormal:
---------------
``CensoredNormal`` learns censored normal distributions, by maximizing the truncated log likelihood.
The algorithm that we use for this procedure is described in the following
paper `Efficient Statistics in High Dimensions from Truncated Samples <https://arxiv.org/abs/1809.03986>`_.

When evaluating censored normal distributions, the user needs two objects; an oracle, which is a Callable that 
indicates whether a sample falls within the truncation set, and the ``CensoredNormal`` module. The ``CensoredNormal`` module accepts 
a parameters object that the user can define for running the SGD procedure.
The possible arguments are: 

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

Additionally, the user can also provide a `Store` object which is a logging object from the `cox <https://github.com/MadryLab/cox>`_, an experimental design and analysis framework 
from MadryLab. The store will track the distribution's train and validation losses. 

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

When evaluating censored multivariate normal distributions, the user needs to ``import`` two objects, an oracle, which is a Callable that 
indicates whether a sample falls within the truncation set, and the ``CensoredMultivariateNormal`` module. The ``CensoredMultivariateNormal`` module accepts 
a parameters object that the user can define for running the SGD procedure.
The possible arguments are: 

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

Additionally, the user can also provide a `Store` object which is a logging object from the `cox <https://github.com/MadryLab/cox>`_, an experimental design and analysis framework 
from MadryLab. The store will track the distribution's train and validation losses. 

In the following code block, here, we show an example of how to use the censored multivariate normal distribution module: 
   
.. code-block:: python

  from delphi.distributions.censored_multivariate_normal import CensoredMultivariateNormal
  from delphi import oracle
  from delphi.utils.helpers import Parameters
  from cox.store import Store

  OUT_DIR = 'PATH_TO_EXPERIMENT_LOGGING_DIRECTORY'
  store = Store(OUT_DIR)

  # left truncate 0 (ie. S = {x >= 0 for all x in S})
  phi = oracle.Left_Distribution([0.0, 0.0])

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

When evaluating truncated normal distributions, the user needs to ``import`` the ``TruncatedNormal`` object. The ``TruncatedNormal`` module accepts 
a parameters object that the user can define for running the SGD procedure. When *debiasing* truncated normal distributions, we don't require a membership 
oracle, as it is unknown. However, after running our procedure, we are able to provide an approximation of what the truncation set is. Since the user 
inputs a membership oracle in the ``args`` object, when the truncation set is known, we add the learned membership oracle to the ``args`` object as well.

**NOTE:** when learning truncation sets, the user can not pass in a ``Parameters`` object directly into the ``TruncatedNormal`` object, because they will not 
be able to access the ``Parameters`` object afterwards.

The possible arguments are: 

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
   
Additionally, the user can also provide a `Store` object which is a logging object from the `cox <https://github.com/MadryLab/cox>`_, an experimental design and analysis framework 
from MadryLab. The store will track the distribution's train and validation losses. 

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

When evaluating truncated multivariate normal distributions, the user needs to ``import`` two objects; an oracle, which is a Callable that 
indicates whether a sample falls within the truncation set, and the ``TruncatedMultivariateNormal`` module. The ``TruncatedNormal`` module accepts 
a parameters object that the user can define for running the PSGD procedure.

**NOTE:** when learning truncation sets, the user can not pass in a ``Parameters`` object directly into the ``TruncatedNormal`` object, because they will not 
be able to access the ``Parameters`` object afterwards.

The possible arguments are: 

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

Additionally, the user can also provide a `Store` object which is a logging object from the `cox <https://github.com/MadryLab/cox>`_, an experimental design and analysis framework 
from MadryLab. The store will track the distribution's train and validation losses. 

In the following code block, here, we show an example of how to use the truncated multivariate normal distribution module: 
   
.. code-block:: python

  from delphi.distributions.truncated_multivariate_normal import TruncatedMultivariateNormal
  from delphi.utils.helpers import Parameters
  from delphi import oracle
  from cox.store import Store

  OUT_DIR = 'PATH_TO_EXPERIMENT_LOGGING_DIRECTORY'
  store = Store(OUT_DIR)

  # left truncate 0 (ie. S = {x >= 0 for all x in S})
  phi = oracle.Left_Distribution(0.0)

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
  M = MultivariateNormal(ch.zeros(k,), ch.eye(k))
  samples = M.rsample([1000,])

  # filter samples with learning membership oracle
  filtered = train_kwargs.phi(samples)
