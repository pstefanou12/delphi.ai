delphi.ai package
=================
Install via ``pip``: ``pip install delphi.ai``

This library holds a collection of algorithms that can be used 
debias models that have been defected due to truncation, or missing data. A few 
projects using the library can found in: 
`Code for "Efficient Truncated Linear Regression with Unknown Noise Variance" <https://github.com/pstefanou12/Truncated-Regression-With-Unknown-Noise-Variance-NeurIPS-2021>_`


We demonstrate how to use the library in a set of walkthroughs and our API
reference. Functionality provided by the library includes:

TruncatedLinearRegression:
--------------------------
``TruncatedLinearRegression`` learns from truncated linear regression model's with the noise 
variance is known or unknown. In the known setting we use the algorithm described in the following
papers `Computationally and Statistically Efficient Truncated Regression "https://arxiv.org/abs/2010.12000"_`


To use the package, the user needs 


When evaluating truncated regression models, the user needs to ``import`` two objects; an oracle, derived from 
the ``delphi.oracle`` class and the ``TruncatedLinearRegression`` object. You can read 
about selecting and or defining the oracle in <>. The ``TruncatedLinearRegression`` module accepts 
a parameters dictionary that the user can define for running the SGD procedure.
The possible arguments are: 
* ``phi`` (delphi.oracle): required argument; callable class that receives num_samples by 1 input ``torch.Tensor``, and returns a num_samples by 1 outputs a num_samples by 1 ``Tensor`` with ``(0, 1)`` representing membership in ``S`` or not.
* ``alpha`` (float): required argument; survivial probability for truncated regression
* ``epochs`` (int): maximum number of times to iterate over dataset
* ``noise_var`` (float): provide noise variance, if the noise variance for the truncated regression model is known, else unknown variance procedure is run by default
* ``fit_intercept`` (bool): whether to fit the intercept or not; default to True
* ``num_trials`` (int): maximum number of trials to perform PSGD; after num_trials, model with smallest loss on the dataset is returned
* ``clamp`` (bool): to use a projection set or not; provides range around empirical estimates for potential optimal values; default True 
* ``val`` (float): percentage of dataset to use for validation set; default .2
* ``lr`` (float): initial learning rate to use for regression weights; default 1e-1
* ``var_lr`` (float): initial learning rate to use variance parameters, when running unknown variance 
* ``step_lr`` (int): number of gradient steps to take before adjusting learning rate by value ``step_lr_gamma``; default 100
* ``step_lr_gamma`` (float): amount to adjust learning rate, every ``step_lr`` steps ``new_lr = curr_lr * step_lr_gamma``
* ``custom_lr_multiplier`` (str): `cosine` or `cyclic` for cosine annealing learning rate scheduling or cyclic learning rate scheduling; default None
* ``momentum`` (float): momentum; default 0.0 
* ``weight_decay`` (float): weight_decay; default 0.0
* ``l1`` (float): l1 regularization
* ``eps`` (float): epsilon denominator for gradients (ie. to prevent divide by zero calculations); default 1e-5
* ``r`` (float): initial projection set radius; default 1.0
* ``rate`` (float): at the end of each trial, the projection set radius is increased at rate `rate`; default 1.5
* ``normalize`` (bool): our methods assume that the :math:`max(||x_{i}||_{2}) <= 1`, so before running the procedure, you must  divide the input featurers :math:`\mathcal{X} = \{x_{(1)}, x_{(2)}, ... , x_{(n)}\}` by :math:`max(||x_{i}||_{2}) \dot \sqrt(k)`, where :math:`\mathcal{k}` represents the number of dimensions the input features have; by default the procedure normalizes the features for the user
* ``batch_size`` (int): the number of samples to use for each gradient step; default 50
* ``tol`` (float): if using early stopping, threshold for when to stop; default 1e-3
* ``workers`` (int): number of workers to use for procedure; default 1
* ``num_samples`` (int): number of samples to sample from distribution in gradient for each sample in batch (ie. if batch size is 10, and num_samples is 100, the each gradient step with sample 100 * 10 samples from a gaussian distribution); default 50
* ``early_stopping`` (bool): whether to check loss for convergence; compares the best avg validation loss at the end of an epoch, with current avg epoch loss estimate, if $best_loss - curr_loss$ < tol for ``n_iter_no_change``, then procedure terminates; default False
* ``n_iter_no_change`` (int): number of iterations to check for change before declaring convergence; default 5
* ``verbose`` (bool): whether to print a verbose output with loss logs, etc.; default False 
   
For example in the code block here:
   
.. code-block:: python

  from delphi.stats.truncated_linear_regression import TruncatedLinearRegression
  from delphi import oracle

  # left truncate linear regression at 0 (ie. S = {x >= 0 for all x in S})
  phi = oracle.Left_Regression(0.0)

  # define trunc linear regression object
  # pass algorithm parameters in through dictionary
  trunc_reg = TruncatedLinearRegression({'phi': phi, 
                                          'alpha': alpha})
  # fit to dataset
  trunc_reg.fit(X, y)


    
