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

When evaluating truncated regression models, the user needs to ``import`` two objects; an oracle, derived from 
the ``delphi.oracle`` class and the ``TruncatedLinearRegression`` object.
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

    
