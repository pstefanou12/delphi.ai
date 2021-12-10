delphi.ai package
=================
Install via ``pip``: ``pip install delphi.ai``

This library holds a collection of algorithms that can be used 
debias models that have been defected due to truncation, or missing data. A few 
projects using the library can found in: 
`Code for "Efficient Truncated Linear Regression with Unknown Noise Variance" <https://github.com/pstefanou12/Truncated-Regression-With-Unknown-Noise-Variance-NeurIPS-2021>`


We demonstrate how to use the library in a set of walkthroughs and our API
reference. Functionality provided by the library includes:

Truncated Linear with Regression with Known and Unknown Noise Variance:

When evaluating truncated regression models, the user needs to ``import`` two objects 
to anaylze the model; an oracle object and the ``TruncatedLinearRegression`` object.
For example in the code block here:
.. code-block:: bash

  from delphi.stats.linear_regression import TruncatedLinearRegression
  from delphi import oracle
  phi = oracle.Left_Regression(0.0)

  trunc_reg = TruncatedLinearRegression(phi=phi, 
                                        alpha=alpha)
  trunc_reg.fit(X, y)

    
