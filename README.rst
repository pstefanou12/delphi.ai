delphi.ai package
=================
Install via ``pip``: ``pip install delphi.ai``

This library holds a collection of algorithms that can be used debias models that have been defected due to truncation, or missing data. 

To install the package, 
.. code-block:: bash

   pip install --user -r requirements.txt

We
demonstrate how to use the library in a set of walkthroughs and our API
reference. Functionality provided by the library includes:

- Evaluating truncated linear regression models, using PSGD on the truncated negative log-likelihood

.. code-block:: bash

  from delphi.stats.linear_regression import TruncatedLinearRegression
  from delphi import oracle
  phi = oracle.Left_Regression(0.0)

  trunc_reg = TruncatedLinearRegression(phi=phi, 
                                        alpha=alpha)
  trunc_reg.fit(X, y)

    
