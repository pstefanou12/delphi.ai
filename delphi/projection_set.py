"""
Projection sets for algorithms.
"""


class TruncatedRegressionProjectionSet:
    """
    Project to domain for linear regression with known variance
    """
    def __init__(self, trunc_reg, radius, alpha, clamp):
        self.emp_weight = trunc_reg.weight.data
        self.emp_bias = trunc_reg.bias.data if trunc_reg.bias else None
        self.radius = radius * (4.0 * ch.log(2.0 / alpha) + 7.0)
        self.clamp = clamp
        if self.clamp:
            self.weight_bounds = Bounds(self.emp_weight.flatten() - self.radius,
                                        self.emp_weight.flatten() + self.radius)
            self.bias_bounds = Bounds(self.emp_bias.flatten() - self.radius,
                                      self.emp_bias.flatten() + self.radius) if trunc_reg.bias is not None else None
        else:
            pass

    def __call__(self, trunc_reg, i, loop_type, inp, target):
        if self.clamp:
            trunc_reg.weight.data = ch.stack(
                [ch.clamp(trunc_reg.weight[i], self.weight_bounds.lower[i], self.weight_bounds.upper[i]) for i in
                 range(trunc_reg.weight.size(0))])
            if trunc_reg.bias is not None:
                trunc_reg.bias.data = ch.clamp(trunc_reg.bias, self.bias_bounds.lower, self.bias_bounds.upper).reshape(
                    trunc_reg.bias.size())
        else:
            pass


class TruncatedRegressionUnknownVarianceProjectionSet:
    """
    Project parameter estimation back into domain of expected results for censored normal distributions.
    """

    def __init__(self, trunc_reg, radius, alpha, clamp):
        """
        :param emp_lin_reg: empirical regression with unknown noise variance
        """
        self.emp_var = turnc_reg.lambda_.data.inverse()
        self.emp_weight = trunc_reg.v.data * self.emp_var
        self.emp_bias = trunc_reg.bias.data * self.emp_var if trunc_reg.bias is not None else None
        self.radius = radius * (12.0 + 4.0 * ch.log(2.0 / config.args.alpha))
        self.clamp = clamp
        if self.clamp:
            self.weight_bounds, self.var_bounds = Bounds(self.emp_weight.flatten() - self.radius,
                                                         self.emp_weight.flatten() + self.radius), Bounds(
                self.emp_var.flatten() / radius, (self.emp_var.flatten()) / alpha.pow(2))
            self.bias_bounds = Bounds(self.emp_bias.flatten() - self.radius,
                                      self.emp_bias.flatten() + self.radius) if trunc_reg.bias else None
        else:
            pass

    def __call__(self, trunc_reg, i, loop_type, inp, target):
        # reparameterize
        var = trunc_reg.lambda_.inverse()
        weight = trunc_reg.v.data * var

        if self.clamp:
            # project noise variance
            trunc_reg.lambda_.data = ch.clamp(var, float(self.var_bounds.lower), float(self.var_bounds.upper)).inverse()
            # project weights
            trunc_reg.v.data = ch.cat(
                [ch.clamp(weight[i].unsqueeze(0), float(self.weight_bounds.lower[i]),
                          float(self.weight_bounds.upper[i]))
                 for i in range(weight.size(0))]) * trunc_reg.lambda_
            # project bias
            if trunc_reg.bias is not None:
                bias = trunc_reg.bias * var
                trunc_reg.bias.data = ch.clamp(bias, float(self.bias_bounds.lower), float(self.bias_bounds.upper)) * M.lambda_
        else:
            pass