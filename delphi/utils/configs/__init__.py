# Author: pstefanou12@
"""Pydantic configuration models for delphi.ai algorithms.

Split by estimator family so no single module has to hold every config.
Every name is re-exported here, so ``from delphi.utils import configs``
followed by ``configs.SomeConfig`` keeps working regardless of which
submodule a config lives in.
"""

from delphi.utils.configs.base import (
    LinearModelConfig,
    OptimizerConfig,
    TrainerConfig,
    _ConfigT,
)
from delphi.utils.configs.classification import (
    SoftmaxRegressionConfig,
    TruncatedLogisticRegressionConfig,
    TruncatedProbitRegressionConfig,
)
from delphi.utils.configs.distributions import (
    TruncatedExponentialFamilyDistributionConfig,
    TruncatedMultivariateNormalConfig,
)
from delphi.utils.configs.lqr import TruncatedLQRConfig
from delphi.utils.configs.regression import (
    TruncatedElasticNetRegressionConfig,
    TruncatedLassoRegressionConfig,
    TruncatedLinearDynamicalSystemConfig,
    TruncatedRegressionConfig,
    TruncatedRegressionFieldsConfig,
    TruncatedRidgeRegressionConfig,
)

__all__ = [
    "LinearModelConfig",
    "OptimizerConfig",
    "SoftmaxRegressionConfig",
    "TrainerConfig",
    "TruncatedElasticNetRegressionConfig",
    "TruncatedExponentialFamilyDistributionConfig",
    "TruncatedLQRConfig",
    "TruncatedLassoRegressionConfig",
    "TruncatedLinearDynamicalSystemConfig",
    "TruncatedLogisticRegressionConfig",
    "TruncatedMultivariateNormalConfig",
    "TruncatedProbitRegressionConfig",
    "TruncatedRegressionConfig",
    "TruncatedRegressionFieldsConfig",
    "TruncatedRidgeRegressionConfig",
    "_ConfigT",
]
