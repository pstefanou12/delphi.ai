import torch as ch
from torch import Tensor
from torch.nn import Linear
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torch import sigmoid as sig
from torch.distributions import Uniform
from torch.distributions.transforms import SigmoidTransform
from torch.distributions.transformed_distribution import TransformedDistribution
from cox.utils import Parameters
import config

from .stats import stats
from ..oracle import oracle
from ..train import train_model
from ..utils.helpers import Bounds


class truncated_logistic_regression(stats):
    """
    """
    def __init__(
            self,
            phi: oracle,
            alpha: float,
            bias: bool=True,
            scale: float = 1.0,
            epochs: int=50,
            lr: float=1e-1,
            num_samples: int=100,
            radius: float=2.0,
            clamp: bool=True,
            eps: float = 1e-5,
            tol: float = 1e-1,
            device: str = "cpu",
            **kwargs):
        """
        """
        super().__init__()
        # initialize hyperparameters for algorithm
        config.args = Parameters({
            'phi': phi,
            'epochs': epochs,
            'lr': lr,
            'num_samples': num_samples,
            'alpha': Tensor([alpha]),
            'radius': Tensor([radius]),
            'scale': Tensor([scale]),
            'bias': bias,
            'clamp': clamp,
            'var': Tensor([1.0]),
            'momentum': 0.0,
            'weight_decay': 0.0,
            'tol': tol,
            'eps': eps,
            'device': device,
        })
        self._lin_reg = None
        self.projection_set = None
        # intialize loss function and add custom criterion to hyperparameters
        self.criterion = TruncatedBCE.apply
        config.args.__setattr__('custom_criterion', self.criterion)

    def fit(
            self,
            S: DataLoader):
        """
        """
        # initialize model with empiricial estimates
        self._log_reg = Linear(in_features=S.dataset.X.size(1), out_features=1, bias=config.args.bias)
        self._log_reg.weight = ch.nn.Parameter(Tensor(S.dataset.log_reg.coef_))
        if config.args.bias:
            self._log_reg.bias = ch.nn.Parameter(Tensor(S.dataset.log_reg.intercept_))
        self.projection_set = TruncatedLogisticRegression(self._log_reg)
        config.args.__setattr__('iteration_hook', self.projection_set)
        # run PGD to predict actual estimates
        return train_model(self._log_reg, (S, None))


# define logistic distribution
base_distribution = Uniform(0, 1)
transforms_ = [SigmoidTransform().inv]
logistic = TransformedDistribution(base_distribution, transforms_)


class TruncatedBCE(ch.autograd.Function):
    @staticmethod
    def forward(ctx, pred, targ):
        ctx.save_for_backward(pred, targ)
        loss = ch.nn.BCEWithLogitsLoss()
        return loss(pred, targ)

    @staticmethod
    def backward(ctx, grad_output):
        pred, targ = ctx.saved_tensors
        stacked = pred[None, ...].repeat(config.args.num_samples, 1, 1)
        # add noise
        noised = stacked + logistic.sample(stacked.size())
        # filter
        filtered = ch.stack([config.args.phi(batch).unsqueeze(1) for batch in noised]).float()
        out = (noised * filtered).sum(dim=0) / (filtered.sum(dim=0) + config.args.eps)
        grad = ch.where(ch.abs(out) > config.args.eps, sig(out), targ) - targ
        return grad / pred.size(0), -grad / pred.size(0)


class TruncatedLogisticRegression:
    """
    Censored logistic regression projection set
    """
    def __init__(self, emp_log_reg):
        """
        Args:
            emp_log_reg (nn.Linear): empirical logistic regression weights and bias
        """
        self.emp_log_reg = emp_log_reg
        self.radius = config.args.radius * (ch.sqrt(2.0 * ch.log(1.0 / config.args.alpha)))
        if config.args.clamp:
            self.weight_bounds = Bounds((self.emp_log_reg.weight.data - config.args.radius).flatten(),
                                        (self.emp_log_reg.weight.data + config.args.radius).flatten())
            if self.emp_log_reg.bias:
                self.bias_bounds = Bounds(float(self.emp_log_reg.bias.data - config.args.radius),
                                          float(self.emp_log_reg.bias.data + config.args.radius))

    def __call__(self, est_log_reg, i, loop_type, inp, target):
        if config.args.clamp:
            # project weight coefficients
            est_log_reg.weight.data = ch.stack([ch.clamp(est_log_reg.weight.data[i], float(self.weight_bounds.lower[i]),
                                                         float(self.weight_bounds.upper[i])) for i in
                                                range(est_log_reg.weight.size(0))])
            # project bias coefficient
            if est_log_reg.bias:
                est_log_reg.bias.data = ch.clamp(est_log_reg.bias, self.bias_bounds.lower, self.bias_bounds.upper).reshape(
                    est_log_reg.bias.size())
        else:
            pass