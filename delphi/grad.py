"""
Gradients for truncated and untruncated latent variable models. 
"""
import torch as ch
from torch import Tensor
from torch import sigmoid as sig
from torch.distributions import Uniform, Gumbel, Laplace
from torch.distributions.transforms import SigmoidTransform
from torch.distributions.transformed_distribution import TransformedDistribution
import config

from .utils.helpers import logistic


class CensoredMultivariateNormalNLL(ch.autograd.Function):
    """
    Computes the negative population log likelihood for censored multivariate normal distribution.
    """

    @staticmethod
    def forward(ctx, loc, covariance_matrix, x):
        ctx.save_for_backward(loc, covariance_matrix, x)
        return ch.zeros(1)

    @staticmethod
    def backward(ctx, grad_output):
        loc, covariance_matrix, x = ctx.saved_tensors
        # reparameterize distribution
        T = covariance_matrix.inverse()
        v = T.matmul(loc.unsqueeze(1)).flatten()
        # rejection sampling
        y = Tensor([])
        M = MultivariateNormal(v, T)
        while y.size(0) < x.size(0):
            s = M.sample(sample_shape=ch.Size([config.args.num_samples, ]))
            y = ch.cat([y, s[config.args.phi(s).nonzero(as_tuple=False).flatten()]])
        # calculate gradient
        grad = (-x + censored_sample_nll(y[:x.size(0)])).mean(0)
        return grad[loc.size(0) ** 2:], grad[:loc.size(0) ** 2].reshape(covariance_matrix.size()), None


class TruncatedMultivariateNormalNLL(ch.autograd.Function):
    """
    Computes the negative population log likelihood for truncated multivariate normal distribution with unknown truncation.
    """

    @staticmethod
    def forward(ctx, u, B, x, loc_grad, cov_grad):
        ctx.save_for_backward(u, B, x, loc_grad, cov_grad)
        return ch.ones(1)

    @staticmethod
    def backward(ctx, grad_output):
        u, B, x, loc_grad, cov_grad = ctx.saved_tensors
        exp = config.args.exp_h(u, B, x)
        psi = config.args.phi.psi_k(x).unsqueeze(1)
        return (loc_grad * exp * psi).mean(0), ((cov_grad.flatten(1) * exp * psi).unflatten(1, B.size())).mean(
            0), None, None, None


class TruncatedMSE(ch.autograd.Function):
    """
    Computes the gradient of the negative population log likelihood for censored regression
    with known noise variance.
    """
    @staticmethod
    def forward(ctx, pred, targ, phi):
        ctx.save_for_backward(pred, targ)
        ctx.phi = phi
        return 0.5 * (pred.float() - targ.float()).pow(2).mean(0)

    @staticmethod
    def backward(ctx, grad_output):
        # import pdb; pdb.set_trace()
        pred, targ = ctx.saved_tensors
        # make args.num_samples copies of pred, N x B x 1
        stacked = pred[None, ...].repeat(config.args.num_samples, 1, 1)
        # add random noise to each copy
        noised = stacked + ch.randn(stacked.size()).to(config.args.device)
        # filter out copies where pred is in bounds
        filtered = ctx.phi(noised)
        # average across truncated indices
        out = (filtered * noised).sum(dim=0) / (filtered.sum(dim=0) + config.args.eps)
        return (out - targ) / pred.size(0), targ / pred.size(0), None


class TruncatedUnknownVarianceMSE(ch.autograd.Function):
    """
    Computes the gradient of negative population log likelihood for truncated linear regression
    with unknown noise variance.
    """
    @staticmethod
    def forward(ctx, pred, targ, lambda_, phi):
        ctx.save_for_backward(pred, targ, lambda_)
        ctx.phi = phi
        return 0.5 * (pred.float() - targ.float()).pow(2).mean(0)

    @staticmethod
    def backward(ctx, grad_output):
        pred, targ, lambda_ = ctx.saved_tensors
        # calculate std deviation of noise distribution estimate
        sigma = ch.sqrt(lambda_.inverse())
        stacked = pred[None, ...].repeat(config.args.num_samples, 1, 1)
        # add noise to regression predictions
        noised = stacked + sigma * ch.randn(stacked.size()).to(config.args.device)
        # filter out copies that fall outside of truncation set
        filtered = ctx.phi(noised)
        z = noised * filtered
        lambda_grad = .5 * (targ.pow(2) - (z.pow(2).sum(dim=0) / (filtered.sum(dim=0) + config.args.eps)))
        """
        multiply the v gradient by lambda, because autograd computes 
        v_grad*x*variance, thus need v_grad*(1/variance) to cancel variance
        factor
        """
        out = z.sum(dim=0) / (filtered.sum(dim=0) + config.args.eps)
        return lambda_ * (out - targ) / pred.size(0), targ / pred.size(0), lambda_grad / pred.size(0), None


class LogisticBCE(ch.autograd.Function):
    @staticmethod
    def forward(ctx, pred, targ):
        ctx.save_for_backward(pred, targ)
        loss = ch.nn.BCEWithLogitsLoss()
        return loss(pred, targ)

    @staticmethod
    def backward(ctx, grad_output):
        pred, targ = ctx.saved_tensors
        stacked = pred[None, ...].repeat(config.args.num_samples, 1, 1)
        rand_noise = logistic.sample(stacked.size())
        # add noise
        noised = stacked + rand_noise
        noised_labs = noised > 0
        # filter
        mask = (noised_labs).eq(targ)
        avg = 1 - 2*((sig(rand_noise)*mask).sum(0) / (mask.sum(0) + 1e-5))
        return avg, None


class TruncatedBCE(ch.autograd.Function):
    @staticmethod
    def forward(ctx, pred, targ):
        ctx.save_for_backward(pred, targ)
        loss = ch.nn.BCEWithLogitsLoss()
        return loss(pred, targ)

    @staticmethod
    def backward(ctx, grad_output):
        pred, targ = ctx.saved_tensors

        # logistic distribution
        base_distribution = Uniform(0, 1)
        transforms_ = [SigmoidTransform().inv]
        logistic = TransformedDistribution(base_distribution, transforms_)

        stacked = pred[None, ...].repeat(config.args.num_samples, 1, 1)
        # add noise
        noised = stacked + logistic.sample(stacked.size())
        # filter
        filtered = config.args.phi(noised).unsqueeze(-1)
        out = (noised * filtered).sum(dim=0) / (filtered.sum(dim=0) + 1e-5)
        grad = ch.where(ch.abs(out) > 1e-5, sig(out), targ) - targ
        return grad / pred.size(0), -grad / pred.size(0)


class GumbelCE(ch.autograd.Function):
    @staticmethod
    def forward(ctx, pred, targ):
        ctx.save_for_backward(pred, targ)
        ce_loss = ch.nn.CrossEntropyLoss()
        return ce_loss(pred, targ)

    @staticmethod
    def backward(ctx, grad_output):
        pred, targ = ctx.saved_tensors
        # gumbel distribution
        gumbel = Gumbel(0, 1)
        # make num_samples copies of pred logits
        stacked = pred[None, ...].repeat(config.args.num_samples, 1, 1)
        # add gumbel noise to logits
        rand_noise = gumbel.sample(stacked.size()).to(config.args.device)
        noised = stacked + rand_noise 
        noised_labs = noised.argmax(-1)
        # remove the logits from the trials, where the kth logit is not the largest value
        mask = noised_labs.eq(targ)[..., None]
        inner_exp = 1 - ch.exp(-rand_noise)
        avg = (inner_exp * mask).sum(0) / (mask.sum(0) + 1e-5) / pred.size(0)
        return -avg , None


class TruncatedCE(ch.autograd.Function):
    @staticmethod
    def forward(ctx, pred, targ):
        ctx.save_for_backward(pred, targ)
        ce_loss = ch.nn.CrossEntropyLoss()
        return ce_loss(pred, targ)

    @staticmethod
    def backward(ctx, grad_output):  
        # import pdb; pdb.set_trace()
        pred, targ = ctx.saved_tensors
        # initialize gumbel distribution
        gumbel = Gumbel(0, 1)
        # make num_samples copies of pred logits
        stacked = pred[None, ...].repeat(config.args.num_samples, 1, 1)
        # add gumbel noise to logits
        rand_noise = gumbel.sample(stacked.size()).to(config.args.device)
        noised = stacked + rand_noise 
        # truncate - if one of the noisy logits does not fall within the truncation set, remove it
        filtered = config.args.phi(noised)[..., None].to(config.args.device)
        noised_labs = noised.argmax(-1)
        # mask takes care of invalid logits and truncation set
        mask = noised_labs.eq(targ)[..., None]
        inner_exp = (1 - ch.exp(-rand_noise))
        avg = (((inner_exp * mask * filtered).sum(0) / ((mask * filtered).sum(0) + 1e-5)) - ((inner_exp * filtered).sum(0) / (filtered.sum(0) + 1e-5))) / pred.size(0)            
        return -avg, None
