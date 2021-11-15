"""
Gradients for truncated and untruncated latent variable models. 
"""
import torch as ch
from torch import Tensor
from torch import sigmoid as sig
from torch.distributions import Uniform, Gumbel, MultivariateNormal
from torch.distributions.transforms import SigmoidTransform
from torch.distributions.transformed_distribution import TransformedDistribution
import math
import config

from .utils.helpers import logistic, censored_sample_nll


class CensoredMultivariateNormalNLL(ch.autograd.Function):
    """
    Computes the negative population log likelihood for censored multivariate normal distribution.
    """
    @staticmethod
    def forward(ctx, v, T, S, S_grad, phi):
        # reparameterize distribution
        sigma = T.inverse()
        mu = sigma@v[...,None].flatten()
        # sigma = T
        # mu = v
        # import pdb; pdb.set_trace()
        # print("mu: {}".format(mu))
        # print("cov: {}".format(sigma))
            
        # s = MultivariateNormal(mu, sigma).rsample(ch.Size([config.args.num_samples, S.size(0)]))
        # reparameterize distribution
        z = Tensor([])
        M = MultivariateNormal(mu, sigma)
        while z.size(0) < S.size(0):
            s = M.sample(sample_shape=ch.Size([config.args.num_samples, ]))
            z = ch.cat([z, s[phi(s).flatten().nonzero().flatten()]])
        z = z[:S.size(0)]
       
        '''
        filtered = phi(y)

        print(filtered.sum(0))
'''
        first_term = .5 * ch.bmm((S@T).view(S.size(0), 1, S.size(1)), S.view(S.size(0), S.size(1), 1)).squeeze(-1) - S@v[None,...]

        second_term = (-.5 * ch.bmm((z@T).view(z.size(0), 1, z.size(1)), z.view(z.size(0), z.size(1), 1)).squeeze(-1) + z@v[None,...]).mean(0)
        
        ctx.save_for_backward(S_grad, z)
        return (first_term + second_term).mean(0)


    @staticmethod
    def backward(ctx, grad_output):
        S_grad, z = ctx.saved_tensors
        '''
        # reparameterize distribution
        T = covariance_matrix.inverse()
        v = T.matmul(loc.unsqueeze(1)).flatten()
        # rejection sampling
        y = Tensor([])
        M = MultivariateNormal(v, T)
        while y.size(0) < x.size(0):
            s = M.sample(sample_shape=ch.Size([config.args.num_samples, ]))
            y = ch.cat([y, s[ctx.phi(s).nonzero(as_tuple=False).flatten()]])
        '''
        # calculate gradient
        # import pdb; pdb.set_trace()
        grad = (-S_grad + censored_sample_nll(z)).mean(0)
        return grad[z.size(1) ** 2:], grad[:z.size(1) ** 2].reshape(z.size(1), z.size(1)), None, None, None


class TruncatedMultivariateNormalNLL(ch.autograd.Function):
    """
    Computes the negative population log likelihood for truncated multivariate normal distribution with unknown truncation.
    """
    @staticmethod
    def forward(ctx, u, B, x, loc_grad, cov_grad, phi, exp_h):
        ctx.save_for_backward(u, B, x, loc_grad, cov_grad)
        ctx.phi = phi
        ctx.exp_h = exp_h
        return ch.zeros(1)

    @staticmethod
    def backward(ctx, grad_output):
        u, B, x, loc_grad, cov_grad = ctx.saved_tensors
        exp = ctx.exp_h(u, B, x)
        psi = ctx.phi.psi_k(x).unsqueeze(1)
        return (loc_grad * exp * psi).mean(0), ((cov_grad.flatten(1) * exp * psi).unflatten(1, B.size())).mean(
            0), None, None, None, None, None


class TruncatedMSE(ch.autograd.Function):
    """
    Computes the gradient of the negative population log likelihood for censored regression
    with known noise variance.
    """
    @staticmethod
    def forward(ctx, pred, targ, phi):
        # make args.num_samples copies of pred, N x B x 1
        stacked = pred[None, ...].repeat(config.args.num_samples, 1, 1)
        # add random noise to each copy
        noised = stacked + math.sqrt(config.args.noise_var) * ch.randn(stacked.size()).to(config.args.device)
        # filter out copies where pred is in bounds
        filtered = phi(noised)
        # average across truncated indices
        z = (filtered * noised).sum(dim=0) / (filtered.sum(dim=0) + config.args.eps)
        z_2 =  (filtered * noised.pow(2)).sum(dim=0) / (filtered.sum(dim=0) + config.args.eps)

        ctx.save_for_backward(pred, targ, z)
        return (-.5 * targ.pow(2) + targ * pred + .5 * z_2 - z * pred).mean(0)

    @staticmethod
    def backward(ctx, grad_output):
        pred, targ, z = ctx.saved_tensors
        '''
        # make args.num_samples copies of pred, N x B x 1
        stacked = pred[None, ...].repeat(config.args.num_samples, 1, 1)
        # add random noise to each copy
        noised = stacked + math.sqrt(config.args.noise_var) * ch.randn(stacked.size()).to(config.args.device)
        # filter out copies where pred is in bounds
        filtered = ctx.phi(noised)
        # average across truncated indices
        out = (filtered * noised).sum(dim=0) / (filtered.sum(dim=0) + config.args.eps)
        '''
        return (z - targ) / pred.size(0), targ / pred.size(0), None


class TruncatedUnknownVarianceMSE(ch.autograd.Function):
    """
    Computes the gradient of negative population log likelihood for truncated linear regression
    with unknown noise variance.
    """
    @staticmethod
    def forward(ctx, pred, targ, lambda_, phi):
        # calculate std deviation of noise distribution estimate
        sigma = ch.sqrt(lambda_.inverse())
        stacked = pred[None, ...].repeat(config.args.num_samples, 1, 1)
        # add noise to regression predictions
        noised = stacked + sigma * ch.randn(stacked.size()).to(config.args.device)
        # filter out copies that fall outside of truncation set
        filtered = phi(noised)
        out = noised * filtered
        z = out.sum(dim=0) / (filtered.sum(dim=0) + config.args.eps)

        '''
        out = z.sum(dim=0) / (filtered.sum(dim=0) + config.args.eps)
        '''
        z_2 = out.pow(2).sum(dim=0) / (filtered.sum(dim=0) + config.args.eps)

        ctx.save_for_backward(pred, targ, lambda_, z, z_2)

        return (-0.5 * lambda_ * targ.pow(2)  + lambda_ * targ * pred + 0.5 * lambda_ * z_2 - lambda_ * z * pred).mean(0) 

    @staticmethod
    def backward(ctx, grad_output):
        pred, targ, lambda_, z, z_2 = ctx.saved_tensors
        '''
        # calculate std deviation of noise distribution estimate
        sigma = ch.sqrt(lambda_.inverse())
        stacked = pred[None, ...].repeat(config.args.num_samples, 1, 1)
        # add noise to regression predictions
        noised = stacked + sigma * ch.randn(stacked.size()).to(config.args.device)
        # filter out copies that fall outside of truncation set
        filtered = ctx.phi(noised)
        z = noised * filtered
        '''
        """
        multiply the v gradient by lambda, because autograd computes 
        v_grad*x*variance, thus need v_grad*(1/variance) to cancel variance
        factor
        """
        '''
        out = z.sum(dim=0) / (filtered.sum(dim=0) + config.args.eps)
        out_2 = z.pow(2).sum(dim=0) / (filtered.sum(dim=0) + config.args.eps)
        '''
        lambda_grad = .5 * (targ.pow(2) - z_2)
        return lambda_ * (z - targ) / pred.size(0), targ / pred.size(0), lambda_grad / pred.size(0), None


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
    def forward(ctx, pred, targ, phi):
        ctx.save_for_backward(pred, targ)
        ctx.phi = phi
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
        filtered = ctx.phi(noised)
        out = (noised * filtered).sum(dim=0) / (filtered.sum(dim=0) + 1e-5)
        grad = ch.where(ch.abs(out) > 1e-5, sig(out), targ) - targ
        return grad / pred.size(0), -grad / pred.size(0), None


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
    def forward(ctx, pred, targ, phi):
        ctx.save_for_backward(pred, targ)
        ctx.phi = phi
        ce_loss = ch.nn.CrossEntropyLoss()
        return ce_loss(pred, targ)

    @staticmethod
    def backward(ctx, grad_output):  
        pred, targ = ctx.saved_tensors
        # initialize gumbel distribution
        gumbel = Gumbel(0, 1)
        # make num_samples copies of pred logits
        stacked = pred[None, ...].repeat(config.args.num_samples, 1, 1)
        # add gumbel noise to logits
        rand_noise = gumbel.sample(stacked.size()).to(config.args.device)
        noised = (stacked) + rand_noise 
        # truncate - if one of the noisy logits does not fall within the truncation set, remove it
        filtered = ctx.phi(stacked)[..., None].to(config.args.device)
        noised_labs = noised.argmax(-1)
        # mask takes care of invalid logits and truncation set
        mask = noised_labs.eq(targ)[..., None]
        inner_exp = (1 - ch.exp(-rand_noise))
        avg = (((inner_exp * mask).sum(0) / ((mask).sum(0) + 1e-5)) - ((inner_exp * filtered).sum(0) / (filtered.sum(0) + 1e-5))) / pred.size(0)       
        return -avg, None, None
