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
    Computes the truncated negative population log likelihood for censored multivariate normal distribution. 
    Function calculates the truncated negative log likelihood in the forward method and then calculates the 
    gradients with respect mu and cov in the backward method. When sampling from the conditional distribution, 
    we sample batch_size * num_samples samples, we then filter out the samples that remain in the truncation set, 
    and retainup to batch_size of the filtered samples. If there are fewer than batch_size number of samples remain,
    we provide a vector of zeros and calculate the untruncated log likelihood. 
    """
    @staticmethod
    def forward(ctx, v, T, S, S_grad, phi, num_samples=10, eps=1e-5):
        """
        Args: 
            v (torch.Tensor): reparameterize mean estimate (cov^(-1) * mu)
            T (torch.Tensor): square reparameterized (cov^(-1)) covariance matrix with dim d
            S (torch.Tensor): batch_size * dims, sample batch 
            S_grad (torch.Tenosr): batch_size * (dims + dim * dims) gradient for batch
            phi (delphi.oracle): oracle for censored distribution
            num_samples (int): number of samples to sample for each sample in batch
        """
        # reparameterize distribution
        sigma = T.inverse()
        mu = (sigma@v).flatten()
        # reparameterize distribution
        M = MultivariateNormal(mu, sigma)
        # sample num_samples * batch size samples from distribution
        s = M.sample([num_samples * S.size(0)])
        filtered = phi(s).nonzero(as_tuple=True)
        """
        TODO: see if there is a better way to do this
        """
        # z is a tensor of size batch size zeros, then fill with up to batch size num samples
        z = ch.zeros(S.size())
        elts = s[filtered][:S.size(0)]
        print("elts: ", elts.size(0))
        z[:elts.size(0)] = elts
        # standard negative log likelihood
        nll = .5 * ch.bmm((S@T).view(S.size(0), 1, S.size(1)), S.view(S.size(0), S.size(1), 1)).squeeze(-1) - S@v[None,...].T
        # normalizing constant for nll
        norm_const = -.5 * ch.bmm((z@T).view(z.size(0), 1, z.size(1)), z.view(z.size(0), z.size(1), 1)).squeeze(-1) + z@v[None,...].T
        ctx.save_for_backward(S_grad, z)
        return (nll + norm_const).mean(0)

    @staticmethod
    def backward(ctx, grad_output):
        S_grad, z = ctx.saved_tensors
        # calculate gradient
        grad = -S_grad + censored_sample_nll(z)
        return grad[:,z.size(1) ** 2:] / z.size(0), (grad[:,:z.size(1) ** 2] / z.size(0)).view(-1, z.size(1), z.size(1)), None, None, None, None, None


class TruncatedMultivariateNormalNLL(ch.autograd.Function):
    """
    Computes the negative population log likelihood for truncated multivariate normal distribution with unknown truncation.
    """
    @staticmethod
    def forward(ctx, u, B, x, pdf, loc_grad, cov_grad, phi, exp_h):
        exp = exp_h(u, B, x)
        psi = phi.psi_k(x)
        loss = exp * pdf * psi
        ctx.save_for_backward(loss, loc_grad, cov_grad)
        return loss.mean(0)

    @staticmethod
    def backward(ctx, grad_output):
        loss, loc_grad, cov_grad = ctx.saved_tensors
        return (loc_grad * loss).mean(0), ((cov_grad.flatten(1) * loss).unflatten(1, ch.Size([loc_grad.size(1), loc_grad.size(1)]))).mean(
            0), None, None, None, None, None, None


class TruncatedMSE(ch.autograd.Function):
    """
    Computes the gradient of the negative population log likelihood for censored regression
    with known noise variance.
    """
    @staticmethod
    def forward(ctx, pred, targ, phi, noise_var, num_samples=10, eps=1e-5):
        # make num_samples copies of pred, N x B x 1
        stacked = pred[None, ...].repeat(num_samples, 1, 1)
        # add random noise to each copy
        noised = stacked + math.sqrt(noise_var) * ch.randn(stacked.size())        
        # filter out copies where pred is in bounds
        filtered = phi(noised)
        # average across truncated indices
        z = (filtered * noised).sum(dim=0) / (filtered.sum(dim=0) + eps)
        z_2 =  -.5 * (filtered * noised.pow(2)).sum(dim=0) / (filtered.sum(dim=0) + eps)
        out = ((-.5 * noised.pow(2) + noised * pred) * filtered).sum(dim=0) / (filtered.sum(dim=0) + eps)

        ctx.save_for_backward(pred, targ, z)
        return (-.5 * targ.pow(2) + targ * pred - out).mean(0)

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
        return (z - targ) / pred.size(0), targ / pred.size(0), None, None, None, None


class TruncatedUnknownVarianceMSE(ch.autograd.Function):
    """
    Computes the gradient of negative population log likelihood for truncated linear regression
    with unknown noise variance.
    """
    @staticmethod
    def forward(ctx, pred, targ, lambda_, phi, num_samples=10, eps=1e-5):
        # calculate std deviation of noise distribution estimate
        sigma = ch.sqrt(lambda_.inverse())
        # stacked = pred[None, ...].repeat(num_samples, 1, 1)
        stacked = pred[..., None].repeat(1, num_samples, 1)

        # add noise to regression predictions
        noised = stacked + sigma * ch.randn(stacked.size())
        # filter out copies that fall outside of truncation set
        filtered = phi(noised)
        out = noised * filtered
        # import pdb; pdb.set_trace()
        z = out.sum(dim=1) / (filtered.sum(dim=1) + eps)
        # z = out.sum(dim=0) / (filtered.sum(dim=0) + eps)
        z_2 = Tensor([])
        for i in range(filtered.size(0)):
            z_2 = ch.cat([z_2, noised[i][filtered[i].squeeze(-1).sort(descending=True).indices[0]].pow(2)[None,...]])
        z_2_ = out.pow(2).sum(dim=1) / (filtered.sum(dim=1) + eps)
        nll = -0.5 * lambda_ * targ.pow(2)  + lambda_ * targ * pred
        const = -0.5 * lambda_ * z_2 + z * pred * lambda_

        ctx.save_for_backward(pred, targ, lambda_, z, z_2, z_2_)
        return nll - const

#        return (0.5 * lambda_ * targ.pow(2)  - lambda_ * targ * pred - 0.5 * lambda_ * z_2 + lambda_ * z * pred).mean(0) 

    @staticmethod
    def backward(ctx, grad_output):
        pred, targ, lambda_, z, z_2, z_2_ = ctx.saved_tensors
        """
        multiply the v gradient by lambda, because autograd computes 
        v_grad*x*variance, thus need v_grad*(1/variance) to cancel variance
        factor
        """
        lambda_grad = .5 * (targ.pow(2) - z_2)
        lambda_grad_ = .5 * (targ.pow(2) - z_2_)

        print("lambda grad: ", lambda_grad.mean())
        print("lambda grad 2: ", lambda_grad_.mean())
        print("y grad: ", (lambda_ * (z - targ)).mean())
    
        return lambda_ * (z - targ) / pred.size(0), targ / pred.size(0), lambda_grad / pred.size(0), None, None, None


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
        avg = 1 - 2*((sig(rand_noise) * mask).sum(0) / (mask.sum(0) + 1e-5))
        return avg, None


class TruncatedBCE(ch.autograd.Function):
    @staticmethod
    def forward(ctx, pred, targ, phi, num_samples=10, eps=1e-5):
        ctx.save_for_backward(pred, targ)
        ctx.phi = phi
        ctx.eps = eps
        ctx.num_samples = num_samples
        loss = ch.nn.BCEWithLogitsLoss()
        return loss(pred, targ)

    @staticmethod
    def backward(ctx, grad_output):
        pred, targ = ctx.saved_tensors
        
        # logistic distribution
        base_distribution = Uniform(0, 1)
        transforms_ = [SigmoidTransform().inv]
        logistic = TransformedDistribution(base_distribution, transforms_)

        stacked = pred[None, ...].repeat(ctx.num_samples, 1, 1)
        # add noise
        noised = stacked + logistic.sample(stacked.size())
        # filter
        filtered = ctx.phi(noised)
        out = (noised * filtered).sum(dim=0) / (filtered.sum(dim=0) + ctx.eps)
        grad = ch.where(ch.abs(out) > 1e-5, sig(out), targ) - targ
        return grad / pred.size(0), -grad / pred.size(0), None, None, None


class GumbelCE(ch.autograd.Function):
    @staticmethod
    def forward(ctx, pred, targ, num_samples=1000, eps=1e-5):
        ctx.save_for_backward(pred, targ)
        ce_loss = ch.nn.CrossEntropyLoss()
        ctx.num_samples = num_samples
        ctx.eps = eps
        return ce_loss(pred, targ)

    @staticmethod
    def backward(ctx, grad_output):
        pred, targ = ctx.saved_tensors
        # gumbel distribution
        gumbel = Gumbel(0, 1)
        # make num_samples copies of pred logits
        stacked = pred[None, ...].repeat(ctx.num_samples, 1, 1)
        # add gumbel noise to logits
        rand_noise = gumbel.sample(stacked.size())
        noised = stacked + rand_noise 
        noised_labs = noised.argmax(-1)
        # remove the logits from the trials, where the kth logit is not the largest value
        mask = noised_labs.eq(targ)[..., None]
        inner_exp = 1 - ch.exp(-rand_noise)
        avg = (inner_exp * mask).sum(0) / (mask.sum(0) + ctx.eps) / pred.size(0)
        return -avg , None, None, None


class TruncatedCE(ch.autograd.Function):
    @staticmethod
    def forward(ctx, pred, targ, phi, num_samples=1000, eps=1e-5):
        ctx.save_for_backward(pred, targ)
        ctx.phi = phi
        ctx.num_samples = num_samples
        ctx.eps = eps
        ce_loss = ch.nn.CrossEntropyLoss()
        return ce_loss(pred, targ)

    @staticmethod
    def backward(ctx, grad_output):  
        pred, targ = ctx.saved_tensors
        # initialize gumbel distribution
        gumbel = Gumbel(0, 1)
        # make num_samples copies of pred logits
        stacked = pred[None, ...].repeat(ctx.num_samples, 1, 1)
        # add gumbel noise to logits
        rand_noise = gumbel.sample(stacked.size())
        noised = stacked + rand_noise 
        # truncate - if one of the noisy logits does not fall within the truncation set, remove it
        filtered = ctx.phi(noised)
        noised_labs = noised.argmax(-1)
        # mask takes care of invalid logits and truncation set
        mask = noised_labs.eq(targ)[..., None] * filtered
        inner_exp = (1 - ch.exp(-rand_noise))
        avg = (((inner_exp * mask).sum(0) / ((mask).sum(0) + 1e-5)) - ((inner_exp * filtered).sum(0) / (filtered.sum(0) + ctx.eps))) / pred.size(0)
        return -avg, None, None, None, None
