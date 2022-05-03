"""
Gradients for truncated and untruncated latent variable models. 
"""

import torch as ch
from torch import sigmoid as sig
from torch.nn import Softmax
from torch.distributions import Gumbel, MultivariateNormal, Bernoulli
import math

from .utils.helpers import logistic, censored_sample_nll

softmax = Softmax(dim=1)


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
        if elts.dim() == 1: elts = elts[...,None]
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
    Calculates the population log-likelihood for the current batch in the forward step, and 
    then calculates its gradient in the backwards step.
    """
    @staticmethod
    def forward(ctx, u, B, x, pdf, loc_grad, cov_grad, phi, exp_h):
        """
        Args: 
            u (torch.Tensor): size (dims,) - current reparameterized mean estimate
            B (torch.Tensor): size (dims, dims) - current reparameterized covariance matrix estimate
            x (torch.Tensor): size (batch_size, dims) - batch of dataset samples 
            pdf (torch.Tensor): size (batch_size, 1) - batch of pdf for dataset samples
            loc_grad (torch.Tensor): (batch_size, dims) - precomputed gradient for mean for batch
            cov_grad (torch.Tensor): (batch_size, dims * dims) - precomputed gradient for covariance matrix for batch 
            phi (oracle.UnknownGaussian): oracle object for learning truncation set 
            exp_h (Exp_h): helper class object for calculating exponential in the gradient
        """
        exp = exp_h(u, B, x)
        psi = phi.psi_k(x)
        loss = exp * pdf * psi
        ctx.save_for_backward(loss, loc_grad, cov_grad)
        return loss / x.size(0)

    @staticmethod
    def backward(ctx, grad_output):
        loss, loc_grad, cov_grad = ctx.saved_tensors
        return (loc_grad * loss) / loc_grad.size(0), ((cov_grad.flatten(1) * loss).unflatten(1, ch.Size([loc_grad.size(1), loc_grad.size(1)]))) / loc_grad.size(0), None, None, None, None, None, None


class TruncatedMSE(ch.autograd.Function):
    """
    Computes the gradient of the negative population log likelihood for truncated regression
    with known noise variance.
    """
    @staticmethod
    def forward(ctx, pred, targ, phi, noise_var, num_samples=10, eps=1e-5):
        """
        Args: 
            pred (torch.Tensor): size (batch_size, 1) matrix for regression model predictions
            targ (torch.Tensor): size (batch_size, 1) matrix for regression target predictions
            phi (oracle.oracle): dependent variable membership oracle
            noise_var (float): noise distribution variance parameter
            num_samples (int): number of samples to generate per sample in batch in rejection sampling procedure
            eps (float): denominator error constant to avoid divide by zero errors
        """
        # make num_samples copies of pred, N x B x 1
        stacked = pred[None, ...].repeat(num_samples, 1, 1)
        # add random noise to each copy
        noised = stacked + math.sqrt(noise_var) * ch.randn(stacked.size())        
        # filter out copies where pred is in bounds
        filtered = phi(noised)
        # average across truncated indices
        z = (filtered * noised).sum(dim=0) / (filtered.sum(dim=0) + eps)
        out = ((-.5 * noised.pow(2) + noised * pred) * filtered).sum(dim=0) / (filtered.sum(dim=0) + eps)

        ctx.save_for_backward(pred, targ, z)
        return (-.5 * targ.pow(2) + targ * pred - out).mean(0)

    @staticmethod
    def backward(ctx, grad_output):
        pred, targ, z = ctx.saved_tensors
        return (z - targ) / pred.size(0), targ / pred.size(0), None, None, None, None


class TruncatedUnknownVarianceMSE(ch.autograd.Function):
    """
    Computes the gradient of negative population log likelihood for truncated linear regression
    with unknown noise variance.
    """
    @staticmethod
    def forward(ctx, pred, targ, lambda_, phi, num_samples=10, eps=1e-5):
        """
        Args: 
            pred (torch.Tensor): size (batch_size, 1) matrix for regression model predictions
            targ (torch.Tensor): size (batch_size, 1) matrix for regression target predictions
            lambda_ (float): current reparameterized variance estimate for noise distribution
            phi (oracle.oracle): dependent variable membership oracle
            num_samples (int): number of samples to generate per sample in batch in rejection sampling procedure
            eps (float): denominator error constant to avoid divide by zero errors
        """
        # calculate std deviation of noise distribution estimate
        sigma = ch.sqrt(lambda_.inverse())
        stacked = pred[..., None].repeat(1, num_samples, 1)

        # add noise to regression predictions
        noised = stacked + sigma * ch.randn(stacked.size())
        # filter out copies that fall outside of truncation set
        filtered = phi(noised)
        out = noised * filtered
        z = out.sum(dim=1) / (filtered.sum(dim=1) + eps)
        z_2 = out.pow(2).sum(dim=1) / (filtered.sum(dim=1) + eps)
        nll = -0.5 * lambda_ * targ.pow(2)  + lambda_ * targ * pred
        const = -0.5 * lambda_ * z_2 + z * pred * lambda_

        ctx.save_for_backward(pred, targ, lambda_, z, z_2)
        return (nll - const).mean(0)

    @staticmethod
    def backward(ctx, grad_output):
        pred, targ, lambda_, z, z_2 = ctx.saved_tensors
        """
        multiply the v gradient by lambda, because autograd computes 
        v_grad*x*variance, thus need v_grad*(1/variance) to cancel variance
        factor
        """
        lambda_grad = .5 * (targ.pow(2) - z_2)
        return lambda_ * (z - targ) / pred.size(0), targ / pred.size(0), lambda_grad / pred.size(0), None, None, None


class TruncatedBCE(ch.autograd.Function):
    """
    Truncated binary cross entropy gradient for truncated binary classification tasks. 
    """
    @staticmethod
    def forward(ctx, pred, targ, phi, num_samples=10, eps=1e-5):
        """
        Args: 
            pred (torch.Tensor): size (batch_size, 1) matrix for regression model predictions
            targ (torch.Tensor): size (batch_size, 1) matrix for regression target predictions
            phi (oracle.oracle): dependent variable membership oracle
            num_samples (int): number of samples to generate per sample in batch in rejection sampling procedure
            eps (float): denominator error constant to avoid divide by zero errors
        """
        ctx.save_for_backward()
        bce_loss = ch.nn.BCEWithLogitsLoss()
        
        stacked = pred[None, ...].repeat(num_samples, 1, 1)
        rand_noise = logistic.sample(stacked.size())
        # add noise
        noised = stacked + rand_noise
        noised_labs = noised >= 0
        # filter
        filtered = phi(noised)
        mask = (noised_labs).eq(targ)
        nll = (filtered * mask * logistic.log_prob(rand_noise)).sum(0) / ((filtered * mask).sum(0) + eps)
        const = (filtered * logistic.log_prob(rand_noise)).sum(0) / (filtered.sum(0) + eps)
        ctx.save_for_backward(mask, filtered, rand_noise)
        ctx.eps = eps
        return -(nll - const) / pred.size(0)
       # return bce_loss(pred, targ)

    @staticmethod
    def backward(ctx, grad_output):
        mask, filtered, rand_noise = ctx.saved_tensors

        avg = 2*(sig(rand_noise) * mask * filtered).sum(0) / ((mask * filtered).sum(0) + ctx.eps) 
        norm_const = (2 * sig(rand_noise) * filtered).sum(0) / (filtered.sum(0) + ctx.eps)
        return -(avg - norm_const) / rand_noise.size(1), None, None, None, None


class TruncatedProbitMLE(ch.autograd.Function): 
    @staticmethod
    def forward(ctx, pred, targ, phi, num_samples=10, eps=1e-5): 
        """
        Args: 
            pred (torch.Tensor): size (batch_size, 1) matrix for regression model predictions
            targ (torch.Tensor): size (batch_size, 1) matrix for regression target predictions
            phi (oracle.oracle): dependent variable membership oracle
            num_samples (int): number of samples to generate per sample in batch in rejection sampling procedure
            eps (float): denominator error constant to avoid divide by zero errors
        """
#        import pdb; pdb.set_trace()
        M = MultivariateNormal(ch.zeros(1,), ch.eye(1, 1))
        stacked = pred[None,...].repeat(num_samples, 1, 1)
        rand_noise = ch.randn(stacked.size())
        noised = stacked + rand_noise 
        noised_labs = noised >= 0
        mask = noised_labs.eq(targ)
        filtered = phi(noised)
        pdf = M.log_prob(rand_noise)[...,None]
        nll = (pdf * filtered * mask).sum(0) / ((filtered * mask).sum(0) + eps)
        const = (filtered * pdf).sum(0) / (filtered.sum(0) + eps)
        ctx.save_for_backward(rand_noise, filtered, mask)
        ctx.eps = eps
        return -(nll - const) / pred.size(0)

    @staticmethod
    def backward(ctx, grad_output): 
        rand_noise, filtered, mask = ctx.saved_tensors
        nll = (mask * filtered * rand_noise).sum(dim=0) / ((mask * filtered).sum(dim=0) + ctx.eps)
        const = (rand_noise * filtered).sum(dim=0) / (filtered.sum(dim=0) + ctx.eps)
        return -(nll - const) / rand_noise.size(1), None, None, None, None


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
        """
        Args: 
            pred (torch.Tensor): size (batch_size, 1) matrix for regression model predictions
            targ (torch.Tensor): size (batch_size, 1) matrix for regression target predictions
            phi (oracle.oracle): dependent variable membership oracle
            num_samples (int): number of samples to generate per sample in batch in rejection sampling procedure
            eps (float): denominator error constant to avoid divide by zero errors
        """
        ctx.save_for_backward(pred, targ)
        ctx.phi = phi
        ctx.num_samples = num_samples
        ctx.eps = eps
        ce_loss = ch.nn.CrossEntropyLoss()
        return ce_loss(pred, targ)
        '''
        # initialize gumbel distribution
        gumbel = Gumbel(0, 1)
        # make num_samples copies of pred logits
        stacked = pred[None, ...].repeat(num_samples, 1, 1)
        # add gumbel noise to logits
        rand_noise = gumbel.sample(stacked.size())
        noised = stacked + rand_noise 
        # truncate - if one of the noisy logits does not fall within the truncation set, remove it
        filtered = phi(noised)
        noised_labs = noised.argmax(-1)
        # mask takes care of invalid logits and truncation set
        mask = noised_labs.eq(targ)[..., None] * filtered
        
        nll = (gumbel.log_prob(rand_noise) * mask).sum(dim=-1)[...,None].sum(dim=0) / (mask.sum(dim=0) + eps)
        const = (gumbel.log_prob(rand_noise) * filtered).sum(dim=-1)[..., None].sum(dim=0) / (filtered.sum(dim=0) + eps) 
        return -(nll - const) / pred.size(0)
        '''
    @staticmethod
    def backward(ctx, grad_output):  
        #if isinstance(ctx.phi, oracle.Left_Distribution):
        #    import pdb; pdb.set_trace()
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
        # noised_labs = softmax(stacked).argmax(-1)
        noised_labs = noised.argmax(-1)
        # mask takes care of invalid logits and truncation set
        mask = noised_labs.eq(targ)[..., None] * filtered
        inner_exp = (1 - ch.exp(-rand_noise))
        nll = ((inner_exp * mask).sum(0) / (mask.sum(0) + ctx.eps))
        const = ((inner_exp * filtered).sum(0) / (filtered.sum(0) + ctx.eps))
        return (-nll + const) / pred.size(0), None, None, None, None


class TruncatedBooleanProductNLL(ch.autograd.Function):
    """
    Computes the truncated negative population log likelihood for a truncated boolean product distribution. 
    Function calculates the truncated negative log likelihood in the forward method and then calculates the 
    gradients with respect to p in the backward method. When sampling from the conditional distribution, 
    we sample batch_size * num_samples samples, we then filter out the samples that remain in the truncation set, 
    and retainup to batch_size of the filtered samples. If there are fewer than batch_size number of samples remain,
    we provide a vector of zeros and calculate the untruncated log likelihood. 
    """
    @staticmethod
    def forward(ctx, z, x, phi, num_samples=10):
        """
        Args: 
            p (torch.Tensor): current logit probability vector estimate for truncated boolean product distribution
            x (torch.Tensor): batch_size * dims, sample batch 
            phi (delphi.oracle): oracle for truncated boolean product distribution
            num_samples (int): number of samples to sample for each sample in batch
        """
        # reparameterize distribution
        B = Bernoulli(logits=z)
        # sample num_samples * batch size samples from distribution
        s = B.sample([num_samples * x.size(0)])
        filtered = phi(s).nonzero(as_tuple=True)
        # z is a tensor of size batch size zeros, then fill with up to batch size num samples
        y = ch.zeros(x.size())
        elts = s[filtered][:x.size(0)]
        y[:elts.size(0)] = elts
        # standard negative log likelihood
        nll = -x*z
        const = ch.log(ch.exp(y*z).sum(0))
        ctx.save_for_backward(x, y)
        return (nll + const) / x.size(0)

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        # calculate gradient
        return (-x + y) / x.size(0), None, None, None, None


class TruncatedLASSOMSE(ch.autograd.Function):
    """
    Computes the gradient of the negative population log likelihood for truncated regression
    with known noise variance.
    """
    @staticmethod
    def forward(ctx, pred, targ, phi, noise_var, model, num_samples=10, eps=1e-5):
        """
        Args: 
            pred (torch.Tensor): size (batch_size, 1) matrix for regression model predictions
            targ (torch.Tensor): size (batch_size, 1) matrix for regression target predictions
            phi (oracle.oracle): dependent variable membership oracle
            noise_var (float): noise distribution variance parameter
            num_samples (int): number of samples to generate per sample in batch in rejection sampling procedure
            eps (float): denominator error constant to avoid divide by zero errors
        """
        # make num_samples copies of pred, N x B x 1
        stacked = pred[None, ...].repeat(num_samples, 1, 1)
        # add random noise to each copy
        noised = stacked + math.sqrt(noise_var) * ch.randn(stacked.size())        
        # filter out copies where pred is in bounds
        filtered = phi(noised)
        # average across truncated indices
        z = (filtered * noised).sum(dim=0) / (filtered.sum(dim=0) + eps)
        out = ((-.5 * noised.pow(2) + noised * pred) * filtered).sum(dim=0) / (filtered.sum(dim=0) + eps)

        ctx.save_for_backward(pred, targ, z)
        ctx.model = model
        return (-.5 * targ.pow(2) + targ * pred - out).mean(0)

    @staticmethod
    def backward(ctx, grad_output):
        pred, targ, z = ctx.saved_tensors
        model = ctx.model
        import pdb; pdb.set_trace()
        return (z - targ) / pred.size(0), targ / pred.size(0), None, None, None, None