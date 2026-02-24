# pylint: disable=anomalous-backslash-in-string
"""
For most use cases, this can just be considered an internal class and
ignored.**
This module contains the abstract class AttackerStep as well as a few subclasses.
AttackerStep is a generic way to implement optimizers specifically for use with
:class:`robustness.attacker.AttackerModel`. In general, except for when you want
to :ref:`create a custom optimization method <adding-custom-steps>`, you probably
do not need to import or edit this module and can just think of it as internal.
Cite: code for attack steps shamelessly taken from:
@misc{robustness,
   title={Robustness (Python Library)},
   author={Logan Engstrom and Andrew Ilyas and Hadi Salman
       and Shibani Santurkar and Dimitris Tsipras},
   year={2019},
   url={https://github.com/MadryLab/robustness}
}
"""

import torch as ch


class AttackerStep:
    """
    Generic class for attacker steps, under perturbation constraints
    specified by an "origin input" and a perturbation magnitude.
    Must implement project, step, and random_perturb
    """

    def __init__(self, orig_input, eps, step_size, use_grad=True):
        """
        Initialize the attacker step with a given perturbation magnitude.
        Args:
            eps (float): the perturbation magnitude
            orig_input (ch.tensor): the original input
        """
        self.orig_input = orig_input
        self.eps = eps
        self.step_size = step_size
        self.use_grad = use_grad

    def project(self, x):
        """
        Given an input x, project it back into the feasible set
        Args:
            ch.tensor x : the input to project back into the feasible set.
        Returns:
            A `ch.tensor` that is the input projected back into
            the feasible set, that is,
        .. math:: \min_{x' \in S} \|x' - x\|_2
        """
        raise NotImplementedError

    def step(self, x, g):
        """
        Given a gradient, make the appropriate step according to the
        perturbation constraint (e.g. dual norm maximization for :math:`\ell_p`
        norms).
        Parameters:
            g (ch.tensor): the raw gradient
        Returns:
            The new input, a ch.tensor for the next step.
        """
        raise NotImplementedError

    def random_perturb(self, x):
        """
        Given a starting input, take a random step within the feasible set
        """
        raise NotImplementedError

    def to_image(self, x):
        """
        Given an input (which may be in an alternative parameterization),
        convert it to a valid image (this is implemented as the identity
        function by default as most of the time we use the pixel
        parameterization, but for alternative parameterizations this functino
        must be overriden).
        """
        return x


### Instantiations of the AttackerStep class


# L-infinity threat model
class LinfStep(AttackerStep):
    """
    Attack step for :math:`\ell_\infty` threat model. Given :math:`x_0`
    and :math:`\epsilon`, the constraint set is given by:
    .. math:: S = \{x | \|x - x_0\|_\infty \leq \epsilon\}
    """

    def project(self, x):
        """Project x back into the Linf ball around orig_input."""
        if self.orig_input.is_cuda:
            x = x.cuda()
        diff = x - self.orig_input
        diff = ch.clamp(diff, -self.eps, self.eps)
        return ch.clamp(diff + self.orig_input, 0, 1)

    def step(self, x, g):
        """Take a sign gradient step of size step_size."""
        step = ch.sign(g) * self.step_size
        return x + step

    def random_perturb(self, x):
        """Apply a random Linf perturbation within eps."""
        new_x = x + 2 * (ch.rand_like(x) - 0.5) * self.eps
        return ch.clamp(new_x, 0, 1)


# L2 threat model
class L2Step(AttackerStep):
    """
    Attack step for :math:`\ell_\infty` threat model. Given :math:`x_0`
    and :math:`\epsilon`, the constraint set is given by:
    .. math:: S = \{x | \|x - x_0\|_2 \leq \epsilon\}
    """

    def project(self, x):
        """Project x back into the L2 ball around orig_input."""
        diff = x - self.orig_input
        diff = diff.renorm(p=2, dim=0, maxnorm=self.eps)
        return ch.clamp(self.orig_input + diff, 0, 1)

    def step(self, x, g):
        """Take a normalized gradient step of size step_size."""
        ndim = len(x.shape) - 1
        g_norm = ch.norm(g.view(g.shape[0], -1), dim=1).view(-1, *([1] * ndim))
        scaled_g = g / (g_norm + 1e-10)
        return x + scaled_g * self.step_size

    def random_perturb(self, x):
        """Apply a random L2 perturbation within eps."""
        ndim = len(x.shape) - 1
        rp = ch.randn_like(x)
        rp_norm = rp.view(rp.shape[0], -1).norm(dim=1).view(-1, *([1] * ndim))
        return ch.clamp(x + self.eps * rp / (rp_norm + 1e-10), 0, 1)


# Unconstrained threat model
class UnconstrainedStep(AttackerStep):
    """
    Unconstrained threat model, :math:`S = [0, 1]^n`.
    """

    def project(self, x):
        """Project x into [0,1]^n."""
        return ch.clamp(x, 0, 1)

    def step(self, x, g):
        """Take a gradient step of size step_size."""
        return x + g * self.step_size

    def random_perturb(self, x):
        """Apply a random perturbation of size step_size."""
        new_x = x + (ch.rand_like(x) - 0.5).renorm(p=2, dim=0, maxnorm=self.step_size)
        return ch.clamp(new_x, 0, 1)


class FourierStep(AttackerStep):
    """
    Step under the Fourier (decorrelated) parameterization of an image.
    See https://distill.pub/2017/feature-visualization/#preconditioning for more information.
    """

    def project(self, x):
        """No projection needed in Fourier parameterization."""
        return x

    def step(self, x, g):
        """Take a gradient step in Fourier space."""
        return x + g * self.step_size

    def random_perturb(self, x):
        """Return x unchanged (no random perturbation in Fourier space)."""
        return x

    def to_image(self, x):
        """Convert Fourier parameterization back to image space via sigmoid(irfft)."""
        return ch.sigmoid(ch.irfft(x, 2, normalized=True, onesided=False))


class RandomStep(AttackerStep):
    """
    Step for Randomized Smoothing.
    """

    def __init__(self, *args, **kwargs):
        """Initialize RandomStep with use_grad=False."""
        super().__init__(*args, **kwargs)
        self.use_grad = False

    def project(self, x):
        """Return x unchanged (no projection for random smoothing)."""
        return x

    def step(self, x, g):
        """Take a random Gaussian step of size step_size."""
        return x + self.step_size * ch.randn_like(x)

    def random_perturb(self, x):
        """Return x unchanged."""
        return x
