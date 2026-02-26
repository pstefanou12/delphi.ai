# Author: pstefanou12@
"""
Multinomial logistic regression that uses gumbel max loss function.
"""
# pylint: disable=duplicate-code

from torch.distributions import Gumbel
from torch.nn import MSELoss

from delphi.stats.losses import GumbelCE
from delphi.utils.helpers import accuracy
from delphi.stats.linear_model import LinearModel

# Module-level constants.
mse_loss = MSELoss()
G = Gumbel(0, 1)


class GumbelCEModel(LinearModel):  # pylint: disable=abstract-method
    """Multinomial logistic regression using Gumbel max loss, for the trainer framework."""

    def __init__(self, args, d, k):  # pylint: disable=invalid-name
        """
        Args:
            args (Parameters): parameter object holding hyperparameters
            d (int): input dimension
            k (int): number of classes
        """
        super().__init__(args, d=d, k=k)  # pylint: disable=no-value-for-parameter,unexpected-keyword-arg

    def pretrain_hook(self):  # pylint: disable=attribute-defined-outside-init
        """Set up model parameters before training."""
        self.model.data = self.weight
        self.model.requires_grad = True
        self.params = [self.model]

    def predict(self, x):
        """Make class predictions using trained model."""
        stacked = (x @ self.model).repeat(self.args.num_samples, 1, 1)
        noised = stacked + G.sample(stacked.size())
        return noised.mean(0).argmax(-1)

    def __call__(self, batch):
        """
        Training step for the model.

        Args:
            batch (Iterable): iterable of (inputs, targets) pairs

        Returns:
            Tuple of (loss, prec1, prec5).
        """
        inp, targ = batch
        z = inp @ self.model
        loss = GumbelCE.apply(z, targ)

        # Calculate precision accuracies.
        prec1, prec5 = None, None
        if z.size(1) >= 5:
            prec1, prec5 = accuracy(z, targ, topk=(1, 5))
        else:
            (prec1,) = accuracy(z, targ, topk=(1,))
        return loss, prec1, prec5

    def calc_logits(self, inp):
        """Calculate logits from input."""
        return inp @ self.model

    def post_training_hook(self):
        """Post-training hook to freeze model parameters."""
        self.model.requires_grad = False
