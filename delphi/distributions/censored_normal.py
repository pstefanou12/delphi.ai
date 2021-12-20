"""
Censored normal distribution with oracle access (ie. known truncation set)
"""

from torch import Tensor
import cox

from .censored_multivariate_normal import CensoredMultivariateNormal, CensoredMultivariateNormalModel
from ..oracle import oracle
from ..trainer import Trainer
from ..utils.datasets import CensoredNormalDataset, make_train_and_val_distr
from ..utils.helpers import PSDError, Parameters 


class CensoredNormal(CensoredMultivariateNormal):
    """
    Censored normal distribution class.
    """
    def __init__(self, 
            args: Parameters,
            store: cox.store.Store=None):
        """
        Args:
           args (delphi.utils.helpers.Parameters): hyperparameters for censored algorithm 
        """
        super().__init__(args, store=store)

    def fit(self, S: Tensor):
        """
        Runs PSGD to minimize the negative population log likelihood of the censored distribution.
        Args: 
            S (torch.Tensor): censored samples dataset, expecting n by d matrix (num_samples by dimensions)
        """
        assert isinstance(S, Tensor), "S is type: {}. expected type torch.Tensor.".format(type(S))
        assert S.size(1) == 1, "censored normal class only accepts 1 dimensional distributions." 
        while True: 
            try:
                self.train_loader_, self.val_loader_ = make_train_and_val_distr(self.args, S, CensoredNormalDataset)
                self.censored = CensoredMultivariateNormalModel(self.args, self.train_loader_.dataset)
                # run PGD to predict actual estimates
                self.trainer = Trainer(self.censored, self.args, store=self.store)
                # run PGD for parameter estimation 
                self.trainer.train_model((self.train_loader_, self.val_loader_))
                return self 
            except PSDError as psd:
                print(psd.message) 
                continue
            except Exception as e: 
                raise e

    @property 
    def variance_(self): 
        """
        Returns the variance for the normal distribution.
        """
        return self.censored.model.covariance_matrix.clone()