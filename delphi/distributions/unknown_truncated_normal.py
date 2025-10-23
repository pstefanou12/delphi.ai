"""
Truncated normal distribution without oracle access (ie. unknown truncation set)
"""

from re import A
from torch import Tensor
from typing import Callable

from .unknown_truncated_multivariate_normal import UnknownTruncationMultivariateNormal 
from ..trainer import Trainer
from ..utils.helpers import PSDError, Parameters
from ..utils.datasets import UnknownTruncationNormalDataset, make_train_and_val_distr


class UnknownTruncationNormal(UnknownTruncationMultivariateNormal):
    """
    Truncated normal distribution class.
    """
    def __init__(self,
                 phi: Callable,
                 alpha: float,
                 args: Parameters):
        super().__init__(args)

    # def fit(self, S: Tensor):
    #     """
    #     Runs PSGD to minimize the negative population log likelihood of the truncated distribution.
    #     Args: 
    #         S (torch.Tensor): censored samples dataset, expecting n by d matrix (num_samples by dimensions)
    #     """
    #     assert isinstance(S, Tensor), "S is type: {}. expected type torch.Tensor.".format(type(S))
    #     assert S.size(1) == 1, "truncated normal class only accepts 1 dimensional distributions." 
        
    #     while True:
    #         try:
    #             self.train_loader_, self.val_loader_ = make_train_and_val_distr(self.args, S, UnknownTruncationNormalDataset)
    #             self.unknown_truncated = UnknownTruncationMultivariateNormalModel(self.args, self.train_loader_.dataset)
    #             # run PGD to predict actual estimates
    #             self.trainer = Trainer(self.unknown_truncated, self.args, store=self.store)
        
    #             # run PGD for parameter estimation 
    #             # self.trainer.train_model(self.train_loader_, self.val_loader_)
    #             best_params, history, params = self.trainer.train_model(self.train_loader_, self.val_loader_)
    #             print(f"best params: {best_params}")
    #             print(f"history: {history}")
    #             print(f"params: {params}")
    #             return self
    #         except PSDError as psd:
    #             print(psd.message) 
    #             continue
    #         except Exception as e: 
    #                 raise e
    
    @property 
    def variance_(self): 
        """
        Returns the standard deviation for the normal distribution.
        """
        return self.unknown_truncated.model.covariance_matrix.clone()
