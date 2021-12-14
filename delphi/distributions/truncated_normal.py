"""
Truncated normal distribution without oracle access (ie. unknown truncation set)
"""

from torch import Tensor
import cox

from .truncated_multivariate_normal import TruncatedMultivariateNormal, TruncatedMultivariateNormalModel
from ..trainer import Trainer
from ..utils.helpers import PSDError
from ..utils.datasets import TruncatedNormalDataset, make_train_and_val_distr

# CONSTANTS 
DEFAULTS = {
        'alpha': (float, 'required'), 
        'epochs': (int, 1),
        'trials': (int, 1),
        'clamp': (bool, True), 
        'val': (float, .2),
        'lr': (float, 1e-1), 
        'step_lr': (int, 100),
        'step_lr_gamma': (float, .9), 
        'custom_lr_multiplier': (str, None), 
        'momentum': (float, 0.0), 
        'weight_decay': (float, 0.0), 
        'l1': (float, 0.0), 
        'eps': (float, 1e-5),
        'r': (float, 1.0), 
        'rate': (float, 1.5), 
        'batch_size': (int, 10),
        'tol': (float, 1e-1),
        'workers': (int, 0),
        'num_samples': (int, 10),
        'covariance_matrix': (Tensor, None), 
        'd': (int, 100),
        'early_stopping': (bool, False), 
        'n_iter_no_change': (int, 5),
        'verbose': (bool, False),
}


class TruncatedNormal(TruncatedMultivariateNormal):
    """
    Truncated normal distribution class.
    """
    def __init__(self,
            args: dict, 
            store: cox.store.Store=None):
        super().__init__(args, store=store)

    def fit(self, S: Tensor):
        """
        Runs PSGD to minimize the negative population log likelihood of the truncated distribution.
        Args: 
            S (torch.Tensor): censored samples dataset, expecting n by d matrix (num_samples by dimensions)
        """
        assert isinstance(S, Tensor), "S is type: {}. expected type torch.Tensor.".format(type(S))
        assert S.size(1) == 1, "truncated normal class only accepts 1 dimensional distributions." 
        
        while True:
            try:
                self.train_loader_, self.val_loader_ = make_train_and_val_distr(self.args, S, TruncatedNormalDataset)
                self.truncated = TruncatedMultivariateNormalModel(self.args, self.train_loader_.dataset)
                # run PGD to predict actual estimates
                self.trainer = Trainer(self.truncated, self.args, store=self.store)
        
                # run PGD for parameter estimation 
                self.trainer.train_model((self.train_loader_, self.val_loader_))
                return self
            except PSDError as psd:
                print(psd.message) 
                continue
            except Exception as e: 
                    raise e
     #    # rescale/standardize
    #    self.truncated.model.covariance_matrix.data = self.truncated.model.covariance_matrix @ self.emp_covariance_matrix
    #    self.truncated.model.loc.data = (self.truncated.model.loc[None,...] @ Tensor(sqrtm(self.emp_covariance_matrix.numpy()))).flatten() + self.emp_loc
    #
    @property 
    def variance(self): 
        """
        Returns the standard deviation for the normal distribution.
        """
        return self.truncated.model.covariance_matrix.clone()
