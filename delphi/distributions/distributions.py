from ..delphi import delphi
from delphi.utils.helpers import Parameters


class distributions(delphi):
    """
    Parent class for distribution models.
    """
    def __init__(self, 
            args: Parameters): 
        '''
        Args: 
            args (cox.utils.Parameters) : parameter object holding hyperparameters
            k (int): number of output logits
        '''
        super().__init__(args)