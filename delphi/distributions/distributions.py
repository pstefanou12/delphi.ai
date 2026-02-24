"""Parent class for all distribution models."""

from delphi.delphi import delphi
from delphi.delphi_logger import delphiLogger
from delphi.utils.helpers import Parameters


class distributions(delphi):  # pylint: disable=invalid-name,abstract-method
    """
    Parent class for distribution models.
    """

    def __init__(self, args: Parameters, logger: delphiLogger):
        """
        Args:
            args (cox.utils.Parameters) : parameter object holding hyperparameters
            k (int): number of output logits
        """
        super().__init__(args, logger)
