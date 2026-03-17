# Author: pstefanou12@
"""Parent class for all distribution models."""

from typing import Generic

from delphi.delphi import delphi
from delphi.utils import configs


class distributions(delphi[configs._ConfigT], Generic[configs._ConfigT]):  # pylint: disable=invalid-name,abstract-method
    """Parent class for distribution models."""

    def __init__(self, hyperparameters: dict):
        """Initialize the distributions model.

        Args:
            hyperparameters: Hyperparameter dict, validated via the subclass
                defaults config.
        """
        super().__init__(hyperparameters)
