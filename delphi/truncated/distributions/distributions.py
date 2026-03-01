# Author: pstefanou12@
"""Parent class for all distribution models."""

from pydantic import BaseModel

from delphi.delphi import delphi
from delphi.delphi_logger import delphiLogger


class distributions(delphi):  # pylint: disable=invalid-name,abstract-method
    """Parent class for distribution models."""

    def __init__(self, args: BaseModel, logger: delphiLogger):
        super().__init__(args)
        self.logger = logger
