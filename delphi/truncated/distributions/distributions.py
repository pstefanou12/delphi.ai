# Author: pstefanou12@
"""Parent class for all distribution models."""

import pydantic

from delphi import delphi_logger
from delphi.delphi import delphi


class distributions(delphi):  # pylint: disable=invalid-name,abstract-method
    """Parent class for distribution models."""

    def __init__(self, args: pydantic.BaseModel, logger: delphi_logger.delphiLogger):
        super().__init__(args)
        self.logger = logger
