# Author: pstefanou12@
"""Dataset and DataLoader configuration.

These are loading concerns rather than model hyperparameters, which is how the
pre-pydantic code grouped them too (``DATASET_DEFAULTS`` in
``delphi.utils.defaults``). ``DatasetConfig`` is mixed into the concrete model
configs so the hyperparameter dict callers pass stays flat, while the fields
themselves are declared once and the loaders in ``delphi.utils.datasets`` can
be typed against them.
"""

import pydantic


class DatasetConfig(pydantic.BaseModel):
    """Configuration consumed by the delphi DataLoader factories.

    Attributes:
        val: Fraction of data held out for validation.
        workers: DataLoader worker processes.
        shuffle: Shuffle the training DataLoader.
        batch_size: Mini-batch size for training.
        normalize: Normalize input covariates before building the loaders.
    """

    model_config = pydantic.ConfigDict(extra="ignore")

    val: float = pydantic.Field(default=0.2, ge=0.0, le=1.0)
    workers: int = pydantic.Field(default=0, ge=0)
    shuffle: bool = True
    # -1 is a sentinel for "full batch", read by make_train_and_val as
    # len(dataset), so this is deliberately unconstrained.
    batch_size: int = 50
    normalize: bool = False
