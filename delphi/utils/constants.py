# Author: pstefanou12@
"""Constants used throughout the delphi library."""

from enum import Enum

# Module-level constants.
JUPYTER = "jupyter"
TERMINAL = "terminal"
IPYTHON = "ipython"

# Base schema for the training logs table.  Model-specific metric columns
# (prefixed with "train_" or "val_") are appended dynamically at runtime.
LOGS_SCHEMA = {
    "epoch": int,
    "train_loss": float,
    "val_loss": float,
    "time": float,
}

EVAL_LOGS_SCHEMA = {"test_loss": float, "time": float}

LOGS_TABLE = "logs"
EVAL_LOGS_TABLE = "eval"

CKPT_NAME = "checkpoint.pt"
BEST_APPEND = ".best"
CKPT_NAME_LATEST = CKPT_NAME + ".latest"
CKPT_NAME_BEST = CKPT_NAME + BEST_APPEND


class ProcedureStage(str, Enum):
    """Training procedure stage identifiers."""

    TRAIN = "train"
    VAL = "val"


class StopReason(str, Enum):
    """Stop reasons recorded by the Trainer when a stop criterion fires."""

    GRAD_TOL = "grad_tol"
    LOSS_TOL = "loss_tol"
    EARLY_STOP = "early_stop"
    MAX_ITERATIONS = "max_iterations"
    MAX_EPOCHS = "max_epochs"
    MODEL_STOP = "model_stop"


class OptimizerType(str, Enum):
    """Built-in optimizer type identifiers."""

    SGD = "sgd"
    LBFGS = "lbfgs"
    ADAM = "adam"
    ADAMW = "adamw"


class SchedulerType(str, Enum):
    """Built-in learning-rate scheduler type identifiers."""

    CYCLIC = "cyclic"
    COSINE = "cosine"
    LINEAR = "linear"
    STEP = "step"
    MULTI_STEP = "multi_step"
    EXPONENTIAL = "exponential"
    REDUCE_ON_PLATEAU = "reduce_on_plateau"


class CheckpointKey(str, Enum):
    """Standard keys used in checkpoint dicts."""

    MODEL = "model"
    OPTIMIZER = "optimizer"
    SCHEDULER = "scheduler"
    EPOCH = "epoch"
    RANDOM_STATES = "random_states"
    TRAINING_STATE = "training_state"
    BEST_LOSS = "best_loss"


class RandomStateKey(str, Enum):
    """Keys for the random-state sub-dict stored in checkpoints."""

    PYTHON = "python"
    NUMPY = "numpy"
    PYTORCH = "pytorch"
