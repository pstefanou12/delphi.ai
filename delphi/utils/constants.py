# Author: pstefanou12@
"""Constants used throughout the delphi library."""

# Module-level constants.
JUPYTER = "jupyter"
TERMINAL = "terminal"
IPYTHON = "ipython"

LOGS_SCHEMA = {
    "epoch": int,
    "val_loss": float,
    "train_loss": float,
    "time": float,
}

EVAL_LOGS_SCHEMA = {"test_loss": float, "time": float}

# Scheduler constants.
CYCLIC = "cyclic"
COSINE = "cosine"
LINEAR = "linear"

LOGS_TABLE = "logs"
EVAL_LOGS_TABLE = "eval"

CKPT_NAME = "checkpoint.pt"
BEST_APPEND = ".best"
CKPT_NAME_LATEST = CKPT_NAME + ".latest"
CKPT_NAME_BEST = CKPT_NAME + BEST_APPEND
