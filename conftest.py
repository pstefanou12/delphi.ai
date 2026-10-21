# Author: pstefanou12@
"""Pytest configuration shared by the whole suite."""

import torch

# The fits run on tensors too small for intra-op threads to pay off: one
# thread makes a fit about 20% faster, and leaves the cores to test workers.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
