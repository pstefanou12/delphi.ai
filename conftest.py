# Author: pstefanou12@
"""Pytest configuration shared by the whole suite."""

import pytest
import torch

# The fits run on tensors too small for intra-op threads to pay off: one
# thread makes a fit about 20% faster, and leaves the cores to test workers.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add ``--coverage-sweep``, which runs the coverage_sweep tests."""
    parser.addoption(
        "--coverage-sweep",
        action="store_true",
        help="Also run the tests marked coverage_sweep.",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Deselect the coverage_sweep tests unless ``--coverage-sweep`` is given."""
    if config.getoption("--coverage-sweep"):
        return
    swept = [item for item in items if item.get_closest_marker("coverage_sweep")]
    if swept:
        config.hook.pytest_deselected(items=swept)
        items[:] = [item for item in items if item not in swept]
