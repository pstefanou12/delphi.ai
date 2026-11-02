# Author: pstefanou12@
"""DataLoader factories and datasets for the statistical estimators."""

from collections.abc import Sized
from typing import cast

import torch as ch
import torch.linalg as LA
import torch.utils.data

from delphi.configs import dataset as dataset_configs


class Normalize:
    """
    Normalizes the input covariate features for truncated
    regression.
    """

    def __init__(self):
        """
        Args:
            X (torch.Tensor): regression input features; shape expected to be
                n (number of samples) by d (number of dimensions)
        """
        super().__init__()
        self._l_inf, self._beta = None, None

    def fit_transform(self, X):  # pylint: disable=invalid-name
        """
        Normalize input features truncated regression
        """
        # normalize input features
        self._l_inf = LA.norm(X, dim=-1, ord=float("inf")).max()  # pylint: disable=not-callable
        self._beta = self._l_inf * (X.size(1) ** 0.5)
        return self

    def transform(self, X):  # pylint: disable=invalid-name
        """Apply the stored normalization to X."""
        return X / self._beta

    @property
    def beta(self):
        """Return the normalization factor beta."""
        return self._beta

    @property
    def l_inf(self):  # pylint: disable=invalid-name
        """Return the L-infinity norm used for normalization."""
        return self._l_inf


def make_train_and_val(  # pylint: disable=invalid-name
    args: dataset_configs.DatasetConfig, X, y
):
    """Create training and validation DataLoaders from tensors X and y.

    Args:
        args: The estimator's config.
        X: Input covariates.
        y: Targets.

    Returns:
        The training and validation DataLoaders.
    """
    # separate into training and validation set
    val = int(args.val * X.size(0))

    x_train, y_train = X[val:], y[val:]  # pylint: disable=invalid-name
    x_val, y_val = X[:val], y[:val]  # pylint: disable=invalid-name
    # normalize input covariates
    if args.normalize:
        train_norm = Normalize().fit_transform(x_train)
        x_train = train_norm.transform(x_train)
        val_norm = Normalize().fit_transform(x_val)
        x_val = val_norm.transform(x_val)

    train_ds = torch.utils.data.TensorDataset(x_train, y_train)
    val_ds = torch.utils.data.TensorDataset(x_val, y_val)

    batch_size = len(train_ds) if args.batch_size == -1 else args.batch_size

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, num_workers=args.workers, shuffle=args.shuffle
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=len(val_ds), num_workers=args.workers, shuffle=args.shuffle
    )

    return train_loader, val_loader


def make_train_and_val_distr(  # pylint: disable=invalid-name
    args: dataset_configs.DatasetConfig, S, ds, kwargs=None
):
    """Create training and validation DataLoaders from a distribution dataset S."""
    if kwargs is None:
        kwargs = {}
    # Separate into training and validation sets.
    rand_indices = ch.randperm(S.size(0))
    val = int(args.val * S.size(0))
    train_indices, val_indices = rand_indices[val:], rand_indices[:val]
    x_train, x_val = S[train_indices], S[val_indices]  # pylint: disable=invalid-name
    train_ds = ds(x_train, **kwargs)
    val_ds = ds(x_val, **kwargs)
    batch_size = len(train_ds) if args.batch_size == -1 else args.batch_size
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=len(val_ds))

    return train_loader, val_loader


class TruncatedExponentialDistributionDataset(ch.utils.data.Dataset):
    """Dataset for truncated exponential distribution samples and sufficient statistics."""

    def __init__(self, S, calc_suff_stat):  # pylint: disable=invalid-name
        """Initialize with samples S and a sufficient statistic function."""
        self.S = S  # pylint: disable=invalid-name
        self.calc_suff_stat = calc_suff_stat
        # precalculate dataset score, so that it doesn't need to be computed within gradient
        self.S_grad = self.calc_suff_stat(S)  # pylint: disable=invalid-name
        self.data = self.S
        self.data = ch.cat([self.S, self.S_grad], dim=1)

    def __len__(self):
        return self.S.size(0)

    def __getitem__(self, idx):
        """Return a (dummy, data) pair for sample at idx."""
        return [
            ch.empty([]),
            self.data[idx],
        ]


class UnknownTruncationMultivariateNormalDataset(ch.utils.data.Dataset):
    """Samples of a normal under unknown truncation, as the objective's rows.

    Each row is ``[T(z), c(z), psi_k(z)]`` for one whitened sample z,
    computed once here, so a step only evaluates
    ``psi_k(z) exp(<theta, T(z)> + c(z))``; the Hermite expansion is never
    re-evaluated during the descent.
    """

    def __init__(self, statistic: ch.Tensor, offset: ch.Tensor, psi: ch.Tensor):
        """Initialize from the precomputed pieces of each row.

        Args:
            statistic: ``T(z)``, shaped (n, p).
            offset: ``c(z)``, shaped (n,).
            psi: ``psi_k(z)``, shaped (n,).
        """
        self.data = ch.cat([statistic, offset[:, None], psi[:, None]], dim=1)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        """Return a (dummy, rows) pair; idx may be a list, for a whole batch."""
        return [ch.empty([]), self.data[idx]]


def batch_loader(
    dataset: ch.utils.data.Dataset, batch_size: int, shuffle: bool
) -> ch.utils.data.DataLoader:
    """Return a loader that fetches each batch with one index.

    A plain DataLoader fetches rows one at a time and stacks them, which made
    a full-batch step over ten thousand rows take a tenth of a second. Any
    dataset whose ``__getitem__`` accepts a list of indices can use this.

    Args:
        dataset: The dataset, indexable by a list of indices.
        batch_size: Rows per batch.
        shuffle: Draw the rows in random order.

    Returns:
        The loader.
    """
    sized = cast(Sized, dataset)
    sampler: ch.utils.data.Sampler = (
        ch.utils.data.RandomSampler(sized)
        if shuffle
        else ch.utils.data.SequentialSampler(sized)
    )
    return ch.utils.data.DataLoader(
        dataset,
        sampler=ch.utils.data.BatchSampler(sampler, batch_size, drop_last=False),
        batch_size=None,
    )
