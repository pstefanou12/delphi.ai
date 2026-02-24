"""
Module containing all the supported datasets, which are subclasses of the
abstract class :class:`robustness.datasets.DataSet`.
Currently supported datasets:
- CIFAR-10 (:class:`delphi.datasets.CIFAR`)
"""

import numpy as np

import torch as ch
import torch.linalg as LA
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch.distributions.multivariate_normal import MultivariateNormal
from torchvision import datasets

from . import folder
from .helpers import cov
from .defaults import DATASET_DEFAULTS, check_and_fill_args
from . import data_augmentation as da
from .. import cifar_models
from .. import imagenet_models


###
# Computer Vision Datasets: (all subclassed from dataset)
# In order:
## - CIFAR
## - Imagenet
###

# required and optional arguments for datasets
CNN_REQUIRED_ARGS = ["num_classes", "mean", "std", "transform_train", "transform_test"]
CNN_OPTIONAL_ARGS = ["custom_class", "label_mapping", "custom_class_args"]


(
    CENSORED_MULTIVARIATE_NORMAL_REQUIRED_ARGS,
    CENSORED_MULTIVARIATE_NORMAL_OPTIONAL_ARGS,
) = (
    ["custom_class", "custom_class_args"],
    ["label_mapping", "transform_train", "transform_test"],
)
(
    TRUNCATED_MULTIVARIATE_NORMAL_REQUIRED_ARGS,
    TRUNCATED_MULTIVARIATE_NORMAL_OPTIONAL_ARGS,
) = (
    ["custom_class", "custom_class_args"],
    ["label_mapping", "transform_train", "transform_test"],
)
TENSOR_REQUIRED_ARGS, TENSOR_OPTIONAL_ARGS = (
    ["custom_class", "custom_class_args"],
    ["label_mapping", "transform_train", "transform_test"],
)


class DataSet:
    """
    Base class for representing a dataset. Meant to be subclassed, with
    subclasses implementing the `get_model` function.
    """

    def __init__(self, ds_name, required_args, optional_args, data_path=None, **kwargs):
        """
        Args:
            ds_name (str) : string identifier for the dataset
            data_path (str) : *optional argument, but required for CNNs* path to the dataset
            num_classes (int) : *required kwarg for CNN*, the number of classes in
                the dataset
            mean (ch.tensor) : *required kwarg for CNN*, the mean to normalize the
                dataset with (e.g.  :samp:`Tensor([0.4914, 0.4822,
                0.4465])` for CIFAR-10)
            std (ch.tensor) : *required kwarg for CNN*, the standard deviation to
                normalize the dataset with (e.g. :samp:`Tensor([0.2023,
                        0.1994, 0.2010])` for CIFAR-10)

            custom_class (type) : *required kwarg*, a
                :samp:`torchvision.models` class corresponding to the
                dataset, if it exists (otherwise :samp:`None`)
            label_mapping (dict[int,str]) : *required kwarg for CNN*, a dictionary
                mapping from class numbers to human-interpretable class
                names (can be :samp:`None`)
            transform_train (torchvision.transforms) : *required kwarg*,
                transforms to apply to the training images from the
                dataset
            transform_test (torchvision.transforms) : *required kwarg*,
                transforms to apply to the validation images from the
                dataset
        """
        missing_args = set(required_args) - set(kwargs.keys())
        if len(missing_args) > 0:
            raise ValueError(f"Missing required args {missing_args}")

        extra_args = set(kwargs.keys()) - set(required_args + optional_args)
        if len(extra_args) > 0:
            raise ValueError(f"Got unrecognized args {extra_args}")
        final_kwargs = {k: kwargs.get(k, None) for k in required_args + optional_args}

        self.ds_name = ds_name
        self.data_path = data_path
        self.__dict__.update(final_kwargs)

    def override_args(self, default_args, kwargs):
        """
        Convenience method for overriding arguments. (Internal)
        """
        for k in kwargs:
            if k not in default_args:
                continue
            req_type = type(default_args[k])
            no_nones = (default_args[k] is not None) and (kwargs[k] is not None)
            if no_nones and (not isinstance(kwargs[k], req_type)):
                raise ValueError(f"Argument {k} should have type {req_type}")
        return {**default_args, **kwargs}

    def get_model(self, arch, pretrained):
        """
        Should be overriden by subclasses. Also, you will probably never
        need to call this function, and should instead by using
        `model_utils.make_and_restore_model </source/delphi.utils.model_utils.html>`_.
        Args:
            arch (str) : name of architecture
            pretrained (bool): whether to try to load torchvision
                pretrained checkpoint
        Returns:
            A model with the given architecture that works for each
            dataset (e.g. with the right input/output dimensions).
        """
        raise NotImplementedError

    def make_loaders(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-branches
        self,
        workers,
        batch_size,
        data_aug=True,
        subset=None,
        subset_type="rand",
        subset_start=0,
        val_batch_size=None,
        train=True,
        val=True,
        shuffle=True,
        seed=1,
        verbose=True,  # pylint: disable=unused-argument
    ):
        """
        Args:
            workers (int) : number of workers for data fetching (*required*).
                batch_size (int) : batch size for the data loaders (*required*).
            data_aug (bool) : whether or not to do train data augmentation.
            subset (None|int) : if given, the returned training data loader
                will only use a subset of the training data; this should be a
                number specifying the number of training data points to use.
            subset_start (int) : only used if `subset` is not None; this specifies the
                starting index of the subset.
            subset_type ("rand"|"first"|"last") : only used if `subset is
                not `None`; "rand" selects the subset randomly, "first"
                uses the first `subset` images of the training data, and
                "last" uses the last `subset` images of the training data.
            seed (int) : only used if `subset == "rand"`; allows one to fix
                the random seed used to generate the subset (defaults to 1).
            val_batch_size (None|int) : if not `None`, specifies a
                different batch size for the validation set loader.
            only_val (bool) : If `True`, returns `None` in place of the
                training data loader
            shuffle (bool) : Whether or not to shuffle the training data
                in the returned DataLoader.
        Returns:
            A training loader and validation loader according to the
            parameters given. These are standard PyTorch data loaders, and
            thus can just be used via:
            >>> train_loader, val_loader = ds.make_loaders(workers=8, batch_size=128)
            >>> for im, lab in train_loader:
            >>>     # Do stuff...
        """
        # check that at least a train loader or validation loader specified to be created
        if not train and not val:
            raise ValueError("Neither training loader nor validation loader specified")
        # initialize loader variables
        train_set, test_set = None, None
        train_loader, test_loader = None, None

        if not val_batch_size:
            val_batch_size = batch_size

        custom_class = self.__dict__.get("custom_class")
        transform_train = self.__dict__.get("transform_train")
        transform_test = self.__dict__.get("transform_test")
        label_mapping = self.__dict__.get("label_mapping")

        if not custom_class:
            if train:
                train_set = folder.ImageFolder(
                    root=self.data_path,
                    transform=transform_train if data_aug else transform_test,
                    label_mapping=label_mapping,
                )
            if val:
                test_set = folder.ImageFolder(
                    root=self.data_path,
                    transform=transform_test,
                    label_mapping=label_mapping,
                )

            train_sample_count = 0
            if train:
                attrs = ["samples", "train_data", "data"]
                vals = {attr: hasattr(train_set, attr) for attr in attrs}
                assert any(vals.values()), f"dataset must expose one of {attrs}"
                train_sample_count = len(
                    getattr(train_set, [k for k in vals if vals[k]][0])
                )

            if (
                (train and not val)
                and (subset is not None)
                and (subset <= train_sample_count)
            ):
                assert train and not val
                if subset_type == "rand":
                    rng = np.random.RandomState(seed)  # pylint: disable=no-member
                    subset = rng.choice(
                        list(range(train_sample_count)),
                        size=subset + subset_start,
                        replace=False,
                    )
                    subset = subset[subset_start:]
                elif subset_type == "first":
                    subset = np.arange(subset_start, subset_start + subset)
                else:
                    subset = np.arange(train_sample_count - subset, train_sample_count)

                train_set = Subset(train_set, subset)
        else:
            custom_class_args = self.__dict__.get("custom_class_args")
            if custom_class_args is None:
                custom_class_args = {}
                self.__dict__["custom_class_args"] = custom_class_args
            if self.data_path is not None:
                if train:
                    train_set = custom_class(
                        root=self.data_path,
                        train=True,
                        download=True,
                        transform=transform_train,
                        **custom_class_args,
                    )
                if val:
                    test_set = custom_class(
                        root=self.data_path,
                        train=False,
                        download=True,
                        transform=transform_test,
                        **custom_class_args,
                    )
        if train_set is not None:
            train_loader = DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=workers,
                pin_memory=True,
            )

        if test_set is not None:
            test_loader = DataLoader(
                test_set,
                batch_size=val_batch_size,
                num_workers=workers,
                pin_memory=True,
            )

        return train_loader, test_loader


class CIFAR(DataSet):
    """
    CIFAR-10 dataset [Kri09]_.
    A dataset with 50k training images and 10k testing images, with the
    following classes:
    * Airplane
    * Automobile
    * Bird
    * Cat
    * Deer
    * Dog
    * Frog
    * Horse
    * Ship
    * Truck
    .. [Kri09] Krizhevsky, A (2009). Learning Multiple Layers of Features
        from Tiny Images. Technical Report.
    """

    def __init__(self, data_path="/tmp/", **kwargs):
        """Initialize CIFAR-10 dataset with default transforms and settings."""
        ds_kwargs = {
            # 'data_path': data_path,
            "num_classes": 10,
            "mean": ch.tensor([0.4914, 0.4822, 0.4465]),
            "std": ch.tensor([0.2023, 0.1994, 0.2010]),
            "custom_class": datasets.CIFAR10,
            "label_mapping": None,
            "transform_train": da.TRAIN_TRANSFORMS_DEFAULT(32),
            "transform_test": da.TEST_TRANSFORMS_DEFAULT(32),
        }
        ds_kwargs = self.override_args(ds_kwargs, kwargs)
        super().__init__(
            "cifar",
            CNN_REQUIRED_ARGS,
            CNN_OPTIONAL_ARGS,
            data_path=data_path,
            **ds_kwargs,
        )

    def get_model(self, arch, pretrained, args=None):  # pylint: disable=arguments-differ,unused-argument
        """Return a CIFAR model for the given architecture."""
        if pretrained:
            raise ValueError("CIFAR does not support pytorch_pretrained=True")
        num_classes = self.__dict__.get("num_classes")
        return cifar_models.__dict__[arch](num_classes=num_classes)


class ImageNet(DataSet):
    """
    ImageNet Dataset [DDS+09]_.
    Requires ImageNet in ImageFolder-readable format.
    ImageNet can be downloaded from http://www.image-net.org. See
    `here <https://pytorch.org/docs/master/torchvision/datasets.html
    #torchvision.datasets.ImageFolder>`_ for more information about the format.
    .. [DDS+09] Deng, J., Dong, W., Socher, R., Li, L., Li, K., & Fei-Fei, L. (2009).
        ImageNet: A large-scale hierarchical image database. 2009 IEEE Conference on
        Computer Vision and Pattern Recognition, 248-255.
    """

    def __init__(self, data_path, **kwargs):
        """Initialize ImageNet dataset with default transforms and settings."""
        ds_kwargs = {
            "num_classes": 1000,
            "mean": ch.tensor([0.485, 0.456, 0.406]),
            "std": ch.tensor([0.229, 0.224, 0.225]),
            "custom_class": None,
            "label_mapping": None,
            "transform_train": da.TRAIN_TRANSFORMS_IMAGENET,
            "transform_test": da.TEST_TRANSFORMS_IMAGENET,
        }
        ds_kwargs = self.override_args(ds_kwargs, kwargs)
        super().__init__(
            "imagenet", CNN_REQUIRED_ARGS, CNN_OPTIONAL_ARGS, data_path, **ds_kwargs
        )

    def get_model(self, arch, pretrained):
        """Return an ImageNet model for the given architecture."""
        num_classes = self.__dict__.get("num_classes")
        return imagenet_models.__dict__[arch](
            num_classes=num_classes, pretrained=pretrained
        )


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


def make_train_and_val(args, X, y):  # pylint: disable=invalid-name
    """Create training and validation DataLoaders from tensors X and y."""
    # check arguments are correct
    args = check_and_fill_args(args, DATASET_DEFAULTS)
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

    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(x_val, y_val)

    batch_size = len(train_ds) if args.batch_size == -1 else args.batch_size

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, num_workers=args.workers, shuffle=args.shuffle
    )
    val_loader = DataLoader(
        val_ds, batch_size=len(val_ds), num_workers=args.workers, shuffle=args.shuffle
    )

    return train_loader, val_loader


def make_train_and_val_distr(args, S, ds, kwargs=None):  # pylint: disable=invalid-name
    """Create training and validation DataLoaders from a distribution dataset S."""
    if kwargs is None:
        kwargs = {}
    # check arguments are correct
    args = check_and_fill_args(args, DATASET_DEFAULTS)
    # separate into training and validation set
    rand_indices = ch.randperm(S.size(0))
    val = int(args.val * S.size(0))
    train_indices, val_indices = rand_indices[val:], rand_indices[:val]
    x_train, x_val = S[train_indices], S[val_indices]  # pylint: disable=invalid-name
    train_ds = ds(x_train, **kwargs)
    val_ds = ds(x_val, **kwargs)
    batch_size = len(train_ds) if args.batch_size == -1 else args.batch_size
    train_loader = DataLoader(train_ds, batch_size=batch_size)
    val_loader = DataLoader(val_ds, batch_size=len(val_ds))

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


class UnknownTruncationNormalDataset(ch.utils.data.Dataset):
    """Dataset for normal distribution samples with unknown truncation."""

    def __init__(self, S):  # pylint: disable=invalid-name
        """Initialize with samples S, computing gradients for loc and covariance."""
        self.S = S  # pylint: disable=invalid-name
        # samples
        self._loc = self.S.mean(0)
        self._covariance_matrix = cov(self.S)
        # compute gradients
        # M = MultivariateNormal(self._loc, self._covariance_matrix)
        self.pdf = ch.exp(
            MultivariateNormal(
                ch.zeros(self.S.size(1)), ch.eye(self.S.size(1))
            ).log_prob(self.S)
        )[..., None]
        # self.pdf = ch.exp(M.log_prob(self.S))[...,None]

        self.loc_grad = self._loc - self.S
        self.cov_grad = 0.5 * (
            ch.bmm(self.S.unsqueeze(2), self.S.unsqueeze(1))
            - self._covariance_matrix
            - self._loc[..., None] @ self._loc[None, ...]
        ).flatten(1)

        self.data = ch.cat([self.S, self.pdf, self.loc_grad, self.cov_grad], dim=1)

    def __len__(self):
        return self.S.size(0)

    def __getitem__(self, idx):
        """
        :returns: (sample, sample pdf, sample mean coeffcient, sample covariance matrix coeffcient)
        """
        return [ch.empty([]), self.data[idx]]

    @property
    def loc(self):
        """Return a clone of the location (mean) parameter."""
        return self._loc.clone()

    @property
    def covariance_matrix(self):
        """Return a clone of the covariance matrix."""
        return self._covariance_matrix.clone()


DATASETS = {
    "imagenet": ImageNet,
    "cifar": CIFAR,
    "tensor": TensorDataset,
}
