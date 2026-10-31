# Author: pstefanou12@
"""Image datasets for adversarial training, ported from madrylab robustness.

Currently supported datasets:
- CIFAR-10 (:class:`delphi.adversarial.datasets.CIFAR`)
- ImageNet (:class:`delphi.adversarial.datasets.ImageNet`)
"""

import numpy as np
import torch as ch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets

from delphi.adversarial import cifar_models, folder, imagenet_models
from delphi.adversarial import data_augmentation as da

# required and optional arguments for datasets
CNN_REQUIRED_ARGS = ["num_classes", "mean", "std", "transform_train", "transform_test"]
CNN_OPTIONAL_ARGS = ["custom_class", "label_mapping", "custom_class_args"]


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
        `delphi.adversarial.model_utils.make_and_restore_model`.
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


DATASETS = {
    "imagenet": ImageNet,
    "cifar": CIFAR,
    "tensor": TensorDataset,
}
